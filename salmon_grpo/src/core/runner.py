# This script is based on https://github.com/salesforce/LAVIS/blob/main/lavis/runners/runner_base.py

import os
import json
import time
import datetime
from pathlib import Path
import logging

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from src.infrastructure.dist_utils import main_process, is_dist_avail_and_initialized, is_main_process, get_rank, get_world_size
from src.infrastructure.logger import MetricLogger, SmoothedValue
from src.infrastructure.utils import get_dataloader, prepare_sample
from src.infrastructure.optims import get_optimizer, LinearWarmupCosineLRScheduler

# Import new modular components
from src.training.grpo_trainer import GRPOTrainer
from src.training.validator import Validator
from src.training.rewards import RewardComputer
from src.util_classes.model_utils import ModelUtils
from src.util_classes.file_utils import FileUtils


class Runner:
    def __init__(self, cfg, model, datasets, job_id):
        self.config = cfg

        # log
        self.output_dir = Path(self.config.config.run.output_dir) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_writter = SummaryWriter(self.output_dir)

        # settings
        self.device = torch.device(self.config.config.run.device)
        self.use_distributed = self.config.config.run.use_distributed
        
        # Resume logic
        resume_path = self.config.config.run.get("resume_path", "")
        resume_epoch = 0
        resume_checkpoint = None
        
        if resume_path and os.path.exists(resume_path):
            if is_main_process():
                logging.info(f"Loading checkpoint from {resume_path}")
            resume_checkpoint = torch.load(resume_path, map_location="cpu")
            resume_epoch = resume_checkpoint.get("epoch", -1) + 1
            if is_main_process():
                logging.info(f"Resuming from epoch {resume_epoch}")

        self.start_epoch = resume_epoch
        
        # Manual override for start epoch (useful if loading weights via ckpt but want to skip seen data)
        manual_start_epoch = self.config.config.run.get("dataset_starting_epoch", None)
        if manual_start_epoch is not None:
            self.start_epoch = int(manual_start_epoch)
            if is_main_process():
                logging.info(f"Manually setting start epoch to {self.start_epoch} (affects dataset & training loop)")
                
        self.max_epoch = self.config.config.run.optims.max_epoch
        
        # If manually skipping data epochs, we might want scheduler to start from 0
        # Calculate offset to subtract from current_epoch when passing to scheduler
        self.scheduler_epoch_offset = 0
        if manual_start_epoch is not None and not resume_path:
             self.scheduler_epoch_offset = self.start_epoch
             if is_main_process():
                 logging.info(f"Scheduler will start from epoch 0 (offset by {self.scheduler_epoch_offset})")
        self.evaluate_only = self.config.config.run.evaluate
        self.cuda_enabled = (self.device.type == "cuda")
        self.epsilon = 1e-8
        
        # Checkpoint selection strategy: 'best_acc', 'best_loss', or 'last'
        self.checkpoint_strategy = self.config.config.run.get("checkpoint_strategy", "best_acc")

        # test prompt
        self.prompt_template = self.config.config.model.get("prompt_template", "")
        test_prompt_path = self.config.config.model.get("test_prompt_path", "")
        if test_prompt_path:
            try:
                with open(test_prompt_path, "r") as f:
                    self.test_prompt_dict = json.load(f)
            except:
                self.test_prompt_dict = None
        else:
            self.test_prompt_dict = None

        # model
        if self.use_distributed:
            model = model.cuda()
            model = DDP(model, device_ids=[self.config.config.run.gpu])
        self.model = model
        self._model = self.model.module if hasattr(self.model, "module") else self.model

        self.initial_state_dict = {}
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                self.initial_state_dict[name] = param.data.clone()
        logging.info(f"Saved {len(self.initial_state_dict)} trainable parameters for reference model")

        # optimizer
        self.optimizer = get_optimizer(self.model, self.config.config.run.optims)

        # dataloader
        self.train_loader = get_dataloader(
            datasets["train"], 
            self.config.config.run, 
            is_train=True, 
            use_distributed=self.use_distributed,
            start_epoch=self.start_epoch,
            shuffle_reasons=self.config.config.run.get("shuffle_reasons", False),
        )
        self.valid_loader = get_dataloader(datasets["valid"], self.config.config.run, is_train=False, use_distributed=self.use_distributed, shuffle_reasons=self.config.config.run.get("shuffle_reasons", False))
        self.test_loader = get_dataloader(datasets["test"], self.config.config.run, is_train=False, use_distributed=self.use_distributed, shuffle_reasons=self.config.config.run.get("shuffle_reasons", False))

        # Load states if resuming
        if resume_checkpoint:
            self.model_utils.unwrap_dist_model(self.model).load_state_dict(resume_checkpoint['model'])
            self.optimizer.load_state_dict(resume_checkpoint['optimizer'])

        self.iters_per_epoch = self.config.config.run.iters_per_epoch
        self.use_amp = self.config.config.run.amp
        if self.use_amp:
            # Use torch.amp.GradScaler if available (PyTorch 2.4+), else fallback
            if hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda")
            else:
                self.scaler = torch.cuda.amp.GradScaler()
            
            if resume_checkpoint and "scaler" in resume_checkpoint and resume_checkpoint["scaler"]:
                self.scaler.load_state_dict(resume_checkpoint["scaler"])
        else:
            self.scaler = None

        # scheduler
        scheduler_type = self.config.config.run.optims.get("scheduler", "cosine")
        scheduler_params = {
            "optimizer": self.optimizer,
            "max_epoch": self.max_epoch,
            "iters_per_epoch": self.iters_per_epoch,
            "min_lr": self.config.config.run.optims.min_lr,
            "init_lr": self.config.config.run.optims.init_lr,
            "warmup_start_lr": self.config.config.run.optims.warmup_start_lr,
            "warmup_steps": self.config.config.run.optims.warmup_steps,
        }

        if scheduler_type.lower() == "cosine":
            from src.infrastructure.optims import LinearWarmupCosineLRScheduler
            self.scheduler = LinearWarmupCosineLRScheduler(**scheduler_params)
        else:
            # Default to cosine
            from src.infrastructure.optims import LinearWarmupCosineLRScheduler
            self.scheduler = LinearWarmupCosineLRScheduler(**scheduler_params)
            logging.info(f"Unknown scheduler '{scheduler_type}', defaulting to Cosine LR scheduler")

        # Initialize modular components
        self.model_utils = ModelUtils(self.use_distributed)
        self.file_utils = FileUtils()
        self.validator = Validator(self.config, self.model_utils, self.file_utils, 
                                 self.output_dir, self.test_prompt_dict)
        
        # Initialize judge based on configuration
        judge_config = self.config.config.get("judge", {})
        judge_type = judge_config.get("type", "internal")
        
        if judge_type == "openrouter":
            from src.training.judges import OpenRouterJudge
            judge = OpenRouterJudge(judge_config.get("openrouter", {}))
            logging.info("Using OpenRouter judge")
        elif judge_type == "neighbor":
            from src.training.judges import NeighborJudge
            neighbor_config = judge_config.get("neighbor", {})
            judge = NeighborJudge(neighbor_config)
            logging.info(f"Using Neighbor judge (model: {neighbor_config.get('model_path', 'default')})")
        else:
            from src.training.judges import LocalJudge
            judge = LocalJudge(self._model)
            logging.info("Using internal model judge")
        
        # Get reward config
        reward_config = self.config.config.get("reward", {})
        
        # Backward compatibility: ensure judge_weight exists
        if "judge_weight" not in reward_config:
            reward_config["judge_weight"] = self.config.config.run.get("judge_weight", 0.0)

        self.reward_computer = RewardComputer(
            self.device, 
            self.validator,
            judge=judge,
            config=reward_config
        )
        self.grpo_trainer = GRPOTrainer(
            self._model, self.config, self.optimizer, self.scheduler, self.scaler,
            self.train_loader, self.output_dir, self.log_writter, self.reward_computer,
            self.epsilon,
            initial_state_dict=self.initial_state_dict,
            prompt_dict=self._model.prompt_dict,  # Pass prompts to GRPO trainer
            scheduler_epoch_offset=getattr(self, 'scheduler_epoch_offset', 0) # Pass offset
        )

        self.log_config()

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_loss = float('inf')
        best_epoch = 0

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if self.evaluate_only:
                break

            # training phase
            logging.info("Training Phase")
            train_stats = self.grpo_trainer.train_epoch(cur_epoch)
            self.log_stats(train_stats, split_name="train")

            # validating phase
            logging.info("Validating Phase")
            # Use decode=True to actually test generation quality
            valid_log = self.validator.valid_epoch(cur_epoch, "valid", self.model, 
                                                 self.valid_loader, decode=True, save_json=True)
            if valid_log is not None:
                if is_main_process():
                    agg_metrics = valid_log["agg_metrics"]
                    val_loss = valid_log["loss"]
                    
                    # Determine if this is the best checkpoint based on strategy
                    is_best = False
                    if self.checkpoint_strategy == "best_acc":
                        if agg_metrics > best_agg_metric:
                            best_agg_metric = agg_metrics
                            best_epoch = cur_epoch
                            is_best = True
                    elif self.checkpoint_strategy == "best_loss":
                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_epoch = cur_epoch
                            is_best = True
                    # For 'last' strategy, we don't save checkpoint_best.pth here
                    
                    if is_best:
                        self.save_checkpoint(cur_epoch, is_best=True)

                    valid_log.update({"best_epoch": best_epoch})
                    self.log_stats(valid_log, split_name="valid")

            self.save_checkpoint(cur_epoch, is_best=False)

            if self.use_distributed:
                dist.barrier()

        # Save final checkpoint as checkpoint_last.pth for 'last' strategy
        if is_main_process() and self.checkpoint_strategy == "last":
            final_epoch = self.max_epoch - 1
            self.save_checkpoint(final_epoch, is_best=True, name_override="last")
            logging.info(f"Saved final checkpoint as checkpoint_last.pth")

        # testing phase
        if self.evaluate_only:
            test_log = self.validator.valid_epoch("best", "test", self.model, 
                                                self.test_loader, decode=True, save_json=True)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
            f.flush()  # Force write to disk immediately

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
                f.flush()  # Force write to disk immediately
        elif isinstance(stats, list):
            pass

    @main_process
    def save_checkpoint(self, cur_epoch, is_best=False, name_override=None):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.model_utils.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        
        # Determine checkpoint name
        if name_override:
            checkpoint_name = name_override
        elif is_best:
            checkpoint_name = "best"
        else:
            checkpoint_name = str(cur_epoch)
            
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format(checkpoint_name),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)
