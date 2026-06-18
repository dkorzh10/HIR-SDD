import argparse
import glob
import logging
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Any, List, Optional, Union

from .utils.config import load_config

# Loggers
from .loggers.hard_label_logger import HardLabelLogger
from .loggers.reasoning_logger import ReasoningLogger
from .loggers.grpo_logger import GRPOLogger
from .loggers.distillation_logger import DistillationLogger

from .dataloaders.builder import get_dataloader

# Judges
from .judges.format_judge import FormatJudge
from .judges.openrouter_judge import OpenRouterJudge

from .trainers import SFTTrainer, GRPOTrainer, DistillationTrainer


def _resolve_pretrained_paths(experiment_dir: str, prefer_best_val_loss: bool = False):
    """Build config path and checkpoint path from an experiment directory.
    Checkpoint is chosen in order:
    - If prefer_best_val_loss: best by validation (sft_epoch_*_best.pt, exclude *_best_acc.pt), then latest.
    - Else: best_balanced_acc, then best by validation, then latest.
    File names must match trainers.base.save_checkpoint(name="sft"):
      - sft_best_acc.pt (symlink or file)
      - sft_epoch_*_best.pt (best by val; exclude sft_epoch_*_best_acc.pt)
      - sft_latest.pt
    Returns (config_path, ckpt_path). Raises FileNotFoundError if required files missing.
    """
    experiment_dir = os.path.abspath(experiment_dir)
    config_path = os.path.join(experiment_dir, "config_resolved.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Pretrained experiment config not found: {config_path}")

    ckpt_dir = os.path.join(experiment_dir, "checkpoints")

    if not prefer_best_val_loss:
        # 1) Prefer best-by-balanced-accuracy checkpoint (file or symlink)
        best_acc_path = os.path.join(ckpt_dir, "sft_best_acc.pt")
        if os.path.lexists(best_acc_path):
            return config_path, best_acc_path

    # 2) Prefer best by validation (sft_epoch_*_best.pt, excluding *_best_acc.pt)
    best_val_pattern = os.path.join(ckpt_dir, "sft_epoch_*_best.pt")
    for path in sorted(glob.glob(best_val_pattern)):
        if "_best_acc.pt" not in os.path.basename(path):
            return config_path, path

    # 3) Fallback to latest checkpoint
    latest_path = os.path.join(ckpt_dir, "sft_latest.pt")
    if not os.path.isfile(latest_path):
        raise FileNotFoundError(
            f"No suitable checkpoint found in {ckpt_dir} "
            "(tried sft_best_acc.pt, sft_epoch_*_best.pt, sft_latest.pt)"
        )
    return config_path, latest_path


class Runner:
    def __init__(
        self,
        config_path: str,
        run_type_override: Optional[str] = None,
        ckpt_override: Optional[str] = None,
        output_dir_override: Optional[str] = None,
        only_one_window: bool = False,
        test_dataset_path_override: Optional[str] = None,
        max_test_samples_override: Optional[int] = None,
    ):
        self._log = logging.getLogger(__name__)
        self.only_one_window = only_one_window
        print("config_path", config_path)
        self.config = load_config(config_path)
        if run_type_override is not None:
            self.config.setdefault("Runner", {})["type"] = run_type_override
        if ckpt_override is not None:
            self.config.setdefault("Model", {})["ckpt"] = ckpt_override
        if test_dataset_path_override is not None:
            self.config.setdefault("Datasets", {})["dataset_test_path"] = test_dataset_path_override
        if max_test_samples_override is not None:
            self.config.setdefault("Datasets", {})["max_test_samples"] = max_test_samples_override

        # Setup Distributed Training
        self.num_gpus = self.config.get("Runner", {}).get("num_gpus", 1)
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        if self.num_gpus > 1:
            if "RANK" in os.environ:
                if not dist.is_initialized():
                    dist.init_process_group(backend="nccl")
                torch.cuda.set_device(self.device)
            else:
                print(f"Warning: num_gpus={self.num_gpus} but distributed environment variables (RANK) not set. "
                      f"Falling back to single GPU training.", flush=True)
                self.num_gpus = 1

        # Determine output directory (override for pretrained test, else timestamped run dir)
        if output_dir_override is not None:
            self.output_dir = os.path.abspath(output_dir_override)
        else:
            base_output_dir = self.config.get("General", {}).get("output_dir", "./outputs")
            timestamp_str = "unknown"
            if self.local_rank == 0:
                from datetime import datetime
                timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
            if self.num_gpus > 1:
                obj_list = [timestamp_str]
                dist.broadcast_object_list(obj_list, src=0)
                timestamp_str = obj_list[0]
            self.output_dir = os.path.join(base_output_dir, f"run_{timestamp_str}")
        print("self.output_dir", self.output_dir)

        if self.local_rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            if output_dir_override is None:
                # Ensure config_resolved.yaml includes effective defaults (e.g. filter_controversial)
                # so the saved config always reflects what will be used
                config_to_save = self._resolve_config_for_save()
                print(f"--- Run Configuration ---", flush=True)
                import yaml
                print(yaml.dump(config_to_save, default_flow_style=False), flush=True)
                with open(os.path.join(self.output_dir, "config_resolved.yaml"), "w") as f:
                    yaml.dump(config_to_save, f, default_flow_style=False)
                print(f"-------------------------", flush=True)

        self.model = self._build_model()
        print("Model built")
        self.model.to(self.device)
        
        if self.num_gpus > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=False)

        self.logger = self._build_logger()
        print("Logger Created")
        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()
        print("Dataloaders Created")
        
    def cleanup(self):
        """Clean up distributed process group to avoid resource leaks."""
        if self.num_gpus > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def _resolve_config_for_save(self) -> Dict[str, Any]:
        """Return a copy of config with effective defaults merged in, so config_resolved.yaml
        always reflects what will actually be used (e.g. filter_controversial, skeptic_batch_size).
        """
        import copy
        cfg = copy.deepcopy(self.config)
        runner = cfg.setdefault("Runner", {})
        trainer_type = runner.get("trainer", "SFT")
        if trainer_type == "GRPO":
            grpo = runner.setdefault("GRPO", {})
            grpo.setdefault("filter_controversial", False)
            grpo.setdefault("skeptic_batch_size", None)
            grpo.setdefault("skeptic_buffer_size", None)
        return cfg

    def _build_model(self):
        model_cfg = self.config.get("Model", {})
        model_name = model_cfg.get("model_name")
        if model_name == "salmon":
            from .models.salmon import SalmonModel
            return SalmonModel(model_cfg)
        elif model_name == "conv_audio_classifier":
            from .models.conv_audio_classifier import ConvAudioClassifierModel
            return ConvAudioClassifierModel(model_cfg)
        raise ValueError(f"Unknown model name: {model_name}")

    def _build_logger(self):
        logger_cfg = self.config.get("Logger", {})
        log_dir = os.path.join(self.output_dir, "logs")
        log_freq = logger_cfg.get("log_freq", 10)
        
        runner_cfg = self.config.get("Runner", {})
        trainer_type = runner_cfg.get("trainer", "SFT")
        
        is_main_process = (self.local_rank == 0)

        if trainer_type == "GRPO":
            return GRPOLogger(log_dir, log_freq if is_main_process else 1000000)
        if trainer_type == "Distillation":
            return DistillationLogger(log_dir, log_freq if is_main_process else 1000000)

        task_type = runner_cfg.get("SFT", {}).get("task_type", "hard_label")
        if task_type == "reasoning":
            return ReasoningLogger(log_dir, log_freq if is_main_process else 1000000)
        else:
            return HardLabelLogger(log_dir, log_freq if is_main_process else 1000000)

    def _get_prompt_templates(self, task_type: str) -> Optional[List[str]]:
        """Load prompt templates from config based on task type.
        
        Supports two formats:
        - List of strings directly in config
        - Path to JSON file containing list of prompts (or dict with 'antispoofing' key)
        """
        prompts_cfg = self.config.get("Prompts", {})
        key = "reasoning_prompts" if task_type == "reasoning" else "hard_label_prompts"
        value = prompts_cfg.get(key)
        
        if value is None:
            return None
        
        # If it's already a list, return it
        if isinstance(value, list):
            return value
        
        # If it's a string, treat as file path and load JSON
        if isinstance(value, str):
            import json
            try:
                with open(value, 'r') as f:
                    prompts = json.load(f)
                    # Handle both list format and dict with 'antispoofing' key
                    if isinstance(prompts, list):
                        return prompts
                    if isinstance(prompts, dict) and "antispoofing" in prompts:
                        return prompts["antispoofing"]
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load prompts from {value}: {e}", flush=True)
        
        return None

    def _build_dataloaders(self):
        data_cfg = self.config.get("Datasets", {})
        runner_cfg = self.config.get("Runner", {})
        model_cfg = self.config.get("Model", {})
        model_name = model_cfg.get("model_name")
        
        whisper_path = None
        model_path = None
        conv_audio_vad_target_sec = None
        conv_audio_sample_rate = None
        if model_name == "salmon":
            whisper_path = model_cfg.get("additional_kwargs", {}).get("salmon", {}).get("pretrained_ckpts", {}).get("whisper_path")
        elif model_name == "conv_audio_classifier":
            cc = model_cfg.get("additional_kwargs", {}).get("conv_audio_classifier", {})
            conv_audio_vad_target_sec = cc.get("target_duration_sec")
            conv_audio_sample_rate = cc.get("sample_rate")

        trainer_type = runner_cfg.get("trainer", "SFT")
        iters_per_epoch = None
        iters_per_epoch_val = None
        train_iters_per_epoch = None  # for dataloader; may differ in skeptic mode
        accum_grad_iters = 1
        
        if trainer_type == "GRPO":
            task_type = "reasoning"
            grpo_cfg = runner_cfg.get("GRPO", {})
            iters_per_epoch = grpo_cfg.get("iters_per_epoch")
            iters_per_epoch_val = grpo_cfg.get("iters_per_epoch_val")
            accum_grad_iters = grpo_cfg.get("accum_grad_iters", 1)
            # Skeptic mode: epoch = GRPO batches, so we need enough dataloader batches to reach that.
            # Pass None so sampler yields full dataset; we stop when grpo_iter >= iters_per_epoch.
            train_iters_per_epoch = None if grpo_cfg.get("filter_controversial") else iters_per_epoch
        elif trainer_type == "Distillation":
            task_type = runner_cfg.get("SFT", {}).get("task_type", "reasoning")
            if "GRPO" in runner_cfg:
                task_type = "reasoning"  # Distillation uses GRPO flow
            sft_cfg = runner_cfg.get("SFT", {})
            iters_per_epoch = sft_cfg.get("iters_per_epoch")
            iters_per_epoch_val = sft_cfg.get("iters_per_epoch_val")
            accum_grad_iters = sft_cfg.get("accum_grad_iters", 1)
            train_iters_per_epoch = iters_per_epoch
        else:
            task_type = runner_cfg.get("SFT", {}).get("task_type", "hard_label")
            iters_per_epoch = runner_cfg.get("SFT", {}).get("iters_per_epoch")
            iters_per_epoch_val = runner_cfg.get("SFT", {}).get("iters_per_epoch_val")
            accum_grad_iters = runner_cfg.get("SFT", {}).get("accum_grad_iters", 1)
            train_iters_per_epoch = iters_per_epoch

        # Use batch sizes from the active trainer (configs often have both SFT and GRPO sections)
        if trainer_type == "GRPO" and "GRPO" in runner_cfg:
            batch_size_train = runner_cfg["GRPO"].get("batch_size_train", 1)
            batch_size_eval = runner_cfg["GRPO"].get("batch_size_eval", 1)
        elif trainer_type == "Distillation" and "SFT" in runner_cfg:
            batch_size_train = runner_cfg["SFT"].get("batch_size_train", 1)
            batch_size_eval = runner_cfg["SFT"].get("batch_size_eval", 1)
        elif "SFT" in runner_cfg:
            batch_size_train = runner_cfg["SFT"].get("batch_size_train", 1)
            batch_size_eval = runner_cfg["SFT"].get("batch_size_eval", 1)
        else:
            batch_size_train = 1
            batch_size_eval = 1
        
        # Load prompt templates based on task type
        prompt_templates = self._get_prompt_templates(task_type)
        grounding_cfg = None
        if trainer_type == "SFT" and task_type == "reasoning":
            grounding_cfg = data_cfg.get("grounding")
            
        # train_samples_offset with backward compat for samples_offset
        train_samples_offset = data_cfg.get("train_samples_offset", data_cfg.get("samples_offset", 0))
        val_samples_offset = data_cfg.get("val_samples_offset", 0)
        test_samples_offset = data_cfg.get("test_samples_offset", 0)
        use_length_grouped_sampler = data_cfg.get("use_length_grouped_sampler", False)
        sampler_seed = int(data_cfg.get("sampler_seed", 0))

        train_loader = get_dataloader(
            data_cfg.get("dataset_train_path", "dummy"), 
            batch_size_train, 
            shuffle=data_cfg.get("shuffle", False), 
            max_samples=data_cfg.get("max_train_samples"),
            task_type=task_type,
            whisper_path=whisper_path,
            model_name=model_name,
            model_path=model_path,
            num_workers=runner_cfg.get("num_workers", 4),
            distributed=(self.num_gpus > 1),
            iters_per_epoch=train_iters_per_epoch,
            prompt_templates=prompt_templates,
            samples_offset=train_samples_offset,
            grounding_cfg=grounding_cfg,
            use_length_grouped_sampler=use_length_grouped_sampler,
            sampler_seed=sampler_seed,
            shuffle_reasons=data_cfg.get("shuffle_reasons", False),
            conv_audio_vad_target_sec=conv_audio_vad_target_sec,
            conv_audio_sample_rate=conv_audio_sample_rate,
            max_audio_duration_s=data_cfg.get("max_audio_duration_s", 30.0),
            max_len_seconds=data_cfg.get("max_len_seconds", None),
        )
        val_loader = get_dataloader(
            data_cfg.get("dataset_val_path", "dummy"), 
            batch_size_eval, 
            shuffle=False, 
            max_samples=data_cfg.get("max_valid_samples"),
            task_type=task_type,
            whisper_path=whisper_path,
            model_name=model_name,
            model_path=model_path,
            num_workers=runner_cfg.get("num_workers", 4),
            distributed=(self.num_gpus > 1),
            iters_per_epoch=iters_per_epoch_val,
            prompt_templates=prompt_templates,
            samples_offset=val_samples_offset,
            use_length_grouped_sampler=use_length_grouped_sampler,
            sampler_seed=sampler_seed,
            shuffle_reasons=data_cfg.get("shuffle_reasons", False),
            conv_audio_vad_target_sec=conv_audio_vad_target_sec,
            conv_audio_sample_rate=conv_audio_sample_rate,
            max_audio_duration_s=data_cfg.get("max_audio_duration_s", 30.0),
            max_len_seconds=data_cfg.get("max_len_seconds", None),
        )
        test_loader = get_dataloader(
            data_cfg.get("dataset_test_path", "dummy"), 
            batch_size_eval, 
            shuffle=False, 
            max_samples=data_cfg.get("max_test_samples"),
            task_type=task_type,
            whisper_path=whisper_path,
            model_name=model_name,
            model_path=model_path,
            num_workers=runner_cfg.get("num_workers", 4),
            distributed=(self.num_gpus > 1),
            prompt_templates=prompt_templates,
            samples_offset=test_samples_offset,
            use_length_grouped_sampler=use_length_grouped_sampler,
            sampler_seed=sampler_seed,
            shuffle_reasons=data_cfg.get("shuffle_reasons", False),
            conv_audio_vad_target_sec=conv_audio_vad_target_sec,
            conv_audio_sample_rate=conv_audio_sample_rate,
            max_audio_duration_s=data_cfg.get("max_audio_duration_s", 30.0),
            max_len_seconds=data_cfg.get("max_len_seconds", None),
        )
        return train_loader, val_loader, test_loader

    def _create_dataloader_from_path(
        self, path: str, task_type: str = "reasoning"
    ) -> DataLoader:
        """Create a dataloader from a dataset JSON path (e.g. intermediate_dataset_iter_N.json)."""
        data_cfg = self.config.get("Datasets", {})
        runner_cfg = self.config.get("Runner", {})
        model_cfg = self.config.get("Model", {})
        model_name = model_cfg.get("model_name")
        whisper_path = None
        model_path = None
        conv_audio_vad_target_sec = None
        conv_audio_sample_rate = None
        if model_name == "salmon":
            whisper_path = model_cfg.get("additional_kwargs", {}).get("salmon", {}).get("pretrained_ckpts", {}).get("whisper_path")
        elif model_name == "conv_audio_classifier":
            cc = model_cfg.get("additional_kwargs", {}).get("conv_audio_classifier", {})
            conv_audio_vad_target_sec = cc.get("target_duration_sec")
            conv_audio_sample_rate = cc.get("sample_rate")
        batch_size_train = runner_cfg.get("SFT", {}).get("batch_size_train", 1)
        prompt_templates = self._get_prompt_templates(task_type)
        use_length_grouped_sampler = data_cfg.get("use_length_grouped_sampler", False)
        sampler_seed = int(data_cfg.get("sampler_seed", 0))
        return get_dataloader(
            path,
            batch_size_train,
            shuffle=data_cfg.get("shuffle", False),
            max_samples=None,
            task_type=task_type,
            whisper_path=whisper_path,
            model_name=model_name,
            model_path=model_path,
            num_workers=runner_cfg.get("num_workers", 4),
            distributed=(self.num_gpus > 1),
            iters_per_epoch=runner_cfg.get("SFT", {}).get("iters_per_epoch"),
            prompt_templates=prompt_templates,
            samples_offset=0,
            use_length_grouped_sampler=use_length_grouped_sampler,
            sampler_seed=sampler_seed,
            shuffle_reasons=data_cfg.get("shuffle_reasons", False),
            conv_audio_vad_target_sec=conv_audio_vad_target_sec,
            conv_audio_sample_rate=conv_audio_sample_rate,
        )

    def _build_judge(self, grpo_cfg):
        judge_type = grpo_cfg.get("judge", "format")
        if judge_type == "format":
            return FormatJudge(grpo_cfg)
        elif judge_type == "openrouter":
            return OpenRouterJudge(grpo_cfg.get("LLM_judge", {}))
        elif judge_type == "local_llm":
            from .judges.local_llm_judge import LocalLLMJudge
            return LocalLLMJudge(grpo_cfg.get("LLM_judge", {}))
        else:
            raise ValueError(f"Unknown judge type: {judge_type}")

    def run(self):
        runner_cfg = self.config.get("Runner", {})
        run_type = runner_cfg.get("type", "train")
        
        if run_type == "train":
            trainer_type = runner_cfg.get("trainer", "SFT")
            amp_enabled = self.config.get("Model", {}).get("amp", True)
            
            if trainer_type == "SFT":
                print("Run SFT")
                sft_cfg = runner_cfg.get("SFT", {})
                grounding_cfg = self.config.get("Datasets", {}).get("grounding", {}) or {}
                grounding_enabled = bool(grounding_cfg.get("enabled", False))
                debug_cfg_value = grounding_cfg.get("debug_save_examples", 6 if grounding_enabled else 0)
                if isinstance(debug_cfg_value, bool):
                    grounding_debug_max_examples = 6 if debug_cfg_value else 0
                else:
                    grounding_debug_max_examples = max(0, int(debug_cfg_value or 0))
                model_cfg = self.config.get("Model", {})
                trainer_config = {
                    **sft_cfg,
                    "lr": sft_cfg.get("optimizator", {}).get("init_lr", 1e-4),
                    "amp": amp_enabled,
                    "iters_per_epoch": sft_cfg.get("iters_per_epoch"),
                    "accum_grad_iters": sft_cfg.get("accum_grad_iters", 1),
                    "grounding_debug_max_examples": grounding_debug_max_examples,
                    "loss_weight_based_on_dataset_label": self.config.get("Datasets", {}).get("loss_weight_based_on_dataset_label", False),
                    "model_name": model_cfg.get("model_name"),
                }
                trainer = SFTTrainer(trainer_config, self.model, self.train_loader, self.val_loader, self.logger, device=self.device, output_dir=self.output_dir)
                trainer.train()
                
            elif trainer_type == "GRPO":
                grpo_cfg = runner_cfg.get("GRPO", {})
                judge = self._build_judge(grpo_cfg)
                trainer_config = {
                    **grpo_cfg, 
                    "lr": grpo_cfg.get("optimizator", {}).get("init_lr", 1e-4), 
                    "amp": amp_enabled,
                    "iters_per_epoch": grpo_cfg.get("iters_per_epoch"),
                    "accum_grad_iters": grpo_cfg.get("accum_grad_iters", 1),
                    "filter_controversial": grpo_cfg.get("filter_controversial", False),
                    "skeptic_batch_size": grpo_cfg.get("skeptic_batch_size"),
                    "skeptic_buffer_size": grpo_cfg.get("skeptic_buffer_size"),
                }
                trainer = GRPOTrainer(trainer_config, self.model, self.train_loader, self.val_loader, self.logger, judge, device=self.device, output_dir=self.output_dir)
                trainer.train()

            elif trainer_type == "Distillation":
                distill_cfg = runner_cfg.get("Distillation", {})
                sft_cfg = runner_cfg.get("SFT", {})
                if not sft_cfg:
                    sft_cfg = {"optimizator": {"init_lr": 1e-4}, "num_epochs": 1}
                model_cfg = self.config.get("Model", {})
                sft_trainer_config = {
                    **sft_cfg,
                    "lr": sft_cfg.get("optimizator", {}).get("init_lr", 1e-4),
                    "amp": amp_enabled,
                    "iters_per_epoch": sft_cfg.get("iters_per_epoch"),
                    "accum_grad_iters": sft_cfg.get("accum_grad_iters", 1),
                    "model_name": model_cfg.get("model_name"),
                }
                sft_trainer = SFTTrainer(
                    sft_trainer_config, self.model, self.train_loader, self.val_loader,
                    self.logger, device=self.device, output_dir=self.output_dir
                )

                grpo_cfg = runner_cfg.get("GRPO", {})
                if not grpo_cfg:
                    grpo_cfg = {"optimizator": {"init_lr": 1e-4}, "num_epochs": 1, "judge": "format"}
                judge = self._build_judge(grpo_cfg)
                grpo_trainer_config = {
                    **grpo_cfg,
                    "lr": grpo_cfg.get("optimizator", {}).get("init_lr", 1e-4),
                    "amp": amp_enabled,
                    "filter_controversial": grpo_cfg.get("filter_controversial", False),
                    "skeptic_batch_size": grpo_cfg.get("skeptic_batch_size"),
                    "skeptic_buffer_size": grpo_cfg.get("skeptic_buffer_size"),
                }
                grpo_trainer = GRPOTrainer(
                    grpo_trainer_config, self.model, self.train_loader, self.val_loader,
                    self.logger, judge, device=self.device, output_dir=self.output_dir
                )

                dataset_forming_epoch = None
                if distill_cfg.get("Filtering"):
                    from .epochs.distillation_dataset_forming_epoch import DistillationDatasetFormingEpoch
                    from .trainers.distillation_trainer import create_forming_loader
                    forming_cfg = {
                        **distill_cfg,
                        "GRPO": grpo_cfg,
                    }
                    forming_batch_size = (
                        distill_cfg.get("Filtering", {})
                        .get("intermediate_dataset_forming", {})
                        .get("batch_size")
                    )
                    forming_loader = (
                        create_forming_loader(self.train_loader, forming_batch_size)
                        if forming_batch_size is not None
                        else self.train_loader
                    )
                    dataset_forming_epoch = DistillationDatasetFormingEpoch(
                        self.model, forming_loader, self.logger, judge, forming_cfg,
                        device=self.device, amp=amp_enabled
                    )

                create_dataloader_from_path = (
                    lambda p: self._create_dataloader_from_path(p, task_type="reasoning")
                )
                trainer = DistillationTrainer(
                    distill_cfg, sft_trainer, grpo_trainer,
                    dataset_forming_epoch=dataset_forming_epoch,
                    initial_train_loader=self.train_loader,
                    create_dataloader_from_path=create_dataloader_from_path,
                )
                trainer.train()
                
        elif run_type == "test":
            self._log.debug("entering test branch, calling _run_test()")
            self._run_test()
            self._log.debug("run() finished (test)")
        else:
            self._log.debug("run() finished (run_type=%s)", run_type)

    def _run_test(self):
        """Run test/evaluation mode using Datasets.dataset_test_path."""
        runner_cfg = self.config.get("Runner", {})
        test_cfg = runner_cfg.get("Test", {})
        data_cfg = self.config.get("Datasets", {})
        
        # Get model format (what format the model was trained with)
        # Falls back to SFT.task_type if not specified
        model_format = test_cfg.get("model_format")
        if not model_format:
            model_format = runner_cfg.get("SFT", {}).get("task_type", "hard_label")
        
        # Dataset format (can differ from model format for cross-evaluation)
        dataset_format = test_cfg.get("dataset_format", model_format)
        
        # Get test dataset path and settings from Datasets section
        ds_path = data_cfg.get("dataset_test_path")
        max_samples = data_cfg.get("max_test_samples")
        ds_name = os.path.basename(ds_path).replace(".json", "").replace(".jsonl", "") if ds_path else "test"
        
        if not ds_path or ds_path == "dummy":
            print("Error: Datasets.dataset_test_path must be set for test mode", flush=True)
            return
        
        # Validate: hard_label model can't be tested on reasoning datasets
        if model_format == "hard_label" and dataset_format == "reasoning":
            print(f"Error: hard_label model cannot be tested on reasoning dataset", flush=True)
            return
        
        print(f"\n{'='*50}", flush=True)
        print(f"Testing on: {ds_name}", flush=True)
        print(f"Model format: {model_format}, Dataset format: {dataset_format}", flush=True)
        print(f"{'='*50}", flush=True)
        
        test_loader = self._build_test_dataloader(ds_path, dataset_format, max_samples)
        test_logger = self._build_test_logger(dataset_format, ds_name)
        
        from .epochs.test_epoch import TestEpoch
        gen_cfg = test_cfg.get("generation", {"max_new_tokens": 5000, "num_beams": 1, "do_sample": False})
        log_freq = self.config.get("Logger", {}).get("log_freq", 10)
        extract_confidence = test_cfg.get("extract_confidence", True)
        
        self._log.debug("building TestEpoch: model_format=%s dataset_format=%s", model_format, dataset_format)
        test_epoch = TestEpoch(
            self.model,
            test_loader,
            test_logger,
            device=self.device,
            gen_cfg=gen_cfg,
            model_format=model_format,
            dataset_format=dataset_format,
            log_freq=log_freq,
            extract_confidence=extract_confidence
        )
        test_epoch.run(epoch_num=0)
        self._log.debug("_run_test() finished")

    def _build_test_dataloader(self, dataset_path: str, dataset_format: str, max_samples: Optional[int] = None):
        """Build a dataloader for testing."""
        data_cfg = self.config.get("Datasets", {})
        runner_cfg = self.config.get("Runner", {})
        model_cfg = self.config.get("Model", {})
        model_name = model_cfg.get("model_name")

        whisper_path = None
        model_path = None
        conv_audio_vad_target_sec = None
        conv_audio_sample_rate = None
        if model_name == "salmon":
            whisper_path = model_cfg.get("additional_kwargs", {}).get("salmon", {}).get("pretrained_ckpts", {}).get("whisper_path")
        elif model_name == "conv_audio_classifier":
            cc = model_cfg.get("additional_kwargs", {}).get("conv_audio_classifier", {})
            conv_audio_vad_target_sec = cc.get("target_duration_sec")
            conv_audio_sample_rate = cc.get("sample_rate")

        test_cfg = runner_cfg.get("Test", {})
        batch_size = test_cfg.get("batch_size", 8)

        # For conv_audio_classifier at test: pass full-length (padded) audio; collator does
        # VAD + overlapping windows when conv_audio_overlap_ratio is set.
        conv_audio_full_wav_at_test = (
            model_name == "conv_audio_classifier" and test_cfg.get("conv_audio_full_wav_at_test", True)
        )
        conv_audio_overlap_ratio = None
        conv_audio_only_first_window = False
        if model_name == "conv_audio_classifier" and conv_audio_full_wav_at_test:
            if self.only_one_window:
                conv_audio_only_first_window = True
            else:
                conv_audio_overlap_ratio = test_cfg.get("generation", {}).get("overlap_ratio") or cc.get("overlap_ratio", 0.5)

        # Load prompt templates
        prompt_templates = self._get_prompt_templates(dataset_format)

        test_samples_offset = data_cfg.get("test_samples_offset", 0)
        use_length_grouped_sampler = data_cfg.get("use_length_grouped_sampler", False)
        sampler_seed = int(data_cfg.get("sampler_seed", 0))
        return get_dataloader(
            dataset_path,
            batch_size,
            shuffle=False,
            max_samples=max_samples or data_cfg.get("max_test_samples"),
            task_type=dataset_format,
            whisper_path=whisper_path,
            model_name=model_name,
            model_path=model_path,
            num_workers=runner_cfg.get("num_workers", 4),
            distributed=(self.num_gpus > 1),
            prompt_templates=prompt_templates,
            samples_offset=test_samples_offset,
            use_length_grouped_sampler=use_length_grouped_sampler,
            sampler_seed=sampler_seed,
            conv_audio_vad_target_sec=conv_audio_vad_target_sec,
            conv_audio_sample_rate=conv_audio_sample_rate,
            conv_audio_full_wav_at_test=conv_audio_full_wav_at_test,
            conv_audio_overlap_ratio=conv_audio_overlap_ratio,
            conv_audio_only_first_window=conv_audio_only_first_window,
        )

    def _build_test_logger(self, dataset_format: str, dataset_name: str):
        """Build logger appropriate for dataset format."""
        log_dir = os.path.join(self.output_dir, "logs", f"test_{dataset_name}")
        log_freq = self.config.get("Logger", {}).get("log_freq", 10)
        is_main_process = (self.local_rank == 0)
        
        test_cfg = self.config.get("Runner", {}).get("Test", {})
        save_all = test_cfg.get("save_all_predictions", False)
        
        if dataset_format == "reasoning":
            return ReasoningLogger(log_dir, log_freq if is_main_process else 1000000, save_all_predictions=save_all)
        else:
            return HardLabelLogger(log_dir, log_freq if is_main_process else 1000000, save_all_predictions=save_all)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml (required unless --pretrained)")
    parser.add_argument("--run_type", type=str, default=None, choices=["train", "test"], help="Override Runner.type (train or test)")
    parser.add_argument("--ckpt", type=str, default=None, help="Override Model.ckpt (checkpoint path for test or resume)")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        metavar="DIR",
        help="Experiment directory (e.g. outputs/exp/run_YYYY_MM_DD_HH_MM). Uses config_resolved.yaml and checkpoints from that run. Requires --run_type test.",
    )
    parser.add_argument(
        "--best-by-val-loss",
        action="store_true",
        help="With --pretrained: load best checkpoint by validation loss (sft_epoch_*_best.pt) instead of best balanced accuracy (sft_best_acc.pt).",
    )
    parser.add_argument(
        "--only-one-window",
        default=True,
        help="Test only first window after VAD per sample, no overlapping (conv_audio_classifier only).",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Override Datasets.dataset_test_path from config.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        metavar="N",
        help="Override Datasets.max_test_samples from config (e.g. for limiting test set size).",
    )
    args = parser.parse_args()

    if args.pretrained is not None:
        if args.run_type != "test":
            parser.error("--pretrained requires --run_type test")
        config_path, ckpt_path = _resolve_pretrained_paths(
            args.pretrained, prefer_best_val_loss=getattr(args, "best_by_val_loss", False)
        )
        run_type_override = "test"
        ckpt_override = ckpt_path
        # Save test results into the same experiment directory as the checkpoint
        output_dir_override = os.path.dirname(config_path)
        print(f"Pretrained: config={config_path}, ckpt={ckpt_path}", flush=True)
    else:
        if args.config is None:
            parser.error("Either --config or --pretrained is required")
        config_path = args.config
        run_type_override = args.run_type
        ckpt_override = args.ckpt
        output_dir_override = None

    runner = Runner(
        config_path,
        run_type_override=run_type_override,
        ckpt_override=ckpt_override,
        output_dir_override=output_dir_override,
        only_one_window=getattr(args, "only_one_window", False),
        test_dataset_path_override=getattr(args, "test_file", None),
        max_test_samples_override=getattr(args, "max_test_samples", None),
    )
    print("Runner Created")
    try:
        runner.run()
    finally:
        # Always cleanup distributed resources, even if training fails
        runner.cleanup()

if __name__ == "__main__":
    main()
