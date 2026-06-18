from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
from ..loggers.base import Logger
from ..epochs.eval_epoch import EvalEpoch

class Trainer(ABC):
    def __init__(self, config: Dict[str, Any], model: torch.nn.Module, 
                 train_loader: Any, val_loader: Any, logger: Logger,
                 device: Optional[torch.device] = None, output_dir: Optional[str] = None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.output_dir = output_dir
        
        # If device not provided, infer from model parameters
        if device is None:
            if hasattr(model, "parameters") and list(model.parameters()):
                self.device = list(model.parameters())[0].device
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        self.num_epochs = config.get("num_epochs", 1)
        
        # Init optimizer/scheduler
        opt_cfg = config.get("optimizator", {})
        
        # Split parameters into weight decay and non-weight decay groups (like in reference SALMON)
        p_wd, p_non_wd = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # Weight decay for multi-dimensional parameters, except for Norm layers and biases
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
        
        optim_params = [
            {"params": p_wd, "weight_decay": opt_cfg.get("weight_decay", 0.05)},
            {"params": p_non_wd, "weight_decay": 0}
        ]
        
        self.optimizer = torch.optim.AdamW(
            optim_params, 
            lr=opt_cfg.get("init_lr", config.get("lr", 1e-4)),
            betas=(0.9, opt_cfg.get("beta2", 0.999))
        )
        
        from ..utils.optims import LinearWarmupCosineLRScheduler
        self.scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            max_epoch=self.num_epochs,
            iters_per_epoch=config.get("iters_per_epoch") or len(train_loader),
            min_lr=opt_cfg.get("min_lr", 1e-6),
            init_lr=opt_cfg.get("init_lr", config.get("lr", 1e-4)),
            warmup_steps=opt_cfg.get("warmup_steps", 0),
            warmup_start_lr=opt_cfg.get("warmup_start_lr", -1)
        )
        
        # AMP settings
        self.amp = config.get("amp", True)
        
        # Best checkpoint tracking (by accuracy, higher is better)
        self.best_val_accuracy = 0.0
        self.best_epoch = None
        # Best checkpoint by balanced accuracy (from evaluation); may differ from best val
        self.best_balanced_accuracy = 0.0
        self.best_epoch_balanced = None
        
        # Grad Scaler for AMP
        self.scaler = None
        self.max_grad_norm = opt_cfg.get("max_grad_norm", 1.0)
        if config.get("amp", True) and self.device.type == "cuda" and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                print("Using bfloat16 (no scaler needed)", flush=True)
                self.scaler = None
            else:
                print("Using float16 with GradScaler", flush=True)
                # Note: For GradScaler to work, model should ideally be in float32
                # If model is already in float16, this might fail with "Attempting to unscale FP16 gradients"
                self.scaler = torch.cuda.amp.GradScaler()

    @abstractmethod
    def train(self):
        pass

    def validate(self, epoch_num: int) -> float:
        gen_cfg = self.config.get("generation", {})
        eval_epoch = EvalEpoch(self.model, self.val_loader, self.logger, device=self.device, gen_cfg=gen_cfg if gen_cfg else None)
        eval_epoch.run(epoch_num=epoch_num)
        accuracy = getattr(self.logger, 'last_accuracy', 0.0)
        return accuracy

    def save_checkpoint(self, epoch_num: int, val_accuracy: float, name: str = "checkpoint"):
        if self.output_dir is None:
            return
            
        import os
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        is_best = val_accuracy > self.best_val_accuracy
        save_best_checkpoint = self.config.get("save_best_checkpoint", True)

        if is_best:
            old_best_epoch = self.best_epoch
            self.best_val_accuracy = val_accuracy
            self.best_epoch = epoch_num

            if save_best_checkpoint:
                if old_best_epoch is not None and old_best_epoch != epoch_num:
                    old_best = os.path.join(ckpt_dir, f"{name}_epoch_{old_best_epoch}_best.pt")
                    if os.path.exists(old_best):
                        best_acc_path = os.path.join(ckpt_dir, f"{name}_best_acc.pt")
                        if self.best_epoch_balanced == old_best_epoch and os.path.lexists(best_acc_path):
                            # best_acc symlink points at old_best — rename to best_acc file, then repoint symlink
                            old_best_acc = os.path.join(ckpt_dir, f"{name}_epoch_{old_best_epoch}_best_acc.pt")
                            os.remove(best_acc_path)
                            os.rename(old_best, old_best_acc)
                            os.symlink(os.path.basename(old_best_acc), best_acc_path)
                            print(f"Renamed old best checkpoint to best_acc: {old_best} -> {old_best_acc}", flush=True)
                        else:
                            os.remove(old_best)
                            print(f"Deleted old best checkpoint: {old_best}", flush=True)

                save_path = os.path.join(ckpt_dir, f"{name}_epoch_{epoch_num}_best.pt")
                torch.save({
                    'epoch': epoch_num,
                    'model': model_to_save.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                    'config': self.config,
                    'val_accuracy': val_accuracy,
                    'best_epoch': self.best_epoch,
                    'best_balanced_accuracy': getattr(self, 'best_balanced_accuracy', 0.0),
                    'best_epoch_balanced': getattr(self, 'best_epoch_balanced', None),
                }, save_path)
                print(f"Best checkpoint saved to {save_path} (accuracy={val_accuracy:.4f})", flush=True)

        # Always save latest (overwrites each epoch)
        latest_path = os.path.join(ckpt_dir, f"{name}_latest.pt")
        torch.save({
            'epoch': epoch_num,
            'model': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config,
            'val_accuracy': val_accuracy,
            'best_epoch': self.best_epoch,
            'best_balanced_accuracy': getattr(self, 'best_balanced_accuracy', 0.0),
            'best_epoch_balanced': getattr(self, 'best_epoch_balanced', None),
        }, latest_path)
        print(f"Latest checkpoint saved to {latest_path}", flush=True)

        # Always save best checkpoint by balanced accuracy (from evaluation); use symlink if same as best val
        balanced_accuracy = getattr(self.logger, 'last_accuracy_balanced', None)
        if balanced_accuracy is not None and balanced_accuracy > self.best_balanced_accuracy:
            old_best_epoch_balanced = self.best_epoch_balanced
            self.best_balanced_accuracy = balanced_accuracy
            self.best_epoch_balanced = epoch_num

            best_acc_path = os.path.join(ckpt_dir, f"{name}_best_acc.pt")
            if self.best_epoch_balanced == self.best_epoch:
                # Same as best validation checkpoint — do not duplicate; use symlink
                if os.path.lexists(best_acc_path):
                    os.remove(best_acc_path)
                target_basename = f"{name}_epoch_{self.best_epoch_balanced}_best.pt"
                os.symlink(target_basename, best_acc_path)
                print(f"Best-by-balanced-acc checkpoint: {best_acc_path} -> {target_basename} (same as best val)", flush=True)
            else:
                # Different epoch — save standalone checkpoint and point best_acc at it
                if old_best_epoch_balanced is not None and old_best_epoch_balanced != self.best_epoch:
                    old_path = os.path.join(ckpt_dir, f"{name}_epoch_{old_best_epoch_balanced}_best_acc.pt")
                    if os.path.lexists(old_path):
                        os.remove(old_path)
                        print(f"Deleted old best-by-balanced checkpoint: {old_path}", flush=True)
                epoch_best_acc_path = os.path.join(ckpt_dir, f"{name}_epoch_{self.best_epoch_balanced}_best_acc.pt")
                torch.save({
                    'epoch': self.best_epoch_balanced,
                    'model': model_to_save.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                    'config': self.config,
                    'val_accuracy': val_accuracy,
                    'best_epoch': self.best_epoch,
                    'best_balanced_accuracy': self.best_balanced_accuracy,
                    'best_epoch_balanced': self.best_epoch_balanced,
                }, epoch_best_acc_path)
                if os.path.lexists(best_acc_path):
                    os.remove(best_acc_path)
                os.symlink(os.path.basename(epoch_best_acc_path), best_acc_path)
                print(f"Best-by-balanced-acc checkpoint saved: {best_acc_path} -> {epoch_best_acc_path} (balanced_accuracy={balanced_accuracy:.4f})", flush=True)

    def load_checkpoint(self, resume_path: str):
        import os
        import torch
        if not os.path.exists(resume_path):
            print(f"Warning: Resume path {resume_path} does not exist. Starting training from scratch.")
            return

        print(f"Loading checkpoint from {resume_path}", flush=True)
        checkpoint = torch.load(resume_path, map_location="cpu")

        # Load model state dict
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(checkpoint['model'], strict=False)
        print("Model state dict loaded.", flush=True)

        # Load optimizer state dict
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state dict loaded.", flush=True)
        else:
            print("Warning: Optimizer state dict not found in checkpoint or optimizer not initialized. Optimizer will restart from scratch.", flush=True)

        # Load scaler state dict if AMP is used
        if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("AMP GradScaler state dict loaded.", flush=True)
        else:
            print("Warning: AMP GradScaler state dict not found in checkpoint or scaler not initialized. Scaler will restart from scratch.", flush=True)

        # Resume epoch, best accuracy, and best epoch index
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_accuracy = checkpoint.get('val_accuracy', 0.0)
        self.best_epoch = checkpoint.get('best_epoch')
        self.best_balanced_accuracy = checkpoint.get('best_balanced_accuracy', 0.0)
        self.best_epoch_balanced = checkpoint.get('best_epoch_balanced')
        print(f"Resuming from epoch {self.start_epoch} with best validation accuracy {self.best_val_accuracy:.4f}", flush=True)

        # Update scheduler's last_epoch to ensure correct LR scheduling
        if hasattr(self.scheduler, 'last_epoch'):
            self.scheduler.last_epoch = self.start_epoch - 1
            print(f"Scheduler last_epoch set to {self.scheduler.last_epoch}", flush=True)





