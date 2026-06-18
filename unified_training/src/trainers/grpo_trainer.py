from typing import Any, Dict, Optional
import torch

from .base import Trainer
from ..epochs.grpo_epoch import GRPOTrainEpoch
from ..judges.base import Judge


class GRPOTrainer(Trainer):
    def __init__(self, config: Dict[str, Any], model: Any, train_loader: Any, val_loader: Any, logger: Any, judge: Judge, device: Optional[object] = None, output_dir: Optional[str] = None):
        super().__init__(config, model, train_loader, val_loader, logger, device=device, output_dir=output_dir)
        self.judge = judge

        self._get_model().swap_weights()

    def _get_model(self):
        return self.model.module if isinstance(
            self.model, torch.nn.parallel.DistributedDataParallel
        ) else self.model

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Starting GRPO Epoch {epoch}", flush=True)
            gen_cfg = self.config.get("generation")
            filter_controversial = self.config.get("filter_controversial", False)
            skeptic_batch_size = self.config.get("skeptic_batch_size")
            skeptic_buffer_size = self.config.get("skeptic_buffer_size")
            if filter_controversial:
                print(
                    "[GRPO] Skeptic mode ENABLED. Training only on samples with mix of correct/incorrect (rejecting all-correct and all-incorrect).",
                    flush=True,
                )
            train_epoch = GRPOTrainEpoch(
                self.model, self.train_loader, self.logger,
                self.optimizer, self.judge,
                device=self.device,
                amp=self.amp,
                num_generations=self.config.get("num_generations", 4),
                beta=self.config.get("beta", 0.1),
                iters_per_epoch=self.config.get("iters_per_epoch"),
                accum_grad_iters=self.config.get("accum_grad_iters", 1),
                gen_cfg=gen_cfg,
                max_grad_norm=self.max_grad_norm,
                filter_controversial=filter_controversial,
                skeptic_buffer_size=skeptic_buffer_size,
                skeptic_batch_size=skeptic_batch_size,
            )
            train_epoch.run(epoch_num=epoch)

            val_accuracy = 0.0
            if self.val_loader:
                val_accuracy = self.validate(epoch)

            self.save_checkpoint(epoch, val_accuracy, name="grpo")
