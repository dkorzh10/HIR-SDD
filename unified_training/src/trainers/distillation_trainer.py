from typing import Any, Callable, Optional, List
import os
import json

from torch.utils.data import DataLoader, Subset

from .sft_trainer import SFTTrainer
from .grpo_trainer import GRPOTrainer


def _get_base_dataset(dataloader: DataLoader):
    """Get underlying dataset from possibly wrapped (Subset, etc.)."""
    ds = dataloader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def create_forming_loader(base_loader: DataLoader, batch_size: int) -> DataLoader:
    """Create a DataLoader for dataset forming with configurable batch_size."""
    return DataLoader(
        base_loader.dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=base_loader.collate_fn,
        num_workers=base_loader.num_workers,
        pin_memory=getattr(base_loader, "pin_memory", False),
    )


def _create_subset_loader(
    base_loader: DataLoader,
    indices: List[int],
) -> DataLoader:
    """Create a new DataLoader from a subset of the base loader's dataset."""
    base_ds = _get_base_dataset(base_loader)
    subset = Subset(base_ds, indices)
    return DataLoader(
        subset,
        batch_size=base_loader.batch_size,
        shuffle=base_loader.shuffle if hasattr(base_loader, "shuffle") else True,
        collate_fn=base_loader.collate_fn,
        num_workers=base_loader.num_workers,
        pin_memory=getattr(base_loader, "pin_memory", False),
    )


class DistillationTrainer:
    """Orchestrates Dataset forming -> SFT -> GRPO cycles. Composes SFTTrainer and GRPOTrainer."""

    def __init__(
        self,
        config: dict,
        sft_trainer: SFTTrainer,
        grpo_trainer: GRPOTrainer,
        dataset_forming_epoch: Optional[Any] = None,
        initial_train_loader: Optional[DataLoader] = None,
        create_dataloader_from_path: Optional[Callable[[str], DataLoader]] = None,
    ):
        self.config = config
        self.sft_trainer = sft_trainer
        self.grpo_trainer = grpo_trainer
        self.dataset_forming_epoch = dataset_forming_epoch
        self.initial_train_loader = initial_train_loader or sft_trainer.train_loader
        self.create_dataloader_from_path = create_dataloader_from_path
        self.num_iters = config.get("num_distillation_iters", 1)
        self.prev_lengths_chars: Optional[List[int]] = None
        self.prev_lengths_tokens: Optional[List[int]] = None

    def train(self):
        for i in range(self.num_iters):
            print(f"Distillation Iteration {i}", flush=True)

            # 1. Form intermediate dataset (before SFT/GRPO)
            if self.dataset_forming_epoch is not None:
                prev_chars = self.prev_lengths_chars
                prev_tokens = self.prev_lengths_tokens
                if i > 0 and (prev_chars is None or prev_tokens is None):
                    prev_chars, prev_tokens = self._load_prev_lengths(i - 1)
                indices, lengths_chars, lengths_tokens = self.dataset_forming_epoch.run(
                    distillation_iter=i,
                    prev_lengths_chars=prev_chars,
                    prev_lengths_tokens=prev_tokens,
                )
                self.prev_lengths_chars = lengths_chars
                self.prev_lengths_tokens = lengths_tokens
                if not indices:
                    print("  No samples passed filtering; ending training.", flush=True)
                    break

                # Use reasoning-format intermediate dataset if saved (with generated traces)
                logger = getattr(self.sft_trainer, "logger", None)
                reason_path = None
                if logger and hasattr(logger, "get_iter_dataset_forming_dir") and self.create_dataloader_from_path:
                    df_dir = logger.get_iter_dataset_forming_dir(i)
                    reason_path = os.path.join(df_dir, f"intermediate_dataset_iter_{i}.json")
                if reason_path and os.path.exists(reason_path):
                    try:
                        with open(reason_path, "r") as f:
                            data = json.load(f)
                        if isinstance(data, list) and len(data) > 0 and "reasoning" in data[0]:
                            sft_loader = self.create_dataloader_from_path(reason_path)
                            self.sft_trainer.train_loader = sft_loader
                            print(f"  SFT: reasoning dataset ({len(data)} samples), GRPO: raw dataset", flush=True)
                        else:
                            sft_loader = _create_subset_loader(self.initial_train_loader, indices)
                            self.sft_trainer.train_loader = sft_loader
                            print(f"  SFT: intermediate dataset ({len(indices)} samples), GRPO: raw dataset", flush=True)
                    except (json.JSONDecodeError, OSError):
                        sft_loader = _create_subset_loader(self.initial_train_loader, indices)
                        self.sft_trainer.train_loader = sft_loader
                        print(f"  SFT: intermediate dataset ({len(indices)} samples), GRPO: raw dataset", flush=True)
                else:
                    sft_loader = _create_subset_loader(self.initial_train_loader, indices)
                    self.sft_trainer.train_loader = sft_loader
                    print(f"  SFT: intermediate dataset ({len(indices)} samples), GRPO: raw dataset", flush=True)

            # GRPO always uses raw dataset (applies its own filtering, e.g. filter_controversial)
            self.grpo_trainer.train_loader = self.initial_train_loader

            # 2. Run SFT (logs to iteration_i/sft/)
            logger = getattr(self.sft_trainer, "logger", None)
            if logger and hasattr(logger, "set_distillation_iter"):
                logger.set_distillation_iter(i, "sft")
            self.sft_trainer.train()

            # 3. Run GRPO (logs to iteration_i/grpo/)
            if logger and hasattr(logger, "set_distillation_iter"):
                logger.set_distillation_iter(i, "grpo")
            self.grpo_trainer.train()

    def _load_prev_lengths(self, prev_iter: int):
        """Load lengths from previous iteration's form_stats for relative filtering."""
        import os
        import json
        logger = getattr(self.sft_trainer, "logger", None)
        if logger is None or not hasattr(logger, "get_iter_dataset_forming_dir"):
            return None, None
        df_dir = logger.get_iter_dataset_forming_dir(prev_iter)
        path = os.path.join(df_dir, f"form_stats_iter_{prev_iter}.json")
        if not os.path.exists(path):
            return None, None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("lengths_chars"), data.get("lengths_tokens")
        except (json.JSONDecodeError, OSError):
            return None, None
