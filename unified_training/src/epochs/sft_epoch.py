from typing import Any, Optional, Tuple
import os
import json
import torch
import numpy as np
import torch.distributed as dist
import soundfile as sf
from tqdm import tqdm
from .base import TrainEpoch
from .utils.batch_utils import split_batch


BONAFIDE_WEIGHT_SUPPORTED_MODELS = ("salmon",)


class SFTTrainEpoch(TrainEpoch):
    def __init__(
        self,
        *args,
        iters_per_epoch: Optional[int] = None,
        accum_grad_iters: int = 1,
        grounding_debug_max_examples: int = 0,
        dataset_label_proportions: Optional[dict] = None,
        bonafide_weight: Optional[float] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.iters_per_epoch = iters_per_epoch
        self.accum_grad_iters = accum_grad_iters
        self.grounding_debug_max_examples = max(0, int(grounding_debug_max_examples or 0))
        self._grounding_examples_seen = 0
        self._grounding_examples_reservoir = []
        self.dataset_label_proportions = dataset_label_proportions
        if bonafide_weight is not None:
            if model_name not in BONAFIDE_WEIGHT_SUPPORTED_MODELS:
                raise ValueError(
                    f"bonafide_weight is only supported for models {BONAFIDE_WEIGHT_SUPPORTED_MODELS}, "
                    f"got model_name={model_name!r}. Set bonafide_weight only when using Salmon."
                )
        self.bonafide_weight = bonafide_weight
        self.model_name = model_name

    def _collect_grounding_examples(self, batch: Any, epoch_num: int, iteration_num: int):
        """Collect grounded examples using reservoir sampling across the full epoch."""
        if self.grounding_debug_max_examples <= 0:
            return
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        grounding_infos = batch.get("grounding_info")
        if not grounding_infos:
            return

        texts = batch.get("text") or []
        audio_ids = batch.get("audio_ids") or []
        original_paths = batch.get("original_path") or []
        raw_wavs = batch.get("raw_wav") or []
        prompts = batch.get("prompts") or []

        for j, ginfo in enumerate(grounding_infos):
            if not isinstance(ginfo, dict) or not ginfo.get("enabled", False):
                continue
            if j >= len(raw_wavs):
                continue

            audio_id = str(audio_ids[j]) if j < len(audio_ids) and audio_ids[j] is not None else f"idx{j}"
            record = {
                "epoch": epoch_num,
                "iteration": iteration_num,
                "sample_index_in_batch": j,
                "audio_id": audio_id,
                "original_path": original_paths[j] if j < len(original_paths) else "",
                "edited_label": texts[j] if j < len(texts) else "",
                "prompt_used": prompts[j] if j < len(prompts) else "",
                "grounding_info": ginfo,
                "wav": np.asarray(raw_wavs[j], dtype=np.float32),
            }

            self._grounding_examples_seen += 1
            if len(self._grounding_examples_reservoir) < self.grounding_debug_max_examples:
                self._grounding_examples_reservoir.append(record)
            else:
                replace_idx = np.random.randint(0, self._grounding_examples_seen)
                if replace_idx < self.grounding_debug_max_examples:
                    self._grounding_examples_reservoir[replace_idx] = record

    def _flush_grounding_examples(self, epoch_num: int):
        """Write sampled grounding examples and metadata to disk."""
        if self.grounding_debug_max_examples <= 0:
            return
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if not self._grounding_examples_reservoir:
            return

        out_dir = os.path.join(self.logger.log_dir, "grounding_examples")
        os.makedirs(out_dir, exist_ok=True)
        meta_path = os.path.join(out_dir, f"grounding_examples_epoch_{epoch_num}.jsonl")
        with open(meta_path, "w") as f:
            for rec in self._grounding_examples_reservoir:
                wav_basename = (
                    f"epoch_{rec['epoch']}_iter_{rec['iteration']}_sample_{rec['sample_index_in_batch']}_"
                    f"{rec['audio_id'].replace('/', '_').replace(' ', '_')}.wav"
                )
                wav_path = os.path.join(out_dir, wav_basename)
                try:
                    sf.write(wav_path, rec["wav"], 16000)
                except Exception as e:
                    wav_path = ""
                    print(f"Warning: failed to save grounding audio example: {e}", flush=True)

                row = {
                    "epoch": rec["epoch"],
                    "iteration": rec["iteration"],
                    "sample_index_in_batch": rec["sample_index_in_batch"],
                    "audio_id": rec["audio_id"],
                    "original_path": rec["original_path"],
                    "audio_path": wav_path,
                    "edited_label": rec["edited_label"],
                    "prompt_used": rec["prompt_used"],
                    "grounding_info": rec["grounding_info"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _get_autocast_dtype(self) -> torch.dtype:
        if self.scaler is None and self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _process_batch(
        self, batch: Any, backward: bool
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """Run SFT forward for one batch; optionally run backward. Returns (loss, loss_val). loss_val is None if NaN/Inf."""
        autocast_dtype = self._get_autocast_dtype()
        with torch.amp.autocast("cuda", enabled=self.amp, dtype=autocast_dtype):
            outputs = self.model(batch)
            loss = outputs["loss"]

        # Reduce per-sample loss: apply dataset-label weights and/or bonafide_weight if configured, then reduce
        if loss.dim() >= 1 and loss.numel() > 1:
            # Per-sample loss (B,) or (B, ...)
            if loss.dim() > 1:
                loss = loss.view(loss.size(0), -1).mean(dim=1)
            batch_size = loss.size(0)
            is_bonafide_tensor = batch.get("is_bonafide")
            if is_bonafide_tensor is not None:
                if is_bonafide_tensor.dim() > 1:
                    is_bonafide_tensor = is_bonafide_tensor.squeeze(-1)
                is_bonafide_bool = (is_bonafide_tensor.to(loss.device) >= 0.5)
            else:
                is_bonafide_bool = None

            if self.dataset_label_proportions is not None and is_bonafide_bool is not None:
                # Weight by inverse proportion: weight_y = 1 / (num_classes * p_y)
                proportions = self.dataset_label_proportions
                num_classes = max(len(proportions), 1)
                labels = is_bonafide_bool.long()
                weights = torch.zeros_like(loss, dtype=loss.dtype, device=loss.device)
                for c, p in proportions.items():
                    if p > 0:
                        weights[labels == c] = 1.0 / (num_classes * p)
                if self.bonafide_weight is not None:
                    weights[is_bonafide_bool] *= self.bonafide_weight
                loss = (weights * loss).sum() / weights.sum().clamp(min=1e-8)
            elif self.bonafide_weight is not None and is_bonafide_bool is not None:
                # Scale bonafide samples loss by bonafide_weight
                weights = torch.ones_like(loss, dtype=loss.dtype, device=loss.device)
                weights[is_bonafide_bool] = self.bonafide_weight
                loss = (weights * loss).sum() / weights.sum().clamp(min=1e-8)
            else:
                loss = loss.mean()
        else:
            loss = loss if loss.numel() == 1 else loss.mean()

        loss = loss / self.accum_grad_iters
        loss_val = loss.item() * self.accum_grad_iters
        if np.isnan(loss_val) or np.isinf(loss_val):
            return loss, None
        if backward:
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        return loss, loss_val

    def _handle_out_of_memory(
        self, batch: Any, batch_size: int, iteration: int
    ) -> Optional[float]:
        """On OOM: split batch and retry. Returns loss_val for logging or None to skip batch."""
        print(
            f"[OOM] CUDA out of memory at iteration {iteration} (batch_size={batch_size}). "
            "Splitting batch and retrying.",
            flush=True,
        )
        torch.cuda.empty_cache()
        self.optimizer.zero_grad()
        autocast_dtype = self._get_autocast_dtype()
        n_splits = min(2, batch_size)
        sub_size = (batch_size + n_splits - 1) // n_splits
        loss_val = None
        for start in range(0, batch_size, sub_size):
            end = min(start + sub_size, batch_size)
            if start >= end:
                continue
            sub_batch = split_batch(batch, start, end)
            try:
                _, lv = self._process_batch(sub_batch, backward=True)
                if lv is not None:
                    loss_val = lv
            except torch.cuda.OutOfMemoryError:
                print(
                    f"[OOM] Sub-batch ({start}:{end}) still OOM; skipping batch.",
                    flush=True,
                )
                torch.cuda.empty_cache()
                loss_val = None
        if loss_val is None:
            self.optimizer.zero_grad()
        return loss_val

    def run(self, epoch_num: int):
        self.logger.set_epoch(epoch_num, "train")
        self.model.train()
        self._grounding_examples_seen = 0
        self._grounding_examples_reservoir = []

        if hasattr(self.dataloader, "sampler") and hasattr(self.dataloader.sampler, "set_epoch"):
            self.dataloader.sampler.set_epoch(epoch_num)

        self.optimizer.zero_grad()

        total_iters = self.iters_per_epoch if self.iters_per_epoch is not None else len(self.dataloader)
        # Set LR for step 0 at start of epoch 0 (warmup_start_lr); otherwise scheduler only runs after first accum step
        if (
            epoch_num == 0
            and self.scheduler
            and hasattr(self.scheduler, "step")
            and self.scheduler.__class__.__name__ == "LinearWarmupCosineLRScheduler"
        ):
            self.scheduler.step(epoch_num, 0)

        use_tqdm = not dist.is_initialized() or dist.get_rank() == 0
        pbar = None
        if use_tqdm:
            pbar = tqdm(
                self.dataloader,
                total=total_iters,
                desc=f"Train Epoch {epoch_num}",
                dynamic_ncols=True,
            )
            iterator = enumerate(pbar)
        else:
            iterator = enumerate(self.dataloader)

        for i, batch in iterator:
            if i >= total_iters:
                break

            batch = self._move_to_device(batch, self.device)
            self._collect_grounding_examples(batch, epoch_num=epoch_num, iteration_num=i)
            prompts = batch.get("prompts")
            audio_ids = batch.get("audio_ids")
            batch_size = len(prompts) if prompts else (len(audio_ids) if audio_ids else 1)

            loss_val = None
            try:
                _, loss_val = self._process_batch(batch, backward=True)
                if loss_val is None:
                    print(f"Warning: NaN/Inf loss detected at iteration {i}, skipping backward!", flush=True)
                    self.optimizer.zero_grad()
                    self.logger.log(loss=0.0, lr=self.optimizer.param_groups[0]['lr'])
                    continue

            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda":
                    raise
                loss_val = self._handle_out_of_memory(batch, batch_size, i)
                if loss_val is None:
                    continue

            if (i + 1) % self.accum_grad_iters == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    if hasattr(self.scheduler, "step") and (self.scheduler.__class__.__name__ == "LinearWarmupCosineLRScheduler"):
                        self.scheduler.step(epoch_num, i)
                    else:
                        self.scheduler.step()

            if loss_val is not None:
                self.logger.log(loss=loss_val, lr=self.optimizer.param_groups[0]['lr'])
                if pbar is not None:
                    pbar.set_postfix(
                        {
                            "loss": f"{loss_val:.4f}",
                            "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        },
                        refresh=False,
                    )

        self._flush_grounding_examples(epoch_num=epoch_num)
        self.logger.log_epoch()

    def _move_to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._move_to_device(v, device) for v in batch]
        return batch





