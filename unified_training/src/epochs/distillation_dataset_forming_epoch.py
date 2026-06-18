"""
Dataset forming for Distillation: chunks, skeptic presampling, correctness/length filtering.
"""
from tqdm import tqdm
import random
import re
from typing import Any, Dict, List, Optional, Tuple
import gc
import torch
import torch.distributed as dist
from torch.utils.data import Subset, DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler

from .base import Epoch
from .utils.text_utils import texts_for_log
from .utils.batch_utils import split_batch
from ..judges.base import Judge
from ..loggers.reasoning_logger import ALL_REASONS


def _parse_reasoning_output(text: str) -> Tuple[str, List[str], Optional[bool]]:
    """Parse generated text into (think_content, reasons_list, is_bonafide) for dataloader.
    Returns (think_content, reasons_list, is_bonafide). is_bonafide=None if answer not parsable."""
    think_content = ""
    reasons_list: List[str] = []
    is_bonafide: Optional[bool] = None

    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()

    reasons_match = re.search(r"<reasons>\s*(\[.*?\])\s*</reasons>", text, re.DOTALL)
    if reasons_match:
        try:
            raw = reasons_match.group(1)
            parsed = eval(raw)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str):
                        if item.startswith("["):
                            inner = eval(item)
                            reasons_list.extend(r.upper() for r in inner if isinstance(r, str))
                        else:
                            reasons_list.append(item.upper())
        except Exception:
            pass

    ans_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if ans_match:
        raw_ans = ans_match.group(1).strip().lower()
        if raw_ans in ("real", "bonafide"):
            is_bonafide = True
        elif raw_ans in ("fake", "spoof"):
            is_bonafide = False

    # Filter reasons to canonical list only (model must not invent new ones)
    valid_set = {r.upper() for r in ALL_REASONS}
    reasons_list = [r for r in reasons_list if r in valid_set]

    return think_content, reasons_list, is_bonafide


class DistillationDatasetFormingEpoch(Epoch):
    """
    Form intermediate dataset by chunking, skeptic presampling, correctness and length filtering.
    Returns (accepted_indices, lengths_chars, lengths_tokens) for creating Subset + next iter stats.
    """

    def __init__(
        self,
        model,
        dataloader,
        logger,
        judge: Judge,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        amp: bool = True,
    ):
        super().__init__(model, dataloader, logger, device=device)
        self.judge = judge
        self.config = config
        self.amp = amp

        forming = config.get("Filtering", {}).get("intermediate_dataset_forming", {})
        skeptic = forming.get("skeptical_presampling", {})

        self.num_generations = config.get("num_generations_per_sample", 8)
        self.filtering_chunk_size = config.get("filtering_chunk_size", 1000)
        self.max_attempts = config.get("max_dataset_forming_attempts", 10)
        self.min_intermediate_size = config.get("min_intermediate_dataset_size", 100)
        self.save_intermediate_dataset = forming.get("save_intermediate_dataset", True)
        self.save_tmp_freq = forming.get("save_intermediate_dataset_tmp_freq", 10)
        self.save_presampling = forming.get("save_presampling", False)
        self.balance_by_rollouts = forming.get("balance_by_rollouts", False)

        self.min_correct = forming.get("min_correct", 1)
        self.max_correct = forming.get("max_correct", self.num_generations - 1)
        self.min_text_length = forming.get("min_text_length", 50)
        self.max_text_length = forming.get("max_text_length", 10000)
        self.min_tokens_len = forming.get("min_tokens_len", 20)
        self.max_tokens_len = forming.get("max_tokens_len", 4096)
        self.relative_text_length_ratio = forming.get("relative_text_length_ratio", 0.8)

        self.skeptic_enable = skeptic.get("enable", False)
        self.skeptic_num_gens = skeptic.get("num_generations", 3)
        self.skeptic_chunk_size = skeptic.get("skeptic_chunk_size")  # None = no limit per attempt
        self.min_skeptic_dataset_size = skeptic.get("min_skeptic_dataset_size", 50)
        self.max_skeptic_attempts = skeptic.get("max_skeptic_attempts", 3)
        self.skeptic_min_correct = skeptic.get("min_correct", 1)
        self.skeptic_max_correct = skeptic.get("max_correct", 2)
        self.skeptic_min_text_length = skeptic.get("min_text_length", 0)
        self.skeptic_max_text_length = skeptic.get("max_text_length", 100000)
        self.skeptic_min_tokens_len = skeptic.get("min_tokens_len", 0)
        self.skeptic_max_tokens_len = skeptic.get("max_tokens_len", 100000)

        self.gen_cfg = config.get("GRPO", {}).get("generation", {"do_sample": True, "top_p": 0.9, "max_new_tokens": 5000})
        if not self.gen_cfg:
            self.gen_cfg = {"do_sample": True, "top_p": 0.9, "max_new_tokens": 5000}

        # Chunk generations so we do (chunk_size * batch_size) rollouts per forward pass.
        # chunk_size=1 => batch_size rollouts per pass (e.g. 8 samples x 1 gen = 8 rollouts).
        self.generation_chunk_size = forming.get("generation_chunk_size", 1)

        self.prev_lengths_chars: Optional[List[int]] = None
        self.prev_lengths_tokens: Optional[List[int]] = None

    def run(
        self,
        distillation_iter: int,
        prev_lengths_chars: Optional[List[int]] = None,
        prev_lengths_tokens: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Run dataset forming. Returns (accepted_indices, lengths_chars, lengths_tokens).
        accepted_indices are indices into the underlying dataset.
        """
        self.prev_lengths_chars = prev_lengths_chars
        self.prev_lengths_tokens = prev_lengths_tokens
        dataset = self.dataloader.dataset
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset  # Subset wraps
        total_size = len(dataset)

        intermediate_indices: List[int] = []
        all_lengths_chars: List[int] = []
        all_lengths_tokens: List[int] = []
        all_audio_ids: List[Any] = []
        samples_processed = 0

        self.logger.set_distillation_iter(distillation_iter, "dataset_forming")

        _is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if _is_main and hasattr(self.logger, "save_dataset_forming_progress"):
            self.logger.save_dataset_forming_progress(
                distillation_iter, "start", 0, 0, n_candidates=0, n_accumulated=0, attempt=0
            )
        base_ds = self.dataloader.dataset
        while hasattr(base_ds, "dataset"):
            base_ds = base_ds.dataset

        if _is_main:
            print(
                f"[Distillation] Dataset forming iter {distillation_iter}: device={self.device}, "
                f"batch_size={self.dataloader.batch_size}, min_size={self.min_intermediate_size}",
                flush=True,
            )

        self.model.eval()

        # Phase 1: Skeptic filtering (attempts until min_skeptic_dataset_size or max_skeptic_attempts)
        candidate_indices: List[int] = []
        n_rollouts_skeptic = 0
        cumulative_samples_processed_skeptic = 0
        if self.skeptic_enable:
            candidate_indices, n_rollouts_skeptic, cumulative_samples_processed_skeptic = self._run_skeptic_phase(
                base_ds, total_size, distillation_iter, _is_main
            )
        else:
            candidate_indices = list(range(total_size))

        # Phase 2: Main filtering on candidates. If not enough, re-run skeptic to get more.
        # List of (idx, chars, tokens, audio_id, reasoning) - multiple rollouts per idx allowed
        accumulated_data: List[Tuple[int, int, int, Any, str]] = []
        n_rollouts_main = 0
        max_main_skeptic_rounds = self.max_skeptic_attempts if self.skeptic_enable else 1
        
        # Track cumulative progress across rounds to avoid progress.json resets
        cumulative_batch_idx_main = 0
        cumulative_samples_processed_main = 0

        for _round in range(max_main_skeptic_rounds):
            if not candidate_indices:
                break
            if len(accumulated_data) >= self.min_intermediate_size:
                break

            phase2_loader = self._create_phase2_loader(base_ds, candidate_indices)
            if _is_main:
                print(f"[Distillation] Phase 2 (round {_round + 1}): Main filtering on {len(candidate_indices)} candidates...", flush=True)
                if hasattr(self.logger, "save_dataset_forming_progress"):
                    # Use cumulative values so progress.json never resets
                    self.logger.save_dataset_forming_progress(
                        distillation_iter, "main",
                        batch_idx=cumulative_batch_idx_main,
                        samples_processed=cumulative_samples_processed_main,
                        n_candidates=len(candidate_indices),
                        n_accumulated=len(accumulated_data),
                        attempt=_round,
                    )

            round_intermediate: List[int] = []
            round_lengths_chars: List[int] = []
            round_lengths_tokens: List[int] = []
            round_audio_ids: List[Any] = []
            round_reasoning_texts: List[str] = []
            samples_processed = 0
            batch_idx = 0
            for batch in phase2_loader:
                # Check stop condition BEFORE processing batch to stop immediately
                if len(accumulated_data) >= self.min_intermediate_size:  # total samples (multi-rollout)
                    if _is_main:
                        print(f"[Distillation] Reached min_intermediate_size={self.min_intermediate_size}, stopping.", flush=True)
                    break
                # Write progress BEFORE expensive work so it updates even when batch takes 10+ min
                if _is_main and hasattr(self.logger, "save_dataset_forming_progress"):
                    n_acc_total = len(accumulated_data) + len(round_intermediate)
                    # Use cumulative values across rounds
                    self.logger.save_dataset_forming_progress(
                        distillation_iter, "main",
                        batch_idx=cumulative_batch_idx_main + batch_idx,
                        samples_processed=cumulative_samples_processed_main + samples_processed,
                        n_candidates=len(candidate_indices),
                        n_accumulated=n_acc_total,
                        attempt=_round,
                    )
                batch = self._move_to_device(batch, self.device)
                batch_sz = len(batch.get("prompts", batch.get("audio_ids", [1])))
                try:
                    accepted_local, lchars, ltokens, accepted_aids, accepted_reasoning, verdicts = self._filter_main_only(
                        batch, distillation_iter
                    )
                except torch.cuda.OutOfMemoryError:
                    if self.device.type != "cuda":
                        raise
                    accepted_local, lchars, ltokens, accepted_aids, accepted_reasoning, verdicts = self._handle_out_of_memory_main(
                        batch, batch_sz, batch_idx, distillation_iter, "main"
                    )
                for local_idx in accepted_local:
                    global_idx = candidate_indices[samples_processed + local_idx]
                    round_intermediate.append(global_idx)
                round_lengths_chars.extend(lchars)
                round_lengths_tokens.extend(ltokens)
                round_audio_ids.extend(accepted_aids)
                round_reasoning_texts.extend(accepted_reasoning)
                n_rollouts_main += batch_sz * self.num_generations
                samples_processed += batch_sz

                if _is_main:
                    n_acc = len(accepted_local)
                    print(
                        f"[Distillation] batch {batch_idx}: processed={samples_processed}, "
                        f"accepted={n_acc}/{batch_sz}, running_total={len(round_intermediate)}",
                        flush=True,
                    )
                    if hasattr(self.logger, "save_dataset_forming_progress"):
                        n_acc_total = len(accumulated_data) + len(round_intermediate)
                        # Use cumulative values across rounds
                        self.logger.save_dataset_forming_progress(
                            distillation_iter, "main",
                            batch_idx=cumulative_batch_idx_main + batch_idx,
                            samples_processed=cumulative_samples_processed_main + samples_processed,
                            n_candidates=len(candidate_indices),
                            n_accumulated=n_acc_total,
                            attempt=_round,
                        )
                    if self.save_intermediate_dataset and self.save_tmp_freq > 0 and (batch_idx + 1) % self.save_tmp_freq == 0:
                        tmp_entries = list(accumulated_data) + [
                            (round_intermediate[i], round_lengths_chars[i], round_lengths_tokens[i],
                             round_audio_ids[i], round_reasoning_texts[i])
                            for i in range(len(round_intermediate))
                        ]
                        tmp_indices = [t[0] for t in tmp_entries]
                        tmp_chars = [t[1] for t in tmp_entries]
                        tmp_tokens = [t[2] for t in tmp_entries]
                        tmp_aids = [t[3] for t in tmp_entries]
                        path = self.logger.save_intermediate_dataset_tmp(
                            distillation_iter,
                            tmp_indices,
                            tmp_chars,
                            tmp_tokens,
                            n_raw=samples_processed,
                            batch_idx=batch_idx,
                            audio_ids=tmp_aids,
                        )
                        print(f"[Distillation] Saved tmp progress to {path}", flush=True)
                if self.save_presampling and verdicts is not None and hasattr(self.logger, "save_presampling_verdicts"):
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    path = self.logger.save_presampling_verdicts(
                        distillation_iter, batch_idx, verdicts, rank=rank
                    )
                    print(f"[Distillation] rank={rank} saved presampling verdicts to {path}", flush=True)
                batch_idx += 1

            accumulated_data.extend([
                (round_intermediate[i], round_lengths_chars[i], round_lengths_tokens[i],
                 round_audio_ids[i], round_reasoning_texts[i])
                for i in range(len(round_intermediate))
            ])
            
            # Update cumulative counters for next round
            cumulative_batch_idx_main += batch_idx
            cumulative_samples_processed_main += samples_processed

            if len(accumulated_data) >= self.min_intermediate_size:
                break
            if self.skeptic_enable and _round < max_main_skeptic_rounds - 1:
                if _is_main:
                    print(f"[Distillation] Not enough after main ({len(accumulated_data)}), re-running skeptic...", flush=True)
                candidate_indices, extra_rollouts, extra_samples = self._run_skeptic_phase(
                    base_ds, total_size, distillation_iter, _is_main
                )
                n_rollouts_skeptic += extra_rollouts
                cumulative_samples_processed_skeptic += extra_samples

        intermediate_indices = [t[0] for t in accumulated_data]
        all_lengths_chars = [t[1] for t in accumulated_data]
        all_lengths_tokens = [t[2] for t in accumulated_data]
        all_audio_ids = [t[3] for t in accumulated_data]
        all_reasoning_texts = [t[4] for t in accumulated_data]

        # Final filtration step: undersample major class (real/fake) so dataset is balanced by rollouts
        if self.balance_by_rollouts and intermediate_indices:
            intermediate_indices, all_lengths_chars, all_lengths_tokens, all_audio_ids, all_reasoning_texts = (
                self._balance_rollouts_by_real_fake(
                    base_ds,
                    intermediate_indices,
                    all_lengths_chars,
                    all_lengths_tokens,
                    all_audio_ids,
                    all_reasoning_texts,
                    distillation_iter,
                    _is_main,
                )
            )

        n_raw = total_size
        n_candidates = len(candidate_indices)
        # Calculate total cumulative samples processed across all phases
        total_samples_processed = cumulative_samples_processed_skeptic + cumulative_samples_processed_main
        if _is_main:
            print(
                f"[Distillation] Dataset forming done: Raw={n_raw}, Candidates={n_candidates}, "
                f"Clean={len(intermediate_indices)}, SamplesProcessed={total_samples_processed}",
                flush=True,
            )
            if hasattr(self.logger, "save_dataset_forming_progress"):
                self.logger.save_dataset_forming_progress(
                    distillation_iter, "done",
                    batch_idx=cumulative_batch_idx_main,
                    samples_processed=total_samples_processed,
                    n_candidates=n_candidates,
                    n_accumulated=len(intermediate_indices),
                    attempt=0,
                )

        # Save final intermediate dataset (reasoning format with generated traces for SFT)
        if _is_main and self.save_intermediate_dataset and hasattr(self.logger, "save_intermediate_dataset"):
            if hasattr(self.logger, "save_intermediate_dataset_reasoning"):
                samples = self._build_reasoning_samples(
                    base_ds, intermediate_indices, all_reasoning_texts,
                    all_lengths_chars, all_lengths_tokens,
                )
                path = self.logger.save_intermediate_dataset_reasoning(
                    distillation_iter, samples, n_raw=n_raw
                )
            else:
                path = self.logger.save_intermediate_dataset(
                    distillation_iter,
                    intermediate_indices,
                    all_lengths_chars,
                    all_lengths_tokens,
                    n_raw=n_raw,
                    audio_ids=all_audio_ids if all_audio_ids else None,
                )
            print(f"[Distillation] Saved intermediate dataset to {path}", flush=True)

        # Log final stats for this iteration (for next-iter relative filter + plots)
        # Only rank 0 writes shared files to avoid concurrent write races
        if dist.is_initialized():
            dist.barrier()
        if _is_main:
            self.logger.log_dataset_forming_stats(
                distillation_iter,
                n_raw=n_raw,
                n_after_skeptic=n_candidates if self.skeptic_enable else None,
                n_final=len(intermediate_indices),
                n_rollouts_skeptic=n_rollouts_skeptic if self.skeptic_enable else None,
                n_rollouts_main=n_rollouts_main,
                lengths_chars=all_lengths_chars if all_lengths_chars else None,
                lengths_tokens=all_lengths_tokens if all_lengths_tokens else None,
            )

        return intermediate_indices, all_lengths_chars, all_lengths_tokens

    def _build_reasoning_samples(
        self,
        base_ds,
        indices: List[int],
        reasoning_texts: List[str],
        lengths_chars: Optional[List[int]] = None,
        lengths_tokens: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Build reasoning-format samples from base dataset + generated reasoning traces.
        Parses <think>, <reasons>, <answer> from generated text for think_content and reasons.
        Always use ground-truth is_bonafide from the dataset so SFT targets and
        real/fake distribution plots are correct (not the model's possibly wrong prediction)."""
        samples = []
        raw_samples = getattr(base_ds, "samples", None)
        if raw_samples is None:
            return samples
        for i, idx in enumerate(indices):
            if idx >= len(raw_samples):
                continue
            item = raw_samples[idx]
            orig_path = item.get("original_path", item.get("audio_path", ""))
            raw_text = reasoning_texts[i] if i < len(reasoning_texts) else ""
            think_content, reasons_list, _ = _parse_reasoning_output(raw_text)
            # Always use ground-truth label so distribution and SFT targets stay correct
            is_bonafide = item.get("is_bonafide")
            if reasons_list:
                reasons = {r: True for r in reasons_list}
            else:
                reasons = item.get("reasons")  # fallback to raw
            sample: Dict[str, Any] = {
                "audio_id": item.get("audio_id"),
                "original_path": orig_path,
                "is_bonafide": is_bonafide,
                "reasons": reasons,
                "reasoning": think_content,
            }
            if lengths_chars is not None and i < len(lengths_chars):
                sample["length_chars"] = lengths_chars[i]
            if lengths_tokens is not None and i < len(lengths_tokens):
                sample["length_tokens"] = lengths_tokens[i]
            samples.append(sample)
        return samples

    def _balance_rollouts_by_real_fake(
        self,
        base_ds,
        intermediate_indices: List[int],
        all_lengths_chars: List[int],
        all_lengths_tokens: List[int],
        all_audio_ids: List[Any],
        all_reasoning_texts: List[str],
        distillation_iter: int,
        _is_main: bool,
    ) -> Tuple[List[int], List[int], List[int], List[Any], List[str]]:
        """Undersample the major class (real or fake) so the intermediate dataset is balanced by rollouts.
        Keeps all rollouts from the minor class and randomly samples the same count from the major;
        rollouts with unknown is_bonafide are kept. Uses distillation_iter as seed for reproducibility."""
        raw_samples = getattr(base_ds, "samples", None)
        if raw_samples is None:
            if _is_main:
                print("[Distillation] balance_by_rollouts: base dataset has no 'samples'; skipping.", flush=True)
            return intermediate_indices, all_lengths_chars, all_lengths_tokens, all_audio_ids, all_reasoning_texts

        n = len(intermediate_indices)
        real_positions: List[int] = []
        fake_positions: List[int] = []
        unknown_positions: List[int] = []
        for i in range(n):
            idx = intermediate_indices[i]
            if idx >= len(raw_samples):
                unknown_positions.append(i)
                continue
            is_bonafide = raw_samples[idx].get("is_bonafide")
            if is_bonafide is True:
                real_positions.append(i)
            elif is_bonafide is False:
                fake_positions.append(i)
            else:
                unknown_positions.append(i)

        n_real, n_fake = len(real_positions), len(fake_positions)
        target = min(n_real, n_fake)
        if target == 0:
            if _is_main:
                print(
                    f"[Distillation] balance_by_rollouts: one class has no rollouts (real={n_real}, fake={n_fake}); no undersampling.",
                    flush=True,
                )
            return intermediate_indices, all_lengths_chars, all_lengths_tokens, all_audio_ids, all_reasoning_texts

        rng = random.Random(distillation_iter)
        if n_real <= n_fake:
            keep_real = real_positions
            keep_fake = rng.sample(fake_positions, target)
        else:
            keep_fake = fake_positions
            keep_real = rng.sample(real_positions, target)
        keep_indices = sorted(keep_real + keep_fake + unknown_positions)

        out_indices = [intermediate_indices[i] for i in keep_indices]
        out_chars = [all_lengths_chars[i] for i in keep_indices]
        out_tokens = [all_lengths_tokens[i] for i in keep_indices]
        out_aids = [all_audio_ids[i] for i in keep_indices]
        out_reasoning = [all_reasoning_texts[i] for i in keep_indices]

        if _is_main:
            print(
                f"[Distillation] balance_by_rollouts: real={n_real}, fake={n_fake}, unknown={len(unknown_positions)} -> "
                f"kept {len(keep_real) + len(keep_fake) + len(unknown_positions)} rollouts (balanced real/fake={target} each).",
                flush=True,
            )
        return out_indices, out_chars, out_tokens, out_aids, out_reasoning

    def _all_reduce_int(self, value: int, op=dist.ReduceOp.SUM) -> int:
        """All-reduce an int across ranks. Returns same value on all ranks.
        Uses all_gather + local reduce to avoid NCCL int64 all_reduce corruption."""
        if not dist.is_initialized():
            return value
        t = torch.tensor([value], dtype=torch.int64, device=self.device)
        gathered = [torch.zeros(1, dtype=torch.int64, device=self.device) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, t)
        vals = [int(g.item()) for g in gathered]
        if op == dist.ReduceOp.SUM:
            return sum(vals)
        elif op == dist.ReduceOp.MAX:
            return max(vals)
        return sum(vals)  # default SUM

    def _broadcast_stop_signal(self, should_stop: bool) -> bool:
        """Broadcast stop signal across ranks. Returns True if any rank wants to stop."""
        if not dist.is_initialized():
            return should_stop
        t = torch.tensor([1 if should_stop else 0], dtype=torch.int64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())

    def _all_gather_candidates(self, local_candidates: List[int]) -> List[int]:
        """Gather candidate indices from all ranks. Returns combined list on all ranks."""
        if not dist.is_initialized():
            return local_candidates
        all_lists = [None] * dist.get_world_size()
        dist.all_gather_object(all_lists, local_candidates)
        out = []
        for lst in all_lists:
            out.extend(lst or [])
        return sorted(set(out))

    def _dlog(self, msg: str, force_rank: Optional[int] = None) -> None:
        """Log with rank prefix in distributed mode for debugging sync issues."""
        if dist.is_initialized():
            r = dist.get_rank() if force_rank is None else force_rank
            print(f"[Distillation] [rank{r}] {msg}", flush=True)
        else:
            print(f"[Distillation] {msg}", flush=True)

    def _run_skeptic_phase(
        self, base_ds, total_size: int, distillation_iter: int, _is_main: bool
    ) -> Tuple[List[int], int, int]:
        """Run skeptic filtering: up to skeptic_chunk_size samples per attempt, retry up to max_skeptic_attempts.
        min_skeptic_dataset_size is checked across all ranks (global total).
        Uses deterministic num_batches (no collective) so all ranks stay in sync."""
        candidate_indices: List[int] = []
        n_rollouts_skeptic = 0
        _distributed = dist.is_initialized()
        cumulative_samples_processed = 0
        cumulative_batch_idx = 0

        for attempt in range(self.max_skeptic_attempts):
            if _distributed:
                dist.barrier()
            chunk_start = attempt * (self.skeptic_chunk_size or total_size)
            if chunk_start >= total_size:
                break
            chunk_end = min(chunk_start + (self.skeptic_chunk_size or total_size), total_size)
            chunk_indices = list(range(chunk_start, chunk_end))
            subset_ds = Subset(base_ds, chunk_indices)

            if _is_main:
                print(
                    f"[Distillation] Phase 1 (attempt {attempt + 1}/{self.max_skeptic_attempts}): "
                    f"Skeptic filtering on chunk [{chunk_start},{chunk_end}) ({len(chunk_indices)} samples)...",
                    flush=True,
                )
            attempt_candidates: List[int] = []
            samples_processed = 0
            batch_idx_skeptic = 0

            if _distributed:
                sampler = DistributedSampler(subset_ds, shuffle=True)
                batch_sampler = BatchSampler(sampler, self.dataloader.batch_size, drop_last=False)
                batch_iter = (
                    (
                        batch_indices,
                        self.dataloader.collate_fn([subset_ds[i] for i in batch_indices]),
                    )
                    for batch_indices in batch_sampler
                )
            else:
                skeptic_loader = DataLoader(
                    subset_ds,
                    batch_size=self.dataloader.batch_size,
                    shuffle=True,
                    collate_fn=self.dataloader.collate_fn,
                    num_workers=self.dataloader.num_workers,
                    pin_memory=getattr(self.dataloader, "pin_memory", False),
                )
                batch_iter = ((None, batch) for batch in skeptic_loader)

            batch_list = list(batch_iter)
            # Deterministic num_batches: same formula on all ranks, no collective needed
            if _distributed:
                world_size = dist.get_world_size()
                batch_size = self.dataloader.batch_size
                samples_per_rank = (len(chunk_indices) + world_size - 1) // world_size
                num_batches = max(1, (samples_per_rank + batch_size - 1) // batch_size)
            else:
                num_batches = len(batch_list)

            if _distributed:
                dist.barrier()

            for i in tqdm(range(num_batches), desc="Skeptic filtering"):
                # Check stop condition BEFORE processing batch to stop immediately
                # All ranks must participate in collectives; compute before _is_main branch
                n_candidates_total_before = self._all_reduce_int(len(attempt_candidates))
                n_cumulative_before = len(candidate_indices) + n_candidates_total_before
                
                # Sync all ranks before checking stop: fast rank waits for slow rank so they exit together
                if _distributed:
                    dist.barrier()
                
                # Broadcast stop signal so all ranks break together when any reaches target
                if n_cumulative_before >= self.min_skeptic_dataset_size:
                    if _is_main:
                        print(f"[Distillation] Reached min_skeptic_dataset_size={self.min_skeptic_dataset_size} (global total), stopping skeptic early.", flush=True)
                    break
                
                if i < len(batch_list):
                    batch_indices_or_none, batch = batch_list[i]
                    batch = self._move_to_device(batch, self.device)
                    batch_sz = len(batch.get("prompts", batch.get("audio_ids", [1])))
                    try:
                        accepted_local = self._filter_skeptic_only(batch)
                    except torch.cuda.OutOfMemoryError:
                        if self.device.type != "cuda":
                            raise
                        accepted_local = self._handle_out_of_memory_skeptic(
                            batch, batch_sz, batch_idx_skeptic, "skeptic"
                        )
                    if _distributed:
                        for local_idx in accepted_local:
                            subset_idx = batch_indices_or_none[local_idx]
                            global_idx = chunk_indices[subset_idx]
                            attempt_candidates.append(global_idx)
                    else:
                        for local_idx in accepted_local:
                            global_idx = chunk_start + samples_processed + local_idx
                            attempt_candidates.append(global_idx)
                    n_rollouts_skeptic += batch_sz * self.skeptic_num_gens
                    samples_processed += batch_sz
                    self._free_batch_audio(batch)
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    batch_idx_skeptic += 1

                # All ranks must participate in collectives; compute before _is_main branch
                n_candidates_total = self._all_reduce_int(len(attempt_candidates))
                samples_processed_total = self._all_reduce_int(samples_processed)
                # Use cumulative progress so progress.json never "resets" when starting a new attempt
                n_cumulative = len(candidate_indices) + n_candidates_total
                samples_cumulative = cumulative_samples_processed + samples_processed_total
                batch_idx_cumulative = cumulative_batch_idx + batch_idx_skeptic
                if _is_main and hasattr(self.logger, "save_dataset_forming_progress"):
                    self.logger.save_dataset_forming_progress(
                        distillation_iter, "skeptic",
                        batch_idx=batch_idx_cumulative,
                        samples_processed=samples_cumulative,
                        n_candidates=n_cumulative,
                        n_accumulated=n_cumulative,
                        attempt=attempt,
                    )

                if self.skeptic_chunk_size is not None and samples_processed_total >= self.skeptic_chunk_size:
                    break

            if _distributed:
                dist.barrier()
            # Extend candidate_indices with new candidates from this attempt (don't replace!)
            attempt_candidates_gathered = self._all_gather_candidates(attempt_candidates)
            candidate_indices.extend(attempt_candidates_gathered)
            # Remove duplicates and sort to keep candidate_indices clean
            candidate_indices = sorted(set(candidate_indices))
            n_rollouts_skeptic_total = self._all_reduce_int(n_rollouts_skeptic)
            samples_processed_total_for_log = self._all_reduce_int(samples_processed)
            cumulative_samples_processed += samples_processed_total_for_log
            cumulative_batch_idx += batch_idx_skeptic
            if _is_main:
                print(f"[Distillation] Attempt {attempt + 1}: Raw={samples_processed_total_for_log}, Candidates={len(candidate_indices)} (cumulative), rollouts={n_rollouts_skeptic_total}", flush=True)
            if len(candidate_indices) >= self.min_skeptic_dataset_size:
                break

        if _distributed:
            dist.barrier()
        n_rollouts_final = self._all_reduce_int(n_rollouts_skeptic)
        cumulative_samples_final = self._all_reduce_int(cumulative_samples_processed)
        return candidate_indices, n_rollouts_final, cumulative_samples_final

    def _handle_out_of_memory_skeptic(
        self, batch: Dict[str, Any], batch_size: int, batch_idx: int, phase_name: str
    ) -> List[int]:
        """On OOM: split batch and retry. Returns combined accepted local indices."""
        _is_main = not dist.is_initialized() or dist.get_rank() == 0
        if _is_main:
            print(
                f"[OOM] CUDA out of memory at {phase_name} batch {batch_idx} (batch_size={batch_size}). "
                "Splitting batch and retrying.",
                flush=True,
            )
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        accepted: List[int] = []
        n_splits = min(2, batch_size)
        sub_size = (batch_size + n_splits - 1) // n_splits
        for start in range(0, batch_size, sub_size):
            end = min(start + sub_size, batch_size)
            if start >= end:
                continue
            sub_batch = split_batch(batch, start, end)
            try:
                sub_accepted = self._filter_skeptic_only(sub_batch)
                for local_idx in sub_accepted:
                    accepted.append(start + local_idx)
            except torch.cuda.OutOfMemoryError:
                if _is_main:
                    print(
                        f"[OOM] Sub-batch ({start}:{end}) still OOM; skipping sub-batch.",
                        flush=True,
                    )
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        return accepted

    def _handle_out_of_memory_main(
        self,
        batch: Dict[str, Any],
        batch_size: int,
        batch_idx: int,
        distillation_iter: int,
        phase_name: str,
    ) -> Tuple[List[int], List[int], List[int], List[Any], List[str], Optional[List[Dict[str, Any]]]]:
        """On OOM: split batch and retry. Returns same structure as _filter_main_only."""
        _is_main = not dist.is_initialized() or dist.get_rank() == 0
        if _is_main:
            print(
                f"[OOM] CUDA out of memory at {phase_name} batch {batch_idx} (batch_size={batch_size}). "
                "Splitting batch and retrying.",
                flush=True,
            )
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        all_accepted: List[int] = []
        all_lengths_chars: List[int] = []
        all_lengths_tokens: List[int] = []
        all_audio_ids: List[Any] = []
        all_reasoning: List[str] = []
        all_verdicts: List[Dict[str, Any]] = [] if self.save_presampling else []
        n_splits = min(2, batch_size)
        sub_size = (batch_size + n_splits - 1) // n_splits
        for start in range(0, batch_size, sub_size):
            end = min(start + sub_size, batch_size)
            if start >= end:
                continue
            sub_batch = split_batch(batch, start, end)
            try:
                accepted, lchars, ltokens, aids, reasoning, verdicts = self._filter_main_only(
                    sub_batch, distillation_iter
                )
                for local_idx in accepted:
                    all_accepted.append(start + local_idx)
                all_lengths_chars.extend(lchars)
                all_lengths_tokens.extend(ltokens)
                all_audio_ids.extend(aids)
                all_reasoning.extend(reasoning)
                if verdicts is not None:
                    all_verdicts.extend(verdicts)
            except torch.cuda.OutOfMemoryError:
                if _is_main:
                    print(
                        f"[OOM] Sub-batch ({start}:{end}) still OOM; skipping sub-batch.",
                        flush=True,
                    )
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        verdicts_out = all_verdicts if self.save_presampling else None
        return all_accepted, all_lengths_chars, all_lengths_tokens, all_audio_ids, all_reasoning, verdicts_out

    def _free_batch_audio(self, batch: Dict[str, Any]) -> None:
        """Remove heavy tensors (audio, etc.) from batch to free memory. Keeps prompts, text, audio_ids."""
        heavy_keys = [
            "audio", "input_ids", "attention_mask", "pixel_values", "image",
            "input_features", "feature_attention_mask", "spectrogram", "raw_wav",
        ]
        for k in heavy_keys:
            if k in batch:
                del batch[k]

    def _create_phase2_loader(self, base_ds, candidate_indices: List[int]) -> DataLoader:
        """Create loader for phase 2: main filtering on candidate subset."""
        subset = Subset(base_ds, candidate_indices)
        return DataLoader(
            subset,
            batch_size=self.dataloader.batch_size,
            shuffle=True,
            collate_fn=self.dataloader.collate_fn,
            num_workers=self.dataloader.num_workers,
            pin_memory=getattr(self.dataloader, "pin_memory", False),
        )

    def _filter_skeptic_only(self, batch: Dict[str, Any]) -> List[int]:
        """Skeptic filter only: correct margin + length. Returns local indices that pass."""
        prompts = batch.get("prompts")
        batch_size = len(prompts) if prompts else len(batch.get("audio_ids", [1]))
        all_texts, _, _, _ = self._generate(batch, self.skeptic_num_gens)
        inputs = batch.get("text", [""] * batch_size)
        gt_raw = batch.get("text", batch.get("reasoning", [""] * batch_size))
        gt = [g if g is not None else "" for g in gt_raw]
        num_gens = self.skeptic_num_gens
        inputs_expanded = inputs * num_gens
        gt_expanded = gt * num_gens
        texts_for_judge = texts_for_log(self.unwrapped_model, all_texts)
        judge_out = self.judge.score(inputs_expanded, texts_for_judge, gt_expanded)
        per_sample = judge_out["meta"]["per_sample"]

        def _count_tokens(t: str) -> int:
            return self.unwrapped_model.count_tokens(t) if hasattr(self.unwrapped_model, "count_tokens") else len(t) // 4

        accepted = []
        for b in range(batch_size):
            correct_flags = []
            rollout_texts = []
            for g in range(num_gens):
                idx = g * batch_size + b
                correct_flags.append(per_sample[idx].get("is_correct", False))
                rollout_texts.append(texts_for_judge[idx])
            n_correct = sum(1 for c in correct_flags if c)
            if not (self.skeptic_min_correct <= n_correct <= self.skeptic_max_correct):
                continue
            at_least_one = False
            for t in rollout_texts:
                nc, nt = len(t), _count_tokens(t)
                text_len_ok = (
                    (self.skeptic_min_text_length is None or self.skeptic_min_text_length <= nc) and
                    (self.skeptic_max_text_length is None or nc <= self.skeptic_max_text_length)
                )
                token_len_ok = (
                    (self.skeptic_min_tokens_len is None or self.skeptic_min_tokens_len <= nt) and
                    (self.skeptic_max_tokens_len is None or nt <= self.skeptic_max_tokens_len)
                )
                if text_len_ok and token_len_ok:
                    at_least_one = True
                    break
            if not at_least_one:
                continue
            accepted.append(b)
        return accepted

    def _filter_main_only(
        self, batch: Dict[str, Any], distillation_iter: int
    ) -> Tuple[List[int], List[int], List[int], List[Any], List[str], Optional[List[Dict[str, Any]]]]:
        """Main filter only: correctness, length, relative. Returns (accepted_indices, lengths_chars, lengths_tokens, audio_ids, reasoning_texts, verdicts)."""
        prompts = batch.get("prompts")
        batch_size = len(prompts) if prompts else len(batch.get("audio_ids", [1]))

        all_texts, _, _, _ = self._generate(batch, self.num_generations)

        inputs = batch.get("text", [""] * batch_size)
        gt_raw = batch.get("text", batch.get("reasoning", [""] * batch_size))
        gt = [g if g is not None else "" for g in gt_raw]
        num_gens = self.num_generations
        inputs_expanded = inputs * num_gens
        gt_expanded = gt * num_gens
        texts_for_judge = texts_for_log(self.unwrapped_model, all_texts)
        judge_out = self.judge.score(inputs_expanded, texts_for_judge, gt_expanded)
        per_sample = judge_out["meta"]["per_sample"]

        ref_median_chars = None
        ref_median_tokens = None
        if self.prev_lengths_chars:
            ref_median_chars = sorted(self.prev_lengths_chars)[len(self.prev_lengths_chars) // 2]
        if self.prev_lengths_tokens:
            ref_median_tokens = sorted(self.prev_lengths_tokens)[len(self.prev_lengths_tokens) // 2]

        accepted = []
        lengths_chars = []
        lengths_tokens = []
        reasoning_texts: List[str] = []
        audio_ids_batch = batch.get("audio_ids", [None] * batch_size)
        verdicts: List[Dict[str, Any]] = [] if self.save_presampling else []

        def _count_tokens(t: str) -> int:
            return self.unwrapped_model.count_tokens(t) if hasattr(self.unwrapped_model, "count_tokens") else len(t) // 4

        for b in range(batch_size):
            correct_flags = []
            rollout_texts = []
            for g in range(num_gens):
                idx = g * batch_size + b
                meta = per_sample[idx]
                correct_flags.append(meta.get("is_correct", False))
                rollout_texts.append(texts_for_judge[idx])
            n_correct = sum(1 for c in correct_flags if c)

            valid_texts = [
                rollout_texts[g]
                for g in range(num_gens)
                if correct_flags[g] and per_sample[g * batch_size + b].get("format_ok", True)
            ]
            texts_for_len = valid_texts if valid_texts else rollout_texts
            char_len = max(len(t) for t in texts_for_len) if texts_for_len else 0
            token_len = max(_count_tokens(t) for t in texts_for_len) if texts_for_len else 0
            rel_char_ratio = char_len / ref_median_chars if ref_median_chars else None
            rel_token_ratio = token_len / ref_median_tokens if ref_median_tokens else None
            main_correct = self.min_correct <= n_correct <= self.max_correct

            if self.save_presampling:
                verdict: Dict[str, Any] = {
                    "audio_id": audio_ids_batch[b],
                    "char_len": char_len,
                    "token_len": token_len,
                    "n_correct": n_correct,
                    "is_correct": main_correct,
                    "relative_text_length_ratio": rel_char_ratio,
                    "relative_token_length_ratio": rel_token_ratio,
                }

            if not (self.min_correct <= n_correct <= self.max_correct):
                if self.save_presampling:
                    verdict["verdict"] = "rejected"
                    verdict["reason"] = f"main_correctness: n_correct={n_correct} not in [{self.min_correct},{self.max_correct}]"
                    verdicts.append(verdict)
                continue

            if not valid_texts:
                if self.save_presampling:
                    verdict["verdict"] = "rejected"
                    verdict["reason"] = "no_valid_rollouts: no correct+format_ok rollout"
                    verdicts.append(verdict)
                continue

            # Collect ALL valid rollouts that pass length + relative filters (per min_correct requirement)
            passing_texts: List[str] = []
            for text in valid_texts:
                nchars = len(text)
                ntokens = _count_tokens(text)
                text_len_ok = (
                    (self.min_text_length is None or self.min_text_length <= nchars) and
                    (self.max_text_length is None or nchars <= self.max_text_length)
                )
                token_len_ok = (
                    (self.min_tokens_len is None or self.min_tokens_len <= ntokens) and
                    (self.max_tokens_len is None or ntokens <= self.max_tokens_len)
                )
                if not (text_len_ok and token_len_ok):
                    continue
                if (ref_median_chars is not None and 
                    self.relative_text_length_ratio is not None and 
                    nchars < self.relative_text_length_ratio * ref_median_chars):
                    continue
                if (ref_median_tokens is not None and 
                    self.relative_text_length_ratio is not None and 
                    ntokens < self.relative_text_length_ratio * ref_median_tokens):
                    continue
                passing_texts.append(text)

            if not passing_texts:
                if self.save_presampling:
                    verdict["verdict"] = "rejected"
                    verdict["reason"] = f"main_length: no rollout passes length (chars [{self.min_text_length},{self.max_text_length}], tokens [{self.min_tokens_len},{self.max_tokens_len}])"
                    verdicts.append(verdict)
                continue

            # Deduplicate: if rollouts have exact duplicates for THIS SAMPLE, only keep one
            seen_texts = set()
            deduplicated_texts = []
            for text in passing_texts:
                if text not in seen_texts:
                    seen_texts.add(text)
                    deduplicated_texts.append(text)
            passing_texts = deduplicated_texts

            if self.save_presampling:
                nchars = max(len(t) for t in passing_texts)
                ntokens = max(_count_tokens(t) for t in passing_texts)
                verdict["char_len"] = nchars
                verdict["token_len"] = ntokens
                verdict["relative_text_length_ratio"] = nchars / ref_median_chars if ref_median_chars else None
                verdict["relative_token_length_ratio"] = ntokens / ref_median_tokens if ref_median_tokens else None
                verdict["verdict"] = "accepted"
                verdict["reason"] = ""
                verdicts.append(verdict)

            for passing_text in passing_texts:
                accepted.append(b)
                lengths_chars.append(len(passing_text))
                lengths_tokens.append(_count_tokens(passing_text))
                reasoning_texts.append(passing_text)

        accepted_audio_ids = [audio_ids_batch[b] for b in accepted]
        return accepted, lengths_chars, lengths_tokens, accepted_audio_ids, reasoning_texts, verdicts if self.save_presampling else None

    def _generate(
        self, batch: Dict[str, Any], num_gens: int
    ) -> Tuple[List[str], torch.Tensor, int, int]:
        """Generate num_gens per sample in chunks. Returns (all_texts, completion_ids, pad_id, batch_size).
        Uses generation_chunk_size: chunk_size=1 => batch_size rollouts per forward pass."""
        prompts = batch.get("prompts")
        batch_size = len(prompts) if prompts else len(batch.get("audio_ids", [1]))
        chunk_size = max(1, min(self.generation_chunk_size, num_gens))
        use_amp = self.amp and self.device.type == "cuda"
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        all_texts: List[str] = []
        for chunk_start in tqdm(range(0, num_gens, chunk_size), desc="Generating rollouts"):
            chunk_end = min(chunk_start + chunk_size, num_gens)
            for _ in range(chunk_end - chunk_start):
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type=self.device.type,
                        enabled=use_amp,
                        dtype=autocast_dtype,
                    ):
                        out = self.unwrapped_model.generate(batch, self.gen_cfg, prompts=prompts, return_outputs=True)
                        if isinstance(out, tuple):
                            texts = out[0]
                        else:
                            texts = out
                    all_texts.extend(texts)
            if chunk_end < num_gens and self.device.type == "cuda":
                torch.cuda.empty_cache()
        pad_id = 0
        if hasattr(self.unwrapped_model, "get_pad_token_id"):
            pid = self.unwrapped_model.get_pad_token_id()
            if pid is not None:
                pad_id = pid
        return all_texts, torch.tensor(0), pad_id, batch_size

    def _move_to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._move_to_device(v, device) for v in batch]
        return batch
