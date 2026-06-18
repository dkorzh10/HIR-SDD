from typing import Any, Optional, List, Dict, Tuple
import gc
import re
import torch
import torch.distributed as dist
import torch.nn.functional as F
from .base import TrainEpoch
from ..judges.base import Judge
from .utils.text_utils import texts_for_log
from .utils.batch_utils import split_batch
from src.distributed import SkepticLoadBalancer
from .utils.grpo_utils import (
    chunk_ranges_for_rollouts,
    chunk_ranges_by_rollout_count,
    generate_rollouts_in_chunks,
    compute_grpo_loss_for_chunk,
)
from torch.nn.utils.rnn import pad_sequence

_DEFAULT_GRPO_GEN_CFG = {"do_sample": True, "top_p": 0.9, "max_new_tokens": 5000}
_SKEPTIC_NUM_GENS = 3  # Number of generations for skeptic check (no grad)


def _parse_answer_from_text(text: str) -> Optional[str]:
    """Parse Real/Fake from <answer>...</answer> tag. Returns 'real', 'fake', or None. Only accepts exact format."""
    match = re.search(r"<answer>(.*?)</answer>", str(text), re.DOTALL)
    if not match:
        return None
    ans = match.group(1).strip().lower()
    return ans if ans in ("real", "fake") else None


class GRPOTrainEpoch(TrainEpoch):
    def __init__(self, model, dataloader, logger, optimizer, judge: Judge,
                 num_generations: int = 4, beta: float = 0.1,
                 iters_per_epoch: Optional[int] = None, accum_grad_iters: int = 1,
                 gen_cfg: Optional[Dict[str, Any]] = None, amp: bool = True,
                 filter_controversial: bool = False,
                 skeptic_buffer_size: Optional[int] = None,
                 skeptic_batch_size: Optional[int] = None,
                 **kwargs):
        super().__init__(model, dataloader, logger, optimizer, amp=amp, **kwargs)
        self.judge = judge
        self.num_generations = num_generations
        self.beta = beta  # KL penalty coefficient
        self.iters_per_epoch = iters_per_epoch
        self.accum_grad_iters = accum_grad_iters
        self.gen_cfg = {**_DEFAULT_GRPO_GEN_CFG, **(gen_cfg or {})}
        self.filter_controversial = filter_controversial
        self.skeptic_batch_size = skeptic_batch_size  # Samples per check batch (rollouts to filter at a time)
        self.skeptic_buffer_size = skeptic_buffer_size  # Min controversial samples before main GRPO loop starts

    def _get_model(self):
        return self.model.module if isinstance(
            self.model, torch.nn.parallel.DistributedDataParallel
        ) else self.model

    def _get_autocast_dtype(self) -> torch.dtype:
        if self.scaler is None and self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _skeptic_generate_and_score(
        self, batch: Dict[str, Any], num_check_gens: int = _SKEPTIC_NUM_GENS, chunk_size: Optional[int] = None
    ) -> Tuple[List[int], List[int], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run num_check_gens generations (no grad), judge, return (controversial_indices, rejected_indices, batch, check_logs, reused_rollouts).
        Controversial = mix of Real and Fake across gens. Rejected = all same (all Real or all Fake).
        check_logs: list of {audio_id, rollouts: [text1..text3], has_passed} for logging skeptic_epoch files.
        reused_rollouts: only for controversial samples, {texts, judge_meta, completion_ids} for GRPO reuse.
        chunk_size: if set, generate in chunks to reduce peak memory (OOM handling)."""
        prompts = batch.get("prompts")
        batch_size = len(prompts) if prompts else len(batch.get("audio_ids", [1]))
        audio_ids = batch.get("audio_ids", [f"unknown_{j}" for j in range(batch_size)])

        if chunk_size is not None and chunk_size < num_check_gens:
            all_texts, all_completion_ids_stacked, pad_id, _ = generate_rollouts_in_chunks(
                self.unwrapped_model,
                batch,
                num_check_gens,
                self.gen_cfg,
                prompts=prompts,
                chunk_size=chunk_size,
                amp=self.amp,
            )
            all_completion_ids = [
                all_completion_ids_stacked[g * batch_size : (g + 1) * batch_size]
                for g in range(num_check_gens)
            ]
        else:
            autocast_dtype = self._get_autocast_dtype()
            all_texts = []
            all_completion_ids = []
            for _ in range(num_check_gens):
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=self.amp, dtype=autocast_dtype):
                        texts, completion_ids, _ = self.unwrapped_model.generate(
                            batch, self.gen_cfg, prompts=prompts, return_outputs=True
                        )
                all_texts.extend(texts)
                all_completion_ids.append(completion_ids)

        inputs = batch.get("text", [""] * batch_size)
        gt_raw = batch.get("text", batch.get("reasoning", [""] * batch_size))
        gt = [g if g is not None else "" for g in gt_raw]
        inputs_expanded = inputs * num_check_gens
        gt_expanded = gt * num_check_gens
        texts_for_judge = texts_for_log(self.unwrapped_model, all_texts)
        judge_out = self.judge.score(inputs_expanded, texts_for_judge, gt_expanded)
        per_sample = judge_out["meta"]["per_sample"]

        controversial = []
        rejected = []
        check_logs: List[Dict[str, Any]] = []
        reused_rollouts: List[Dict[str, Any]] = []
        for b in range(batch_size):
            correct_flags = []
            rollouts_b = []
            judge_meta_b = []
            completion_ids_b = []
            for g in range(num_check_gens):
                idx = g * batch_size + b
                meta = per_sample[idx]
                correct_flags.append(meta.get("is_correct", False))
                rollouts_b.append(texts_for_judge[idx])
                judge_meta_b.append(meta)
                completion_ids_b.append(all_completion_ids[g][b : b + 1])
            n_correct = sum(1 for c in correct_flags if c)
            n_incorrect = len(correct_flags) - n_correct
            has_passed = n_correct > 0 and n_incorrect > 0
            if has_passed:
                controversial.append(b)
                reused_rollouts.append({
                    "texts": rollouts_b,
                    "judge_meta": judge_meta_b,
                    "completion_ids": completion_ids_b,
                })
            else:
                rejected.append(b)
            check_logs.append({
                "audio_id": audio_ids[b],
                "rollouts": rollouts_b,
                "has_passed": has_passed,
            })
        return controversial, rejected, batch, check_logs, reused_rollouts

    def _step_1_generate(
        self, batch: Dict[str, Any], num_gens: Optional[int] = None
    ) -> Tuple[List[str], torch.Tensor, int, int]:
        """1. Generate multiple completions per sample. Returns (all_texts, all_completion_ids, pad_id, batch_size).
        num_gens: if None, use self.num_generations."""
        n = num_gens or self.num_generations
        prompts = batch.get("prompts")
        batch_size = len(prompts) if prompts else len(batch.get("audio_ids", [1]))
        autocast_dtype = self._get_autocast_dtype()
        all_texts = []
        all_completion_ids = []
        for _ in range(n):
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self.amp, dtype=autocast_dtype):
                    texts, completion_ids, _ = self.unwrapped_model.generate(
                        batch, self.gen_cfg, prompts=prompts, return_outputs=True
                    )
            all_texts.extend(texts)
            all_completion_ids.append(completion_ids)
        pad_id = getattr(self.unwrapped_model.model, "llama_tokenizer", None)
        pad_id = pad_id.pad_token_id if pad_id else 0
        max_len = max(t.size(1) for t in all_completion_ids)
        padded_ids = [F.pad(t, (0, max_len - t.size(1)), value=pad_id) for t in all_completion_ids]
        all_completion_ids = torch.cat(padded_ids, dim=0)
        return all_texts, all_completion_ids, pad_id, batch_size

    def _process_batch_with_reused_rollouts(
        self,
        batch: Dict[str, Any],
        reused_rollouts: List[Dict[str, Any]],
        backward: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """Run GRPO with 3 reused check rollouts + (num_generations-3) new generations. reused_rollouts: list of {texts, judge_meta, completion_ids} per sample."""
        prompts = batch.get("prompts")
        batch_size = len(prompts) if prompts else len(batch.get("audio_ids", [1]))
        pad_id = 0
        if hasattr(self.unwrapped_model, "model") and hasattr(self.unwrapped_model.model, "llama_tokenizer"):
            tok = self.unwrapped_model.model.llama_tokenizer
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

        # 1. Combine reused (3) + new (num_generations-3) = num_generations
        all_texts: List[str] = []
        all_completion_ids_list: List[torch.Tensor] = []
        all_judge_meta: List[Dict] = []

        for b in range(batch_size):
            reused = reused_rollouts[b]
            all_texts.extend(reused["texts"])
            for cids in reused["completion_ids"]:
                all_completion_ids_list.append(cids.squeeze(0))
            all_judge_meta.extend(reused["judge_meta"])

        # 2. Generate (num_generations - 3) new per sample (same pattern as _step_1_generate)
        num_extra = self.num_generations - _SKEPTIC_NUM_GENS
        autocast_dtype = self._get_autocast_dtype()
        new_texts = []
        new_completion_ids = []
        for _ in range(num_extra):
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self.amp, dtype=autocast_dtype):
                    texts, completion_ids, _ = self.unwrapped_model.generate(
                        batch, self.gen_cfg, prompts=batch.get("prompts"), return_outputs=True
                    )
            new_texts.extend(texts)
            new_completion_ids.append(completion_ids)

        inputs = batch.get("text", [""] * batch_size)
        gt_raw = batch.get("text", batch.get("reasoning", [""] * batch_size))
        gt = [g if g is not None else "" for g in gt_raw]
        inputs_expanded = inputs * num_extra
        gt_expanded = gt * num_extra
        new_texts_for_judge = texts_for_log(self.unwrapped_model, new_texts)
        judge_out = self.judge.score(inputs_expanded, new_texts_for_judge, gt_expanded)
        new_per_sample = judge_out["meta"]["per_sample"]

        for b in range(batch_size):
            for g in range(num_extra):
                idx = g * batch_size + b
                all_texts.append(new_texts_for_judge[idx])
                all_judge_meta.append(new_per_sample[idx])
                cid = new_completion_ids[g][b : b + 1]
                all_completion_ids_list.append(cid.squeeze(0))

        # 3. Pad and stack completion_ids
        padded = []
        for t in all_completion_ids_list:
            if isinstance(t, torch.Tensor):
                t = t.to(self.device)
                if t.dim() == 2:
                    t = t.squeeze(0)
            else:
                t = torch.tensor(t, device=self.device)
            padded.append(t)
        max_len = max(t.numel() for t in padded)
        padded = [F.pad(t.reshape(-1), (0, max_len - t.numel()), value=pad_id) for t in padded]
        all_completion_ids = torch.stack(padded, dim=0)

        texts_for_judge_and_log = texts_for_log(self.unwrapped_model, all_texts)
        G = self.num_generations
        B = batch_size

        def _skeptic_meta_idx(b: int, g: int) -> int:
            """all_judge_meta layout: reused sample-major (b*3+g), then new gen-major (3*B + (g-3)*B+b)."""
            if g < _SKEPTIC_NUM_GENS:
                return b * _SKEPTIC_NUM_GENS + g
            return B * _SKEPTIC_NUM_GENS + (g - _SKEPTIC_NUM_GENS) * B + b

        judge_meta_gen_major = [all_judge_meta[_skeptic_meta_idx(b, g)] for g in range(G) for b in range(B)]
        texts_gen_major = [texts_for_judge_and_log[_skeptic_meta_idx(b, g)] for g in range(G) for b in range(B)]
        scores = [judge_meta_gen_major[g * B + b]["score"] for b in range(B) for g in range(G)]
        rewards = torch.tensor(scores, device=self.device).view(batch_size, G)
        gt_batch = [g if g is not None else "" for g in (batch.get("text", batch.get("reasoning", [""] * batch_size)))]

        advantages = self._step_3_advantages(rewards, batch_size)
        ref_logits, current_logits = self._step_4_ref_and_current_logits(batch, all_completion_ids, num_gens=G)
        loss, kl_div = self._step_5_6_grpo_loss(
            all_completion_ids, ref_logits, current_logits, pad_id, advantages
        )
        if backward:
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        logs = self._step_7_build_logs(
            batch, batch_size, rewards, advantages, judge_meta_gen_major, texts_gen_major, gt_batch, num_gens=G
        )
        return loss, kl_div, rewards, logs

    def _process_batch_with_reused_rollouts_chunking(
        self,
        batch: Dict[str, Any],
        reused_rollouts: List[Dict[str, Any]],
        backward: bool,
        new_gen_chunk_size: int,
        logits_chunk_size: int,
        max_rollouts_per_chunk: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """Skeptic GRPO with chunked new generation and logits. Reduces peak memory for OOM handling.
        max_rollouts_per_chunk: if set, use rollout-level chunking (1 rollout at a time when=1)."""
        prompts = batch.get("prompts")
        batch_size = len(prompts) if prompts else len(batch.get("audio_ids", [1]))
        pad_id = 0
        if hasattr(self.unwrapped_model, "model") and hasattr(self.unwrapped_model.model, "llama_tokenizer"):
            tok = self.unwrapped_model.model.llama_tokenizer
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        G = self.num_generations
        B = batch_size
        num_extra = G - _SKEPTIC_NUM_GENS

        # 1. Reused (3) + new in chunks
        all_texts: List[str] = []
        all_completion_ids_list: List[torch.Tensor] = []
        all_judge_meta: List[Dict] = []

        for b in range(B):
            reused = reused_rollouts[b]
            all_texts.extend(reused["texts"])
            for cids in reused["completion_ids"]:
                all_completion_ids_list.append(cids.squeeze(0))
            all_judge_meta.extend(reused["judge_meta"])

        new_texts, new_completion_ids, _, _ = generate_rollouts_in_chunks(
            self.unwrapped_model,
            batch,
            num_extra,
            self.gen_cfg,
            prompts=batch.get("prompts"),
            chunk_size=new_gen_chunk_size,
            amp=self.amp,
        )
        new_texts_for_judge = texts_for_log(self.unwrapped_model, new_texts)
        inputs = batch.get("text", [""] * batch_size)
        gt_raw = batch.get("text", batch.get("reasoning", [""] * batch_size))
        gt = [g if g is not None else "" for g in gt_raw]
        inputs_expanded = inputs * num_extra
        gt_expanded = gt * num_extra
        judge_out = self.judge.score(inputs_expanded, new_texts_for_judge, gt_expanded)
        new_per_sample = judge_out["meta"]["per_sample"]

        for b in range(B):
            for g in range(num_extra):
                idx = g * B + b
                all_texts.append(new_texts_for_judge[idx])
                all_judge_meta.append(new_per_sample[idx])
                all_completion_ids_list.append(new_completion_ids[g * B + b].to(self.device))

        def _skeptic_meta_idx(b: int, g: int) -> int:
            if g < _SKEPTIC_NUM_GENS:
                return b * _SKEPTIC_NUM_GENS + g
            return B * _SKEPTIC_NUM_GENS + (g - _SKEPTIC_NUM_GENS) * B + b

        judge_meta_gen_major = [all_judge_meta[_skeptic_meta_idx(b, g)] for g in range(G) for b in range(B)]
        texts_gen_major = [all_texts[_skeptic_meta_idx(b, g)] for g in range(G) for b in range(B)]
        texts_gen_major = texts_for_log(self.unwrapped_model, texts_gen_major)

        scores = [judge_meta_gen_major[g * B + b]["score"] for b in range(B) for g in range(G)]
        rewards = torch.tensor(scores, device=self.device).view(batch_size, G)
        gt_batch = [g if g is not None else "" for g in (batch.get("text", batch.get("reasoning", [""] * batch_size)))]

        advantages = self._step_3_advantages(rewards, batch_size)

        padded = []
        for t in all_completion_ids_list:
            t = t.to(self.device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=self.device)
            if t.dim() == 2:
                t = t.squeeze(0)
            padded.append(t.reshape(-1))
        max_len = max(t.numel() for t in padded)
        padded = [F.pad(t, (0, max_len - t.numel()), value=pad_id) for t in padded]
        all_completion_ids = torch.stack(padded, dim=0)

        reorder_idx = torch.tensor(
            [_skeptic_meta_idx(b, g) for g in range(G) for b in range(B)],
            device=all_completion_ids.device,
        )
        all_completion_ids = all_completion_ids[reorder_idx]

        total_rollouts = B * G
        if max_rollouts_per_chunk is not None:
            chunk_ranges = chunk_ranges_by_rollout_count(total_rollouts, max_rollouts_per_chunk)
        else:
            chunk_ranges = chunk_ranges_for_rollouts(B, G, logits_chunk_size)
        loss_sum = 0.0
        kl_sum = 0.0
        autocast_dtype = self._get_autocast_dtype()
        for start_idx, end_idx in chunk_ranges:
            chunk_completion_ids = all_completion_ids[start_idx:end_idx].to(self.device)
            chunk_advantages = advantages[start_idx:end_idx].to(self.device)
            n_rollouts_chunk = end_idx - start_idx
            if max_rollouts_per_chunk is not None:
                batch_expanded = self._merge_batches([
                    split_batch(batch, (start_idx + j) % B, (start_idx + j) % B + 1)
                    for j in range(n_rollouts_chunk)
                ])
            else:
                batch_expanded = self._expand_batch(batch, n_rollouts_chunk // B)
            self._get_model().swap_weights()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self.amp, dtype=autocast_dtype):
                    ref_logits = self.unwrapped_model.compute_logits_for_completions(
                        batch_expanded, chunk_completion_ids
                    )
            self._get_model().swap_weights()
            with torch.amp.autocast("cuda", enabled=self.amp, dtype=autocast_dtype):
                current_logits = self.unwrapped_model.compute_logits_for_completions(
                    batch_expanded, chunk_completion_ids
                )
            pg_loss, kl_div_chunk = compute_grpo_loss_for_chunk(
                chunk_completion_ids, chunk_advantages, ref_logits, current_logits, pad_id
            )
            loss_chunk = (pg_loss + self.beta * kl_div_chunk) / (
                self.accum_grad_iters * len(chunk_ranges)
            )
            if backward:
                if self.scaler:
                    self.scaler.scale(loss_chunk).backward()
                else:
                    loss_chunk.backward()
            loss_sum += pg_loss.item()
            kl_sum += kl_div_chunk.item()
            torch.cuda.empty_cache()
        loss = torch.tensor(loss_sum / len(chunk_ranges), device=self.device)
        kl_div = torch.tensor(kl_sum / len(chunk_ranges), device=self.device)
        logs = self._step_7_build_logs(
            batch, batch_size, rewards, advantages, judge_meta_gen_major, texts_gen_major, gt_batch, num_gens=G
        )
        return loss, kl_div, rewards, logs

    def _handle_skeptic_oom(
        self,
        grpo_batch: Dict[str, Any],
        reused_rollouts: List[Dict[str, Any]],
        grpo_iter: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]]:
        """On OOM during skeptic GRPO: retry with chunked generation and logits.
        Reduces to 1 rollout at a time ([batch]->[sample]->[rollout]) before giving up."""
        batch_size = len(grpo_batch.get("prompts") or grpo_batch.get("audio_ids", [1]))
        total_rollouts = batch_size * self.num_generations
        print(
            f"[OOM] Skeptic GRPO batch OOM at iter {grpo_iter} (batch_size={batch_size}). "
            "Retrying with chunked generation and logits.",
            flush=True,
        )
        self.optimizer.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
        new_chunk = max(1, (self.num_generations - _SKEPTIC_NUM_GENS) // 2)
        logits_chunk = max(1, self.num_generations // 2)
        max_rollouts = None
        while True:
            try:
                return self._process_batch_with_reused_rollouts_chunking(
                    grpo_batch, reused_rollouts, backward=True,
                    new_gen_chunk_size=new_chunk, logits_chunk_size=logits_chunk,
                    max_rollouts_per_chunk=max_rollouts,
                )
            except torch.cuda.OutOfMemoryError:
                if max_rollouts is not None and max_rollouts <= 1:
                    print("[OOM] Skeptic GRPO 1-rollout-at-a-time still OOM; skipping batch.", flush=True)
                    self.optimizer.zero_grad()
                    gc.collect()
                    torch.cuda.empty_cache()
                    return None
                if max_rollouts is None and new_chunk <= 1 and logits_chunk <= 1:
                    max_rollouts = max(1, total_rollouts // 2)
                    print(f"[OOM] Retrying with max_rollouts_per_chunk={max_rollouts}.", flush=True)
                elif max_rollouts is not None:
                    max_rollouts = max(1, max_rollouts // 2)
                    print(f"[OOM] Retrying with max_rollouts_per_chunk={max_rollouts}.", flush=True)
                else:
                    if new_chunk > 1:
                        new_chunk = max(1, new_chunk // 2)
                    else:
                        logits_chunk = max(1, logits_chunk // 2)
                    print(f"[OOM] Retrying with new_gen_chunk={new_chunk}, logits_chunk={logits_chunk}.", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()

    def _step_2_score(
        self, batch: Dict[str, Any], batch_size: int, all_texts: List[str]
    ) -> Tuple[torch.Tensor, List[Dict], List[str], List[str]]:
        """2. Score all generations with Judge. Returns (rewards, judge_meta, texts_for_judge_and_log, gt)."""
        inputs = batch.get("text", [""] * batch_size)
        gt_raw = batch.get("text", batch.get("reasoning", [""] * batch_size))
        gt = [g if g is not None else "" for g in gt_raw]
        inputs_expanded = inputs * self.num_generations
        gt_expanded = gt * self.num_generations
        texts_for_judge_and_log = texts_for_log(self.unwrapped_model, all_texts)
        judge_out = self.judge.score(inputs_expanded, texts_for_judge_and_log, gt_expanded)
        per_sample = judge_out["meta"]["per_sample"]
        # all_texts order is [s0_g0, s1_g0, s0_g1, s1_g1, ...] (gen-major: index = g*B+b)
        # Reorder so rewards[b,g] = per_sample[g*batch_size+b]
        G = self.num_generations
        scores = [per_sample[g * batch_size + b]["score"] for b in range(batch_size) for g in range(G)]
        rewards = torch.tensor(scores, device=self.device).view(batch_size, G)
        return rewards, per_sample, texts_for_judge_and_log, gt

    def _step_3_advantages(self, rewards: torch.Tensor, batch_size: int) -> torch.Tensor:
        """3. Compute advantages (normalized per group)."""
        advantages_2d = torch.zeros_like(rewards)
        for b in range(batch_size):
            group_rewards = rewards[b]
            advantages_2d[b] = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)
        # Flatten in gen-major order [s0_g0, s1_g0, s0_g1, ...] to match all_completion_ids
        return advantages_2d.permute(1, 0).contiguous().view(-1)

    def _step_4_ref_and_current_logits(
        self, batch: Dict[str, Any], all_completion_ids: torch.Tensor, num_gens: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """4. Compute ref and current logits for GRPO. num_gens: if None, use self.num_generations."""
        n = num_gens or self.num_generations
        autocast_dtype = self._get_autocast_dtype()
        self._get_model().swap_weights()
        with torch.no_grad():
            batch_expanded = self._expand_batch(batch, n)
            with torch.amp.autocast("cuda", enabled=self.amp, dtype=autocast_dtype):
                ref_logits = self.unwrapped_model.compute_logits_for_completions(batch_expanded, all_completion_ids)
        self._get_model().swap_weights()
        batch_expanded = self._expand_batch(batch, n)
        with torch.amp.autocast("cuda", enabled=self.amp, dtype=autocast_dtype):
            current_logits = self.unwrapped_model.compute_logits_for_completions(batch_expanded, all_completion_ids)
        T = all_completion_ids.size(1)
        def safe_slice_logits(logits):
            if logits.dim() == 3:
                return logits[:, -T:, :]
            elif logits.dim() == 2:
                return logits
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
        current_logits = safe_slice_logits(current_logits).to(self.device)
        ref_logits = safe_slice_logits(ref_logits).to(self.device)
        return ref_logits, current_logits

    def _step_5_6_grpo_loss(
        self,
        all_completion_ids: torch.Tensor,
        ref_logits: torch.Tensor,
        current_logits: torch.Tensor,
        pad_id: int,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """5.–6. Compute log-probs and GRPO loss. Returns (loss, kl_div)."""
        if current_logits.dim() == 3:
            current_log_probs = F.log_softmax(current_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            all_completion_ids = all_completion_ids.to(self.device)
            token_log_probs = torch.gather(current_log_probs, dim=-1, index=all_completion_ids.unsqueeze(-1)).squeeze(-1)
            ref_token_log_probs = torch.gather(ref_log_probs, dim=-1, index=all_completion_ids.unsqueeze(-1)).squeeze(-1)
        else:
            token_log_probs = current_logits
            ref_token_log_probs = ref_logits
        mask = (all_completion_ids != pad_id).float()
        total_log_probs = (token_log_probs * mask).sum(dim=-1)
        ref_total_log_probs = (ref_token_log_probs * mask).sum(dim=-1)
        pg_loss = -(advantages * total_log_probs).mean()
        kl_div = (total_log_probs - ref_total_log_probs).mean()
        loss = (pg_loss + self.beta * kl_div) / self.accum_grad_iters
        return loss, kl_div

    def _step_7_build_logs(
        self,
        batch: Dict[str, Any],
        batch_size: int,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        judge_meta: List[Dict],
        texts_for_judge_and_log: List[str],
        gt: List[str],
        num_gens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """7. Build per-sample logs for judge scores. num_gens: if None, use self.num_generations."""
        G = num_gens or self.num_generations
        B = batch_size
        audio_ids = batch.get("audio_ids", [f"unknown_{j}" for j in range(B)])
        is_bonafide_batch = batch.get("is_bonafide")
        logs = []
        # all_texts/judge_meta/advantages use gen-major order: index = g*B+b
        for b in range(B):
            is_bonafide = is_bonafide_batch[b].item() if is_bonafide_batch is not None and hasattr(is_bonafide_batch[b], "item") else None
            logs.append({
                "audio_id": audio_ids[b],
                "gt": gt[b],
                "is_bonafide": is_bonafide,
                "generations": [
                    {
                        "text": texts_for_judge_and_log[g * B + b],
                        "reward": rewards[b, g].item(),
                        "advantage": advantages[g * B + b].item(),
                        "format_ok": judge_meta[g * B + b]["format_ok"],
                        "is_correct": judge_meta[g * B + b]["is_correct"],
                        "correct_reasons_overlap": judge_meta[g * B + b].get("correct_reasons_overlap"),
                    }
                    for g in range(G)
                ]
            })
        return logs

    def _process_batch(
        self, batch: Dict[str, Any], backward: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """Run GRPO forward for one batch; optionally run backward. Returns (loss, kl_div, rewards, logs)."""
        all_texts, all_completion_ids, pad_id, batch_size = self._step_1_generate(batch)
        rewards, judge_meta, texts_for_judge_and_log, gt = self._step_2_score(batch, batch_size, all_texts)
        advantages = self._step_3_advantages(rewards, batch_size)
        ref_logits, current_logits = self._step_4_ref_and_current_logits(batch, all_completion_ids)
        loss, kl_div = self._step_5_6_grpo_loss(
            all_completion_ids, ref_logits, current_logits, pad_id, advantages
        )
        if backward:
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        logs = self._step_7_build_logs(
            batch, batch_size, rewards, advantages, judge_meta, texts_for_judge_and_log, gt
        )
        return loss, kl_div, rewards, logs

    def _process_batch_with_rollout_chunking(
        self, batch: Dict[str, Any], backward: bool, chunk_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """Process batch with generation and logits split by rollout chunks. Reduces peak memory."""
        all_texts, all_completion_ids, pad_id, batch_size = generate_rollouts_in_chunks(
            self.unwrapped_model,
            batch,
            self.num_generations,
            self.gen_cfg,
            prompts=batch.get("prompts"),
            chunk_size=chunk_size,
            amp=self.amp,
        )
        rewards, judge_meta, texts_for_judge_and_log, gt = self._step_2_score(
            batch, batch_size, all_texts
        )
        advantages = self._step_3_advantages(rewards, batch_size)
        autocast_dtype = self._get_autocast_dtype()
        loss_sum = 0.0
        kl_sum = 0.0
        n_chunks = 0
        chunk_ranges = chunk_ranges_for_rollouts(
            batch_size, self.num_generations, chunk_size
        )
        for start_idx, end_idx in chunk_ranges:
            chunk_completion_ids = all_completion_ids[start_idx:end_idx].to(self.device)
            chunk_advantages = advantages[start_idx:end_idx].to(self.device)
            n_rollouts_chunk = (end_idx - start_idx) // batch_size
            batch_expanded = self._expand_batch(batch, n_rollouts_chunk)
            self._get_model().swap_weights()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self.amp, dtype=autocast_dtype):
                    ref_logits = self.unwrapped_model.compute_logits_for_completions(
                        batch_expanded, chunk_completion_ids
                    )
            self._get_model().swap_weights()
            with torch.amp.autocast("cuda", enabled=self.amp, dtype=autocast_dtype):
                current_logits = self.unwrapped_model.compute_logits_for_completions(
                    batch_expanded, chunk_completion_ids
                )
            T = chunk_completion_ids.size(1)
            if ref_logits.dim() == 3:
                ref_logits = ref_logits[:, -T:, :].to(self.device)
                current_logits = current_logits[:, -T:, :].to(self.device)
            pg_loss, kl_div_chunk = compute_grpo_loss_for_chunk(
                chunk_completion_ids, chunk_advantages, ref_logits, current_logits, pad_id
            )
            loss_chunk = (pg_loss + self.beta * kl_div_chunk) / (
                self.accum_grad_iters * len(chunk_ranges)
            )
            if backward:
                if self.scaler:
                    self.scaler.scale(loss_chunk).backward()
                else:
                    loss_chunk.backward()
            loss_sum += pg_loss.item()
            kl_sum += kl_div_chunk.item()
            n_chunks += 1
        loss = torch.tensor(loss_sum / n_chunks, device=self.device)
        kl_div = torch.tensor(kl_sum / n_chunks, device=self.device)
        logs = self._step_7_build_logs(
            batch, batch_size, rewards, advantages, judge_meta, texts_for_judge_and_log, gt
        )
        return loss, kl_div, rewards, logs

    def _handle_out_of_memory(
        self, batch: Dict[str, Any], batch_size: int, iteration: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]]:
        """On OOM: try rollout chunking, halving chunk_size each retry until 1. Skip batch if chunk_size=1 fails."""
        print(
            f"[OOM] CUDA out of memory at iteration {iteration} (batch_size={batch_size}). "
            "Retrying with rollout chunking.",
            flush=True,
        )
        self.optimizer.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
        chunk_size = max(1, self.num_generations // 2)
        while True:
            try:
                return self._process_batch_with_rollout_chunking(
                    batch, backward=True, chunk_size=chunk_size
                )
            except torch.cuda.OutOfMemoryError:
                if chunk_size <= 1:
                    print("[OOM] chunk_size=1 still OOM; skipping batch.", flush=True)
                    self.optimizer.zero_grad()
                    gc.collect()
                    torch.cuda.empty_cache()
                    return None
                chunk_size = max(1, chunk_size // 2)
                print(f"[OOM] Retrying with rollout chunk_size={chunk_size}.", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()

    def run_filter_controversial(self, epoch_num: int):
        """Run GRPO with skeptic filtering: only train on samples where model is inconsistent (mix of correct/incorrect).
        Rejected: all correct or all incorrect (incl. wrong format). Controversial: mix of correct and incorrect.
        Phase 1: Check batch with 3 gens (no grad). Phase 2: Full GRPO on controversial samples only."""
        self.logger.set_epoch(epoch_num, "train")
        self.model.train()

        if hasattr(self.dataloader, "sampler") and hasattr(self.dataloader.sampler, "set_epoch"):
            self.dataloader.sampler.set_epoch(epoch_num)

        self.optimizer.zero_grad()
        check_batch_size = self.skeptic_batch_size  # samples per check batch; None = use dataloader batch as-is
        buffer_size = self.skeptic_buffer_size or self.num_generations  # controversial samples before GRPO loop
        controversial_buffer: List[Dict[str, Any]] = []
        grpo_iter = 0
        epoch_rejected = 0
        epoch_accepted = 0
        rejected_per_iter: List[int] = []
        accepted_per_iter: List[int] = []
        pending_samples: List[Dict[str, Any]] = []  # accumulated for check_batch_size

        _is_main = not dist.is_initialized() or dist.get_rank() == 0
        _skeptic_progress_freq = 5  # print every N check batches
        _use_load_balancer = dist.is_initialized() and self.iters_per_epoch is not None
        load_balancer = None
        if _use_load_balancer:
            load_balancer = SkepticLoadBalancer(
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                device=self.device,
            )
            load_balancer.set_loser()
        if _is_main:
            print(
                f"[Skeptic] Check batch size={check_batch_size or 'dataloader'}, buffer_size={buffer_size}. "
                f"Logging every {_skeptic_progress_freq} check batches."
                + (" Load balancing enabled." if _use_load_balancer else ""),
                flush=True,
            )

        def _run_check_batch(
            batch_to_check: Dict[str, Any],
        ) -> Tuple[List[int], List[int], List[Dict[str, Any]], List[Dict[str, Any]]]:
            try:
                controversial, rejected, _, check_logs, reused_rollouts = self._skeptic_generate_and_score(
                    batch_to_check
                )
                return controversial, rejected, check_logs, reused_rollouts
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda":
                    raise
                if _is_main:
                    print(
                        "[OOM] Skeptic check batch OOM; retrying with chunked generation (chunk_size=1).",
                        flush=True,
                    )
                torch.cuda.empty_cache()
                controversial, rejected, _, check_logs, reused_rollouts = self._skeptic_generate_and_score(
                    batch_to_check, chunk_size=1
                )
                return controversial, rejected, check_logs, reused_rollouts

        # Epoch = iters_per_epoch GRPO batches (not dataloader iters), so validation runs after full GRPO epochs
        for i, batch in enumerate(self.dataloader):
            if self.iters_per_epoch is not None and grpo_iter >= self.iters_per_epoch:
                break

            batch = self._move_to_device(batch, self.device)
            prompts = batch.get("prompts")
            batch_sz = len(prompts) if prompts else len(batch.get("audio_ids", [1]))

            if check_batch_size is not None:
                for idx in range(batch_sz):
                    pending_samples.append(split_batch(batch, idx, idx + 1))
                while len(pending_samples) >= check_batch_size:
                    check_batch = self._merge_batches(pending_samples[:check_batch_size])
                    pending_samples = pending_samples[check_batch_size:]
                    controversial, rejected, check_logs, reused_rollouts = _run_check_batch(check_batch)
                    epoch_rejected += len(rejected)
                    epoch_accepted += len(controversial)
                    rejected_per_iter.append(len(rejected))
                    accepted_per_iter.append(len(controversial))
                    for i, idx in enumerate(controversial):
                        controversial_buffer.append({
                            "batch": split_batch(check_batch, idx, idx + 1),
                            "reused_rollouts": reused_rollouts[i],
                        })
                    check_iter = len(rejected_per_iter) - 1
                    self.logger.log_skeptic_check_scores(check_logs, check_iter=check_iter)
            else:
                controversial, rejected, check_logs, reused_rollouts = _run_check_batch(batch)
                epoch_rejected += len(rejected)
                epoch_accepted += len(controversial)
                rejected_per_iter.append(len(rejected))
                accepted_per_iter.append(len(controversial))
                for i, idx in enumerate(controversial):
                    controversial_buffer.append({
                        "batch": split_batch(batch, idx, idx + 1),
                        "reused_rollouts": reused_rollouts[i],
                    })
                check_iter = len(rejected_per_iter) - 1
                self.logger.log_skeptic_check_scores(check_logs, check_iter=check_iter)

            check_batches_done = len(rejected_per_iter)
            if _is_main and check_batches_done > 0 and check_batches_done % _skeptic_progress_freq == 0:
                print(
                    f"[Skeptic] Check batch {check_batches_done}: rejected={epoch_rejected}, accepted={epoch_accepted}, "
                    f"buffer={len(controversial_buffer)}/{buffer_size}, GRPO batches={grpo_iter}",
                    flush=True,
                )

            if load_balancer is not None:
                received, _ = load_balancer.exchange_round(my_finished=False)
                for w in received:
                    controversial_buffer.append(w)

            while len(controversial_buffer) >= buffer_size and (
                self.iters_per_epoch is None or grpo_iter < self.iters_per_epoch
            ):
                entries = controversial_buffer[:buffer_size]
                controversial_buffer = controversial_buffer[buffer_size:]
                grpo_batch = self._merge_batches([e["batch"] for e in entries])
                reused_rollouts = [e["reused_rollouts"] for e in entries]
                try:
                    loss, kl_div, rewards, logs = self._process_batch_with_reused_rollouts(
                        grpo_batch, reused_rollouts, backward=True
                    )
                except torch.cuda.OutOfMemoryError:
                    if self.device.type != "cuda":
                        raise
                    result = self._handle_skeptic_oom(grpo_batch, reused_rollouts, grpo_iter)
                    if result is None:
                        continue
                    loss, kl_div, rewards, logs = result

                if (grpo_iter + 1) % self.accum_grad_iters == 0:
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
                        if hasattr(self.scheduler, "step") and self.scheduler.__class__.__name__ == "LinearWarmupCosineLRScheduler":
                            self.scheduler.step(epoch_num, grpo_iter)
                        else:
                            self.scheduler.step()

                self.logger.log_judge_scores(logs, grpo_iter=grpo_iter)
                reward_mean = rewards.mean().item() if rewards.dim() > 0 else rewards.item()
                overlap_vals = [
                    g.get("correct_reasons_overlap")
                    for entry in logs
                    for g in entry.get("generations", [])
                ]
                overlap_mean = (
                    sum(v for v in overlap_vals if v is not None) / len(overlap_vals)
                    if overlap_vals and any(v is not None for v in overlap_vals)
                    else None
                )
                log_kw = dict(
                    loss=loss.item() * self.accum_grad_iters,
                    lr=self.optimizer.param_groups[0]["lr"],
                    reward=reward_mean,
                    kl_div=kl_div.item(),
                    skeptic_rejected_per_check_batch=sum(rejected_per_iter[-10:]),  # last 10 check batches
                    skeptic_accepted_per_check_batch=sum(accepted_per_iter[-10:]),
                )
                if overlap_mean is not None:
                    log_kw["correct_reasons_overlap"] = overlap_mean
                self.logger.log(**log_kw)
                grpo_iter += 1
                if _is_main:
                    print(
                        f"[Skeptic] GRPO batch {grpo_iter}: loss={loss.item():.4f}, reward={reward_mean:.4f}, "
                        f"buffer leftover={len(controversial_buffer)}",
                        flush=True,
                    )

        # Flush remaining pending samples (when check_batch_size is set)
        if check_batch_size is not None and pending_samples:
            check_batch = self._merge_batches(pending_samples)
            controversial, rejected, check_logs, reused_rollouts = _run_check_batch(check_batch)
            epoch_rejected += len(rejected)
            epoch_accepted += len(controversial)
            rejected_per_iter.append(len(rejected))
            accepted_per_iter.append(len(controversial))
            for i, idx in enumerate(controversial):
                controversial_buffer.append({
                    "batch": split_batch(check_batch, idx, idx + 1),
                    "reused_rollouts": reused_rollouts[i],
                })
            check_iter = len(rejected_per_iter) - 1
            self.logger.log_skeptic_check_scores(check_logs, check_iter=check_iter)

        # Process any remaining GRPO batches (from flush or main loop)
        while len(controversial_buffer) >= buffer_size and (
            self.iters_per_epoch is None or grpo_iter < self.iters_per_epoch
        ):
            entries = controversial_buffer[:buffer_size]
            controversial_buffer = controversial_buffer[buffer_size:]
            grpo_batch = self._merge_batches([e["batch"] for e in entries])
            reused_rollouts = [e["reused_rollouts"] for e in entries]
            try:
                loss, kl_div, rewards, logs = self._process_batch_with_reused_rollouts(
                    grpo_batch, reused_rollouts, backward=True
                )
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda":
                    raise
                result = self._handle_skeptic_oom(grpo_batch, reused_rollouts, grpo_iter)
                if result is None:
                    continue
                loss, kl_div, rewards, logs = result
            if (grpo_iter + 1) % self.accum_grad_iters == 0:
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
                    if hasattr(self.scheduler, "step") and self.scheduler.__class__.__name__ == "LinearWarmupCosineLRScheduler":
                        self.scheduler.step(epoch_num, grpo_iter)
                    else:
                        self.scheduler.step()
            self.logger.log_judge_scores(logs, grpo_iter=grpo_iter)
            reward_mean = rewards.mean().item() if rewards.dim() > 0 else rewards.item()
            overlap_vals = [g.get("correct_reasons_overlap") for entry in logs for g in entry.get("generations", [])]
            overlap_mean = sum(v for v in overlap_vals if v is not None) / len(overlap_vals) if overlap_vals and any(v is not None for v in overlap_vals) else None
            log_kw = dict(loss=loss.item() * self.accum_grad_iters, lr=self.optimizer.param_groups[0]["lr"], reward=reward_mean, kl_div=kl_div.item(),
                         skeptic_rejected_per_check_batch=sum(rejected_per_iter[-10:]), skeptic_accepted_per_check_batch=sum(accepted_per_iter[-10:]))
            if overlap_mean is not None:
                log_kw["correct_reasons_overlap"] = overlap_mean
            self.logger.log(**log_kw)
            grpo_iter += 1
            if _is_main:
                print(f"[Skeptic] GRPO batch {grpo_iter}: loss={loss.item():.4f}, reward={reward_mean:.4f}, buffer leftover={len(controversial_buffer)}", flush=True)

        if load_balancer is not None:
            load_balancer.set_leader()
            self._run_leader_helper_loop(load_balancer, _run_check_batch, buffer_size, check_batch_size, _is_main)

        if hasattr(self.logger, "set_skeptic_epoch_stats"):
            self.logger.set_skeptic_epoch_stats(epoch_rejected, epoch_accepted)
        if _is_main:
            total = epoch_rejected + epoch_accepted
            print(
                f"[Skeptic] Epoch {epoch_num} done: checked={total}, rejected={epoch_rejected}, "
                f"accepted={epoch_accepted}, GRPO batches={grpo_iter}",
                flush=True,
            )
        self.logger.log_epoch()

    def _run_leader_helper_loop(
        self,
        load_balancer: SkepticLoadBalancer,
        run_check_batch,
        buffer_size: int,
        check_batch_size: Optional[int],
        is_main: bool,
    ):
        """Leader GPUs search for controversial samples and send to losers via load balancer."""
        dataloader_iter = iter(self.dataloader)
        samples_found = 0
        while True:
            _, all_done = load_balancer.exchange_round(my_finished=True)
            if all_done:
                break
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.dataloader)
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    continue
            batch = self._move_to_device(batch, self.device)
            prompts = batch.get("prompts")
            batch_sz = len(prompts) if prompts else len(batch.get("audio_ids", [1]))
            if check_batch_size is not None:
                for idx in range(batch_sz):
                    single = split_batch(batch, idx, idx + 1)
                    check_batch = self._merge_batches([single])
                    controversial, _, _, reused_rollouts = run_check_batch(check_batch)
                    for i in range(len(controversial)):
                        work = {"batch": single, "reused_rollouts": reused_rollouts[i]}
                        if load_balancer.leader_put_work(work):
                            samples_found += 1
            else:
                controversial, rejected, _, reused_rollouts = run_check_batch(batch)
                for i, idx in enumerate(controversial):
                    work = {
                        "batch": split_batch(batch, idx, idx + 1),
                        "reused_rollouts": reused_rollouts[i],
                    }
                    if load_balancer.leader_put_work(work):
                        samples_found += 1
        if is_main:
            print(f"[Skeptic] Load balancing: leader sent {samples_found} samples to losers", flush=True)

    def _merge_batches(self, batches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a list of single-sample batches into one batch."""
        if not batches:
            raise ValueError("Cannot merge empty batch list")
        merged = {}
        for key in batches[0].keys():
            vals = [b[key] for b in batches]
            if isinstance(vals[0], torch.Tensor):
                merged[key] = torch.cat(vals, dim=0)
            elif isinstance(vals[0], list):
                merged[key] = [x for v in vals for x in v]
            else:
                merged[key] = vals[0]
        return merged

    def run(self, epoch_num: int):
        if self.filter_controversial:
            self.run_filter_controversial(epoch_num)
            return

        self.logger.set_epoch(epoch_num, "train")
        self.model.train()

        if hasattr(self.dataloader, "sampler") and hasattr(self.dataloader.sampler, "set_epoch"):
            self.dataloader.sampler.set_epoch(epoch_num)

        self.optimizer.zero_grad()

        for i, batch in enumerate(self.dataloader):
            if self.iters_per_epoch is not None and i >= self.iters_per_epoch:
                break

            batch = self._move_to_device(batch, self.device)
            prompts = batch.get("prompts")
            batch_size = len(prompts) if prompts else len(batch.get("audio_ids", [1]))

            loss = None
            kl_div = None
            rewards = None
            logs = None

            try:
                loss, kl_div, rewards, logs = self._process_batch(batch, backward=True)
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda":
                    raise
                result = self._handle_out_of_memory(batch, batch_size, i)
                if result is None:
                    continue
                loss, kl_div, rewards, logs = result

            if loss is None:
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
                    if hasattr(self.scheduler, "step") and self.scheduler.__class__.__name__ == "LinearWarmupCosineLRScheduler":
                        self.scheduler.step(epoch_num, i)
                    else:
                        self.scheduler.step()

            self.logger.log_judge_scores(logs)
            reward_mean = rewards.mean().item() if rewards.dim() > 0 else rewards.item()
            overlap_vals = [
                g.get("correct_reasons_overlap")
                for entry in logs
                for g in entry.get("generations", [])
            ]
            overlap_mean = (
                sum(v for v in overlap_vals if v is not None) / len(overlap_vals)
                if overlap_vals and any(v is not None for v in overlap_vals)
                else None
            )
            log_kw = dict(
                loss=loss.item() * self.accum_grad_iters,
                lr=self.optimizer.param_groups[0]['lr'],
                reward=reward_mean,
                kl_div=kl_div.item()
            )
            if overlap_mean is not None:
                log_kw["correct_reasons_overlap"] = overlap_mean
            self.logger.log(**log_kw)

        self.logger.log_epoch()

    
    def _expand_batch(self, batch: Dict[str, Any], num_repeats: int) -> Dict[str, Any]:
        """Expand batch to create num_repeats copies of each sample."""
        expanded_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Repeat tensor along batch dimension
                expanded_batch[key] = value.repeat_interleave(num_repeats, dim=0)
            elif isinstance(value, list):
                # Repeat list elements
                expanded_batch[key] = [item for item in value for _ in range(num_repeats)]
            else:
                expanded_batch[key] = value
        return expanded_batch
    
    def _move_to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._move_to_device(v, device) for v in batch]
        return batch
