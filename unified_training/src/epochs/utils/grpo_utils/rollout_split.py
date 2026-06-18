"""
Chunked generation and logits for OOM: split by rollouts to reduce peak memory.

Flow:
1. Generate rollouts in chunks (e.g. 2 at a time) -> collect texts (lightweight), completion_ids
2. Score + compute advantages (lightweight, once we have texts)
3. Compute logits and backward in chunks (split by rollout groups, gradient accumulation)

All functions are pure/modular and testable.
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def chunk_ranges_for_rollouts(
    batch_size: int, num_gens: int, chunk_size: int
) -> List[Tuple[int, int]]:
    """
    Return (start_idx, end_idx) for each chunk in completion_ids (gen-major order).
    index = g * batch_size + b.
    chunk_size = number of generations per chunk; each chunk has chunk_size * batch_size rollouts.
    """
    if chunk_size >= num_gens:
        return [(0, batch_size * num_gens)]
    ranges = []
    for chunk_start in range(0, num_gens, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_gens)
        start_idx = chunk_start * batch_size
        end_idx = chunk_end * batch_size
        ranges.append((start_idx, end_idx))
    return ranges


def chunk_ranges_by_rollout_count(
    total_rollouts: int, max_rollouts_per_chunk: int
) -> List[Tuple[int, int]]:
    """
    Return (start_idx, end_idx) for each chunk, where each chunk has at most
    max_rollouts_per_chunk rollouts. Enables true 1-rollout-at-a-time processing.
    """
    if max_rollouts_per_chunk >= total_rollouts:
        return [(0, total_rollouts)]
    return [
        (i, min(i + max_rollouts_per_chunk, total_rollouts))
        for i in range(0, total_rollouts, max_rollouts_per_chunk)
    ]


def generate_rollouts_in_chunks(
    model,
    batch: Dict[str, Any],
    num_gens: int,
    gen_cfg: Dict[str, Any],
    prompts: Optional[List] = None,
    chunk_size: int = 1,
    amp: bool = True,
) -> Tuple[List[str], torch.Tensor, int, int]:
    """
    Generate num_gens rollouts per sample in chunks of chunk_size to reduce peak memory.
    Returns (all_texts, all_completion_ids, pad_id, batch_size).
    Data layout: gen-major order, index = g * batch_size + b.
    """
    prompts = prompts or batch.get("prompts")
    batch_size = len(prompts) if prompts else len(batch.get("audio_ids", [1]))
    autocast_dtype = torch.float16
    if not amp and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16

    all_texts: List[str] = []
    all_completion_ids_list: List[torch.Tensor] = []
    pad_id = 0
    if hasattr(model, "model") and hasattr(model.model, "llama_tokenizer"):
        tok = model.model.llama_tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    for chunk_start in range(0, num_gens, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_gens)
        n_this_chunk = chunk_end - chunk_start
        for _ in range(n_this_chunk):
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=amp, dtype=autocast_dtype):
                    texts, completion_ids, _ = model.generate(
                        batch, gen_cfg, prompts=prompts, return_outputs=True
                    )
            all_texts.extend(texts)
            all_completion_ids_list.append(completion_ids)
        if chunk_end < num_gens:
            torch.cuda.empty_cache()

    max_len = max(t.size(1) for t in all_completion_ids_list)
    padded = [
        F.pad(t, (0, max_len - t.size(1)), value=pad_id) for t in all_completion_ids_list
    ]
    all_completion_ids = torch.cat(padded, dim=0)
    return all_texts, all_completion_ids, pad_id, batch_size


def compute_grpo_loss_for_chunk(
    chunk_completion_ids: torch.Tensor,
    chunk_advantages: torch.Tensor,
    ref_logits: torch.Tensor,
    current_logits: torch.Tensor,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GRPO loss for a chunk of rollouts. Returns (pg_loss, kl_div).
    Handles 2D and 3D logits.
    """
    T = chunk_completion_ids.size(1)
    if ref_logits.dim() == 3:
        ref_logits = ref_logits[:, -T:, :]
        current_logits = current_logits[:, -T:, :]
    current_log_probs = F.log_softmax(current_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    token_log_probs = torch.gather(
        current_log_probs, dim=-1, index=chunk_completion_ids.unsqueeze(-1)
    ).squeeze(-1)
    ref_token_log_probs = torch.gather(
        ref_log_probs, dim=-1, index=chunk_completion_ids.unsqueeze(-1)
    ).squeeze(-1)
    mask = (chunk_completion_ids != pad_id).float()
    total_log_probs = (token_log_probs * mask).sum(dim=-1)
    ref_total_log_probs = (ref_token_log_probs * mask).sum(dim=-1)
    pg_loss = -(chunk_advantages * total_log_probs).mean()
    kl_div = (total_log_probs - ref_total_log_probs).mean()
    return pg_loss, kl_div
