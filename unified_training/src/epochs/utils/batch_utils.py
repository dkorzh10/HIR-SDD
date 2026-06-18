"""Utilities for splitting batches (e.g. for OOM recovery)."""
from typing import Any, Dict, List


def split_batch(batch: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    """
    Return a sub-batch containing elements [start:end].
    Handles dicts of tensors (sliced on dim=0), lists (sliced), and nested structures.
    """
    if start >= end:
        raise ValueError(f"Invalid slice start={start} end={end}")
    out = {}
    for key, value in batch.items():
        out[key] = _slice_batch_value(value, start, end)
    return out


def _slice_batch_value(value: Any, start: int, end: int) -> Any:
    if isinstance(value, dict):
        return {k: _slice_batch_value(v, start, end) for k, v in value.items()}
    if isinstance(value, list):
        return value[start:end]
    if hasattr(value, "shape") and hasattr(value, "__getitem__"):
        # Tensor or array: slice first dimension
        return value[start:end]
    return value
