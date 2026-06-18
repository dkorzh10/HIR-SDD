"""Work item for distributed load balancing: serialize/deserialize work for transfer."""
from typing import Any, Dict
import io
import torch


def serialize_work_item(work_item: Dict[str, Any]) -> bytes:
    """Serialize a work item (e.g. batch + reused_rollouts) to bytes for transfer.
    Uses torch.save which handles tensors efficiently."""
    buffer = io.BytesIO()
    torch.save(work_item, buffer, _use_new_zipfile_serialization=False)
    return buffer.getvalue()


def deserialize_work_item(data: bytes, device: torch.device) -> Dict[str, Any]:
    """Deserialize bytes to work item and move tensors to device."""
    buffer = io.BytesIO(data)
    work_item = torch.load(buffer, map_location=device, weights_only=False)
    return _move_work_item_to_device(work_item, device)


def _move_work_item_to_device(item: Any, device: torch.device) -> Any:
    """Recursively move tensors in work item to device."""
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, dict):
        return {k: _move_work_item_to_device(v, device) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        return type(item)(_move_work_item_to_device(x, device) for x in item)
    return item


class WorkItem:
    """Helpers for work items. GRPO format: {"batch": ..., "reused_rollouts": ...}."""

    @staticmethod
    def create(batch: Dict[str, Any], reused_rollouts: Any) -> Dict[str, Any]:
        """Create a work item from a single-sample batch and its reused rollouts."""
        return {"batch": batch, "reused_rollouts": reused_rollouts}

    @staticmethod
    def is_valid(item: Dict[str, Any]) -> bool:
        """Check if item has required keys for GRPO work."""
        return isinstance(item, dict) and "batch" in item and "reused_rollouts" in item
