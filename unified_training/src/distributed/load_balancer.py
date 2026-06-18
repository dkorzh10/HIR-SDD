"""
Generic work load balancer: leaders (finished GPUs) send work to losers (still training).

When a leader has work:
- If a loser is free: send directly
- If all busy: store in queue (rank 0 holds queue). When first loser gets free, it gets the work.
The leader can proceed after putting in queue.

SkepticLoadBalancer is the GRPO-specific alias (work format: batch + reused_rollouts).
"""
from typing import Any, Dict, List, Optional
import struct
import torch
import torch.distributed as dist

from .work_item import serialize_work_item, deserialize_work_item

# Tags for distributed messages
_TAG_WORK = 100
_MAX_BUF = 10 * 1024 * 1024  # 10MB for work item transfer


def _send_bytes(data: bytes, dst: int, tag: int, device: torch.device):
    """Send bytes to dst. Sends length (4 bytes) then payload. Empty data = length 0."""
    length = struct.pack(">I", len(data) if data else 0)
    payload = length + (data or b"")
    if len(payload) < _MAX_BUF:
        payload = payload + b"\x00" * (_MAX_BUF - len(payload))
    else:
        payload = payload[:_MAX_BUF]
    tensor = torch.ByteTensor(list(payload)).to(device)
    dist.send(tensor, dst, tag=tag)


def _recv_bytes(src: int, tag: int, device: torch.device) -> bytes:
    """Receive bytes from src. Blocking. Returns empty if length was 0."""
    buf = torch.zeros(_MAX_BUF, dtype=torch.uint8, device=device)
    dist.recv(buf, src, tag=tag)
    data = buf.cpu().numpy().tobytes()
    if len(data) >= 4:
        (size,) = struct.unpack(">I", data[:4])
        if size > 0 and size < _MAX_BUF:
            return data[4 : 4 + size]
    return b""


class GenericWorkLoadBalancer:
    """
    Coordinates transfer of work items from leader GPUs to loser GPUs.
    Rank 0 holds the central queue.
    Work items are arbitrary dicts (serialized via torch.save).
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        max_queue_size: int = 64,
        serialize_fn=None,
        deserialize_fn=None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.max_queue_size = max_queue_size
        self._serialize = serialize_fn or serialize_work_item
        self._deserialize = deserialize_fn or deserialize_work_item
        self._queue: List[bytes] = []  # Serialized work items (rank 0 only)
        self._leader_pending: List[Dict[str, Any]] = []
        self._is_leader = False
        self._is_loser = False
        self._all_done = False

    def set_leader(self):
        """This rank has finished and will help by producing work for losers."""
        self._is_leader = True
        self._is_loser = False

    def set_loser(self):
        """This rank still needs work and may receive from leaders."""
        self._is_loser = True
        self._is_leader = False

    def set_all_done(self):
        """Signal that all losers have finished; stop load balancing."""
        self._all_done = True

    def leader_put_work(self, work_item: Dict[str, Any]) -> bool:
        """
        Leader found work. Add to pending queue (will be sent in next exchange_round).
        Returns True if added (False if queue overflows).
        """
        if not self._is_leader or self._all_done:
            return False
        if len(self._leader_pending) >= self.max_queue_size:
            return False
        self._leader_pending.append(work_item)
        return True

    def leader_pending_count(self) -> int:
        """Number of work items the leader has pending to send."""
        return len(self._leader_pending)

    def loser_request_work(self) -> Optional[Dict[str, Any]]:
        """
        Loser requests work. Runs one exchange round.
        Returns work item or None.
        """
        if not self._is_loser:
            return None
        received = self.exchange_round([])
        return received[0] if received else None

    def exchange_round(
        self,
        extra_leader_work: Optional[List[Dict[str, Any]]] = None,
        my_finished: bool = False,
    ) -> tuple[List[Dict[str, Any]], bool]:
        """
        Run one round of work exchange. All ranks must call this together (barrier inside).
        - Leaders: work from _leader_pending + extra_leader_work is sent
        - Losers: receive work
        - my_finished: True if this rank has finished its epoch (entered helper loop)
        Returns: (received_work_items, all_done). all_done=True when all ranks have my_finished=True.
        """
        if not dist.is_initialized():
            return [], True

        extra = extra_leader_work or []
        leader_work = self._leader_pending + extra
        self._leader_pending = []

        work_to_send = [self._serialize(w) for w in leader_work]
        num_to_send = len(work_to_send)
        num_needed = 1 if self._is_loser else 0

        status = torch.tensor(
            [int(self._is_leader), int(self._is_loser), num_to_send, num_needed, int(my_finished)],
            dtype=torch.long,
            device=self.device,
        )
        all_status = [torch.zeros_like(status) for _ in range(self.world_size)]
        dist.all_gather(all_status, status)

        received: List[Dict[str, Any]] = []

        if self.rank == 0:
            self._coordinator_receive(all_status, work_to_send)
            if self._is_loser and self._queue:
                data = self._queue.pop(0)
                received.append(self._deserialize(data, self.device))
            self._coordinator_send(all_status, received)
        else:
            for data in work_to_send:
                _send_bytes(data, 0, _TAG_WORK, self.device)
            if self._is_loser:
                work = self._request_from_rank0()
                if work is not None:
                    received.append(work)

        all_done = all(all_status[r][4].item() == 1 for r in range(self.world_size))
        if all_done:
            self._all_done = True
        dist.barrier()
        return received, all_done

    def _coordinator_receive(self, all_status: List[torch.Tensor], my_work: List[bytes]):
        """Rank 0: receive work from leaders."""
        for r in range(1, self.world_size):
            if all_status[r][0].item() == 1 and all_status[r][2].item() > 0:
                for _ in range(int(all_status[r][2].item())):
                    data = _recv_bytes(r, _TAG_WORK, self.device)
                    if data and len(data) > 0:
                        self._queue.append(data)
        if self._is_leader and my_work:
            self._queue.extend(my_work)

    def _coordinator_send(self, all_status: List[torch.Tensor], my_received: List[Dict[str, Any]]):
        """Rank 0: send work to losers."""
        for r in range(1, self.world_size):
            if all_status[r][1].item() == 1 and all_status[r][3].item() > 0:
                if self._queue:
                    data = self._queue.pop(0)
                    _send_bytes(data, r, _TAG_WORK, self.device)
                else:
                    _send_bytes(b"", r, _TAG_WORK, self.device)

    def _request_from_rank0(self) -> Optional[Dict[str, Any]]:
        """Non-rank-0 loser: receive work from rank 0."""
        data = _recv_bytes(0, _TAG_WORK, self.device)
        if data and len(data) > 0:
            return self._deserialize(data, self.device)
        return None


class SkepticLoadBalancer(GenericWorkLoadBalancer):
    """
    GRPO-specific load balancer: work items are {"batch": ..., "reused_rollouts": ...}.
    Uses default serialize/deserialize (torch.save/load).
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        max_queue_size: int = 64,
    ):
        super().__init__(
            rank=rank,
            world_size=world_size,
            device=device,
            max_queue_size=max_queue_size,
            serialize_fn=serialize_work_item,
            deserialize_fn=deserialize_work_item,
        )
