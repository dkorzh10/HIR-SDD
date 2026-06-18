"""Samplers for training: stateful, distributed, and length-grouped."""
import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, List

from .audio_length import get_audio_duration_sec


class StatefulSampler(Sampler):
    def __init__(self, dataset, shuffle=False, iters_per_epoch=None, batch_size=1, stateful=True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.iters_per_epoch = iters_per_epoch
        self.batch_size = batch_size
        self.stateful = stateful
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        n = len(self.dataset)
        if n == 0:
            return iter([])
        if self.shuffle:
            g = torch.Generator()
            seed = (self.seed + self.epoch) if self.stateful else self.seed
            g.manual_seed(seed)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        if self.iters_per_epoch is not None:
            samples_per_epoch = self.iters_per_epoch * self.batch_size
            start_idx = (self.epoch * samples_per_epoch) % n if self.stateful else 0
            res_indices = []
            for i in range(samples_per_epoch):
                res_indices.append(indices[(start_idx + i) % n])
            return iter(res_indices)

        return iter(indices)

    def __len__(self):
        if self.iters_per_epoch is not None:
            return self.iters_per_epoch * self.batch_size
        return len(self.dataset)


class StatefulDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, iters_per_epoch=None, batch_size=1, stateful=True, seed: int = 0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=int(seed))
        self.iters_per_epoch = iters_per_epoch
        self.batch_size = batch_size
        self.stateful = stateful

    def __iter__(self):
        n = len(self.dataset)
        if n == 0:
            return iter([])
        if self.shuffle:
            g = torch.Generator()
            seed = (self.seed + self.epoch) if self.stateful else self.seed
            g.manual_seed(seed)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        if self.iters_per_epoch is not None:
            samples_per_epoch_total = self.iters_per_epoch * self.batch_size * self.num_replicas
            start_idx = (self.epoch * samples_per_epoch_total) % n if self.stateful else 0
            epoch_indices = []
            for i in range(samples_per_epoch_total):
                epoch_indices.append(indices[(start_idx + i) % n])
            rank_indices = epoch_indices[self.rank :: self.num_replicas]
            return iter(rank_indices)

        return super().__iter__()

    def __len__(self):
        if self.iters_per_epoch is not None:
            return self.iters_per_epoch * self.batch_size
        return super().__len__()


class LengthGroupedDistributedSampler(DistributedSampler):
    """
    Distributed sampler that balances load by grouping samples by audio length.
    Each rank gets a mix of short and long samples (similar total duration).
    Preserves iters_per_epoch and stateful logic from StatefulDistributedSampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=False,
        iters_per_epoch: Optional[int] = None,
        batch_size: int = 1,
        stateful: bool = True,
        length_fn=None,
        seed: int = 0,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=int(seed))
        self.iters_per_epoch = iters_per_epoch
        self.batch_size = batch_size
        self.stateful = stateful
        self._length_fn = length_fn
        self._lengths_cache: Optional[List[float]] = None

    def _get_lengths(self) -> List[float]:
        """Get length (seconds) for each sample. Cached."""
        if self._lengths_cache is not None:
            return self._lengths_cache
        dataset = self.dataset
        n = len(dataset)
        if hasattr(dataset, "samples") and dataset.samples:
            lengths = []
            for i in range(n):
                item = dataset.samples[i]
                path = item.get("original_path", "")
                if self._length_fn is not None:
                    lengths.append(self._length_fn(path))
                else:
                    lengths.append(get_audio_duration_sec(path))
            self._lengths_cache = lengths
        else:
            self._lengths_cache = [1.0] * n
        return self._lengths_cache

    def __iter__(self):
        n = len(self.dataset)
        if n == 0:
            return iter([])
        lengths = self._get_lengths()
        indices_with_length = list(zip(range(n), lengths))
        indices_with_length.sort(key=lambda x: x[1])
        sorted_indices = [i for i, _ in indices_with_length]

        if self.iters_per_epoch is not None:
            samples_per_epoch_total = self.iters_per_epoch * self.batch_size * self.num_replicas
            start_idx = (self.epoch * samples_per_epoch_total) % n if self.stateful else 0
            epoch_indices = []
            for i in range(samples_per_epoch_total):
                epoch_indices.append(sorted_indices[(start_idx + i) % n])
        else:
            epoch_indices = sorted_indices

        if self.shuffle:
            g = torch.Generator()
            seed = (self.seed + self.epoch) if self.stateful else self.seed
            g.manual_seed(seed)
            perm = torch.randperm(len(epoch_indices), generator=g).tolist()
            epoch_indices = [epoch_indices[j] for j in perm]

        rank_indices = epoch_indices[self.rank :: self.num_replicas]
        return iter(rank_indices)

    def __len__(self):
        if self.iters_per_epoch is not None:
            return self.iters_per_epoch * self.batch_size
        n = len(self.dataset)
        return (n + self.num_replicas - 1 - self.rank) // self.num_replicas
