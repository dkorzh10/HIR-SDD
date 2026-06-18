"""Logger for Distillation runs. Separate from GRPOLogger. Logs to iteration_i/{dataset_forming,sft,grpo}."""
from typing import Any, Dict, List, Optional
import os
import json

from .reasoning_logger import ReasoningLogger


def _atomic_write_json(path: str, data: Any, indent: Optional[int] = 2) -> None:
    """Write JSON atomically to avoid partial reads when multiple processes may write.
    Uses write-to-tmp + rename which is atomic on POSIX."""
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        if indent is not None:
            json.dump(data, f, indent=indent)
        else:
            json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


class DistillationLogger(ReasoningLogger):
    def __init__(self, log_dir: str, log_freq: int, save_all_predictions: bool = False):
        super().__init__(log_dir, log_freq, save_all_predictions)
        self.base_log_dir = log_dir
        self.distillation_iter: Optional[int] = None
        self.distillation_phase: Optional[str] = None

    def get_iter_dataset_forming_dir(self, iteration: int) -> str:
        """Path to logs/iteration_{i}/dataset_forming/."""
        return os.path.join(self.base_log_dir, f"iteration_{iteration}", "dataset_forming")

    def set_distillation_iter(self, distillation_iter: int, phase: str):
        """Set current distillation iteration and phase. Updates log_dir so all logs go to iteration_i/{phase}/."""
        self.distillation_iter = distillation_iter
        self.distillation_phase = phase
        self.log_dir = os.path.join(self.base_log_dir, f"iteration_{distillation_iter}", phase)
        os.makedirs(self.log_dir, exist_ok=True)

    def save_intermediate_dataset(
        self,
        distillation_iter: int,
        indices: List[int],
        lengths_chars: List[int],
        lengths_tokens: List[int],
        n_raw: int,
        audio_ids: Optional[List[Any]] = None,
    ) -> str:
        """Save final intermediate dataset to iteration_i/dataset_forming/intermediate_dataset_iter_{i}.json."""
        dst_dir = self.get_iter_dataset_forming_dir(distillation_iter)
        path = os.path.join(dst_dir, f"intermediate_dataset_iter_{distillation_iter}.json")
        data = {
            "distillation_iter": distillation_iter,
            "n_raw": n_raw,
            "n_final": len(indices),
            "indices": indices,
            "lengths_chars": lengths_chars,
            "lengths_tokens": lengths_tokens,
        }
        if audio_ids is not None:
            data["audio_ids"] = audio_ids
        _atomic_write_json(path, data)
        return path

    def save_intermediate_dataset_reasoning(
        self,
        distillation_iter: int,
        samples: List[Dict[str, Any]],
        n_raw: int = 0,
    ) -> str:
        """Save intermediate dataset as reasoning-format JSON (list of samples with reasoning traces). SFT loads from this."""
        dst_dir = self.get_iter_dataset_forming_dir(distillation_iter)
        path = os.path.join(dst_dir, f"intermediate_dataset_iter_{distillation_iter}.json")
        _atomic_write_json(path, samples)
        return path

    def save_intermediate_dataset_tmp(
        self,
        distillation_iter: int,
        indices: List[int],
        lengths_chars: List[int],
        lengths_tokens: List[int],
        n_raw: int,
        batch_idx: int,
        audio_ids: Optional[List[Any]] = None,
    ) -> str:
        """Save in-progress intermediate dataset to iteration_i/dataset_forming/intermediate_dataset_iter_{i}_tmp.json."""
        dst_dir = self.get_iter_dataset_forming_dir(distillation_iter)
        path = os.path.join(dst_dir, f"intermediate_dataset_iter_{distillation_iter}_tmp.json")
        data = {
            "distillation_iter": distillation_iter,
            "n_raw": n_raw,
            "n_final": len(indices),
            "batch_idx": batch_idx,
            "indices": indices,
            "lengths_chars": lengths_chars,
            "lengths_tokens": lengths_tokens,
        }
        if audio_ids is not None:
            data["audio_ids"] = audio_ids
        _atomic_write_json(path, data)
        return path

    def save_presampling_verdicts(
        self,
        distillation_iter: int,
        batch_idx: int,
        verdicts: List[Dict[str, Any]],
        rank: int = 0,
    ) -> str:
        """Save presampling verdicts to iteration_i/dataset_forming/presampling/presampling_batch_{j}_rank_{k}.json."""
        dst_dir = self.get_iter_dataset_forming_dir(distillation_iter)
        presampling_dir = os.path.join(dst_dir, "presampling")
        path = os.path.join(presampling_dir, f"presampling_batch_{batch_idx}_rank_{rank}.json")
        _atomic_write_json(path, {"verdicts": verdicts})
        return path

    def save_dataset_forming_progress(
        self,
        distillation_iter: int,
        phase: str,
        batch_idx: int,
        samples_processed: int,
        n_candidates: int = 0,
        n_accumulated: int = 0,
        attempt: int = 0,
    ):
        """Write progress.json every batch for dense logging. Uses atomic write to avoid races."""
        dst_dir = self.get_iter_dataset_forming_dir(distillation_iter)
        path = os.path.join(dst_dir, "progress.json")
        data = {
            "distillation_iter": distillation_iter,
            "phase": phase,
            "batch_idx": batch_idx,
            "samples_processed": samples_processed,
            "n_candidates": n_candidates,
            "n_accumulated": n_accumulated,
            "attempt": attempt,
        }
        _atomic_write_json(path, data)

    def log_dataset_forming_stats(
        self,
        distillation_iter: int,
        n_raw: int,
        n_after_skeptic: Optional[int] = None,
        n_after_correctness: Optional[int] = None,
        n_after_length: Optional[int] = None,
        n_final: int = 0,
        n_rollouts_skeptic: Optional[int] = None,
        n_rollouts_main: Optional[int] = None,
        text_length_chars: Optional[List[float]] = None,
        text_length_tokens: Optional[List[float]] = None,
        lengths_chars: Optional[List[int]] = None,
        lengths_tokens: Optional[List[int]] = None,
        **kwargs
    ):
        """Log dataset forming stats for plots and metrics."""
        data = {
            "type": "distillation_dataset_forming",
            "distillation_iter": distillation_iter,
            "n_raw": n_raw,
            "n_final": n_final,
            **kwargs,
        }
        if lengths_chars is not None:
            data["lengths_chars"] = lengths_chars
        if lengths_tokens is not None:
            data["lengths_tokens"] = lengths_tokens
        if n_after_skeptic is not None:
            data["n_after_skeptic"] = n_after_skeptic
        if n_after_correctness is not None:
            data["n_after_correctness"] = n_after_correctness
        if n_after_length is not None:
            data["n_after_length"] = n_after_length
        if n_rollouts_skeptic is not None:
            data["n_rollouts_skeptic"] = n_rollouts_skeptic
        if n_rollouts_main is not None:
            data["n_rollouts_main"] = n_rollouts_main
        if text_length_chars:
            data["text_length_chars_mean"] = sum(text_length_chars) / len(text_length_chars)
            data["text_length_chars_median"] = sorted(text_length_chars)[len(text_length_chars) // 2]
        if text_length_tokens:
            data["text_length_tokens_mean"] = sum(text_length_tokens) / len(text_length_tokens)
            data["text_length_tokens_median"] = sorted(text_length_tokens)[len(text_length_tokens) // 2]

        self._save_json(data, "metrics.jsonl")

        # Also save per-iteration stats (for next-iter relative filter + plots)
        dst_dir = self.get_iter_dataset_forming_dir(distillation_iter)
        stats_path = os.path.join(dst_dir, f"form_stats_iter_{distillation_iter}.json")
        _atomic_write_json(stats_path, data)

    def log_text_lengths(self, distillation_iter: int, lengths_chars: List[int], lengths_tokens: List[int]):
        """Save text lengths for next iteration's relative filtering."""
        dst_dir = self.get_iter_dataset_forming_dir(distillation_iter)
        path = os.path.join(dst_dir, f"form_stats_iter_{distillation_iter}.json")
        data = {"lengths_chars": lengths_chars, "lengths_tokens": lengths_tokens}
        _atomic_write_json(path, data, indent=None)

    def _save_json(self, data: Dict[str, Any], filename: str):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, filename)
        with open(path, "a") as f:
            f.write(json.dumps(data) + "\n")
            f.flush()
        # Also append to global metrics.jsonl for distillation-level plots (SFT/GRPO accuracy, etc.)
        if filename == "metrics.jsonl" and self.distillation_iter is not None:
            global_data = dict(data)
            global_data["distillation_iter"] = self.distillation_iter
            global_path = os.path.join(self.base_log_dir, "metrics.jsonl")
            os.makedirs(self.base_log_dir, exist_ok=True)
            with open(global_path, "a") as f:
                f.write(json.dumps(global_data) + "\n")
                f.flush()
