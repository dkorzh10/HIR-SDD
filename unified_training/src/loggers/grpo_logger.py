from typing import Any, List, Dict, Optional
import os
import json
import torch.distributed as dist
from .reasoning_logger import ReasoningLogger

class GRPOLogger(ReasoningLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewards = []
        self.kl_divs = []
        self.skeptic_epoch_rejected: Optional[int] = None
        self.skeptic_epoch_accepted: Optional[int] = None

    def set_skeptic_epoch_stats(self, rejected: int, accepted: int):
        self.skeptic_epoch_rejected = rejected
        self.skeptic_epoch_accepted = accepted
        
    def log_skeptic_check_scores(
        self,
        check_logs: List[Dict[str, Any]],
        check_iter: int,
    ):
        """Log skeptic check phase: audio_id, rollouts 1-3, has_passed. Writes to judge_logs/skeptic/."""
        skeptic_dir = os.path.join(self.log_dir, "judge_logs", "skeptic")
        os.makedirs(skeptic_dir, exist_ok=True)
        rank_suffix = ""
        if dist.is_initialized():
            rank_suffix = f"_rank{dist.get_rank()}"
        filename = f"skeptic_epoch_{self.epoch_num}_iter_{check_iter}{rank_suffix}.json"
        filepath = os.path.join(skeptic_dir, filename)
        with open(filepath, "w") as f:
            json.dump(check_logs, f, indent=2)
            f.flush()

    def log_judge_scores(
        self,
        logs: List[Dict[str, Any]],
        grpo_iter: Optional[int] = None,
    ):
        """Log GRPO batch judge scores. Uses epoch_E_iter_N_rank_K.json (same format for skeptic and regular modes).
        Pass grpo_iter in skeptic mode; otherwise uses self.iteration_num."""
        judge_dir = os.path.join(self.log_dir, "judge_logs")
        os.makedirs(judge_dir, exist_ok=True)

        rank_suffix = ""
        if dist.is_initialized():
            rank_suffix = f"_rank{dist.get_rank()}"
        iter_num = grpo_iter if grpo_iter is not None else self.iteration_num
        filename = f"epoch_{self.epoch_num}_iter_{iter_num}{rank_suffix}.json"
        filepath = os.path.join(judge_dir, filename)

        # Form gt for log file: use reasoning if present, else "Real"/"Fake" from is_bonafide
        out_logs = []
        for entry in logs:
            gt = entry.get("gt")
            is_bonafide = entry.get("is_bonafide")
            if (gt is None or (isinstance(gt, str) and gt.strip() == "")) and is_bonafide is not None:
                gt_display = "Real" if is_bonafide else "Fake"
            else:
                gt_display = gt
            out_entry = {k: v for k, v in entry.items() if k != "is_bonafide"}
            out_entry["gt"] = gt_display
            out_logs.append(out_entry)

        with open(filepath, "w") as f:
            json.dump(out_logs, f, indent=2)
            f.flush()

    def log(self, loss=None, lr=None, reward=None, kl_div=None, **kwargs):
        super().log(loss=loss, lr=lr, **kwargs)
        if reward is not None:
            self.rewards.append(reward)
        if kl_div is not None:
            self.kl_divs.append(kl_div)

    def log_epoch(self):
        if self.skeptic_epoch_rejected is not None and self.skeptic_epoch_accepted is not None:
            metrics_path = os.path.join(self.log_dir, "metrics.jsonl")
            os.makedirs(self.log_dir, exist_ok=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "epoch": self.epoch_num,
                    "type": "skeptic_epoch",
                    "skeptic_epoch_rejected": self.skeptic_epoch_rejected,
                    "skeptic_epoch_accepted": self.skeptic_epoch_accepted,
                    "skeptic_rejected_accepted_ratio": (
                        self.skeptic_epoch_rejected / self.skeptic_epoch_accepted
                        if self.skeptic_epoch_accepted > 0 else float("inf")
                    ),
                }) + "\n")
                f.flush()
            self.skeptic_epoch_rejected = None
            self.skeptic_epoch_accepted = None
        super().log_epoch()





