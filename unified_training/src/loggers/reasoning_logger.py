from typing import Any, Optional, List, Dict, Set
import numpy as np
import os
import json
import re
import torch.distributed as dist
from .base import Logger

# All possible reason categories for antispoofing
ALL_REASONS = [
    'UNNATURAL_PAUSES', 'MONOTONY_OF_STYLE', 'STRANGE_VOICE', 
    'STRANGE_INTONATION', 'INCORRECT_WORD_STRESS', 'STRANGE_ACCENT',
    'INCORRECT_ABBREVIATIONS', 'TOO_LONG_WITH_NO_PAUSE', 'OTHER', 'TOO_FAST'
]

class ReasoningLogger(Logger):
    def __init__(self, log_dir: str, log_freq: int, save_all_predictions: bool = False):
        super().__init__(log_dir, log_freq)
        self.reasons_log = []
        self.save_all_predictions = save_all_predictions
        self._predictions_file = None
        self._reset_incremental_stats()

    def set_epoch(self, epoch_num: int, epoch_type: str):
        super().set_epoch(epoch_num, epoch_type)
        self._reset_incremental_stats()
        
        # Open predictions file for streaming writes
        if self.save_all_predictions and epoch_type in ["validation", "test"]:
            os.makedirs(self.log_dir, exist_ok=True)
            # Add rank suffix for distributed training
            rank_suffix = ""
            if dist.is_initialized():
                rank = dist.get_rank()
                rank_suffix = f"_rank{rank}"
            pred_path = os.path.join(self.log_dir, f"predictions_{epoch_type}_epoch_{epoch_num}{rank_suffix}.jsonl")
            self._predictions_file = open(pred_path, "w")
        else:
            self._predictions_file = None

    def _reset_incremental_stats(self):
        """Reset incremental accuracy stats for new epoch."""
        self.correct_count = 0
        self.total_count = 0
        self.real_correct = 0
        self.real_total = 0
        self.fake_correct = 0
        self.fake_total = 0
        self.reason_stats = {r: {"tp": 0, "total": 0} for r in ALL_REASONS}
        self.sample_outputs = []  # Keep only a few samples for visualization
        self.sample_seen_count = 0
        
        # Confidence tracking
        self.all_confidences = []
        self.correct_confidences = []
        self.incorrect_confidences = []
        self.real_confidences = []
        self.fake_confidences = []
        
        # TP/FP/TN/FN confidence tracking
        self.tp_confidences = []  # True Positive: predicted Fake, actually Fake
        self.fp_confidences = []  # False Positive: predicted Fake, actually Real
        self.tn_confidences = []  # True Negative: predicted Real, actually Real
        self.fn_confidences = []  # False Negative: predicted Real, actually Fake

    def _parse_reasons(self, text: str) -> Set[str]:
        """Extract reasons set from <reasons>[...]</reasons> tag."""
        match = re.search(r'<reasons>\s*(\[.*?\])\s*</reasons>', str(text), re.DOTALL)
        if not match:
            return set()
        try:
            raw = match.group(1)
            # Handle nested list strings like ["['REASON1', 'REASON2']"]
            parsed = eval(raw)
            if isinstance(parsed, list):
                result = set()
                for item in parsed:
                    if isinstance(item, str):
                        # Could be "['REASON1', 'REASON2']" or just "REASON1"
                        if item.startswith('['):
                            inner = eval(item)
                            result.update(r.upper() for r in inner)
                        else:
                            result.add(item.upper())
                return result
        except:
            pass
        return set()

    def log(self, loss: Optional[float] = None, lr: Optional[float] = None, 
            outputs: Optional[Any] = None, gt: Optional[Any] = None, 
            output_len: Optional[int] = None, reasons: Optional[List[str]] = None, 
            correct: Optional[float] = None, total: Optional[float] = None,
            confidences: Optional[list] = None, audio_ids: Optional[list] = None, **kwargs):
        
        if loss is not None:
            self.losses.append(loss)
        if lr is not None:
            self.lrs.append(lr)
            
        if reasons:
            self.reasons_log.extend(reasons)
            
        if correct is not None and total is not None:
            if not hasattr(self, 'token_correct'): self.token_correct = 0
            if not hasattr(self, 'token_total'): self.token_total = 0
            self.token_correct += correct
            self.token_total += total
        
        # Accumulate accuracy stats incrementally (validation/test only)
        if outputs and gt and self.epoch_type in ["validation", "test"]:
            outputs_list = outputs if isinstance(outputs, list) else [outputs]
            gt_list = gt if isinstance(gt, list) else [gt]
            conf_list = confidences if confidences else [None] * len(outputs_list)
            audio_ids_list = (audio_ids if isinstance(audio_ids, list) else [audio_ids] if audio_ids is not None else [])
            audio_ids_list = (audio_ids_list + [None] * len(outputs_list))[:len(outputs_list)]
            
            for idx, (o, g) in enumerate(zip(outputs_list, gt_list)):
                conf = conf_list[idx] if idx < len(conf_list) else None
                aid = audio_ids_list[idx] if idx < len(audio_ids_list) else None
                
                # Keep a representative subset using reservoir sampling.
                sample = {"gt": g, "output": o}
                if aid is not None:
                    sample["audio_id"] = aid
                if conf:
                    sample.update(conf)
                self.sample_seen_count += 1
                if len(self.sample_outputs) < 20:
                    self.sample_outputs.append(sample)
                else:
                    replace_idx = np.random.randint(0, self.sample_seen_count)
                    if replace_idx < 20:
                        self.sample_outputs[replace_idx] = sample
                
                # Stream all predictions to file
                if self._predictions_file:
                    record = {"gt": g, "output": o}
                    if aid is not None:
                        record["audio_id"] = aid
                    if conf:
                        record.update(conf)
                    self._predictions_file.write(json.dumps(record) + "\n")
                
                # Extract and compare answers
                o_match = re.search(r"<answer>(.*?)</answer>", str(o), re.DOTALL)
                g_match = re.search(r"<answer>(.*?)</answer>", str(g), re.DOTALL)
                
                o_ans = o_match.group(1).strip().lower() if o_match else str(o).lower()
                g_ans = g_match.group(1).strip().lower() if g_match else str(g).lower()
                
                is_correct = (o_ans == g_ans)
                self.total_count += 1
                if is_correct:
                    self.correct_count += 1
                
                # Track confidences
                if conf:
                    self.all_confidences.append(conf["confidence"])
                    if is_correct:
                        self.correct_confidences.append(conf["confidence"])
                    else:
                        self.incorrect_confidences.append(conf["confidence"])
                    
                    if g_ans == "real":
                        self.real_confidences.append(conf["confidence"])
                    elif g_ans == "fake":
                        self.fake_confidences.append(conf["confidence"])
                    
                    # TP/FP/TN/FN tracking
                    if o_ans == "fake" and g_ans == "fake":
                        self.tp_confidences.append(conf["confidence"])  # True Positive
                    elif o_ans == "fake" and g_ans == "real":
                        self.fp_confidences.append(conf["confidence"])  # False Positive
                    elif o_ans == "real" and g_ans == "real":
                        self.tn_confidences.append(conf["confidence"])  # True Negative
                    elif o_ans == "real" and g_ans == "fake":
                        self.fn_confidences.append(conf["confidence"])  # False Negative
                
                # Per-class stats
                if g_ans == "real":
                    self.real_total += 1
                    if is_correct:
                        self.real_correct += 1
                elif g_ans == "fake":
                    self.fake_total += 1
                    if is_correct:
                        self.fake_correct += 1
                
                # Per-reason stats
                gt_reasons = self._parse_reasons(g)
                pred_reasons = self._parse_reasons(o)
                
                for reason in ALL_REASONS:
                    if reason in gt_reasons:
                        self.reason_stats[reason]["total"] += 1
                        if reason in pred_reasons:
                            self.reason_stats[reason]["tp"] += 1
            
        self.iteration_num += 1
        
        if self.epoch_type == "train" and self.iteration_num % self.log_freq == 0:
             avg_loss = float(np.mean(self.losses[-self.log_freq:])) if self.losses else 0.0
             print(f"Epoch {self.epoch_num} [{self.iteration_num}] ({self.epoch_type}): Loss={avg_loss:.4f} LR={lr}", flush=True)
             
             # Also save batch metrics to file immediately
             if self.log_freq < 1e8:
                 batch_data = {
                     "epoch": self.epoch_num,
                     "iteration": self.iteration_num,
                     "type": "train_batch",
                     "loss": avg_loss,
                     "lr": lr
                 }
                 for k in ("reward", "kl_div", "correct_reasons_overlap",
                           "skeptic_rejected_per_check_batch", "skeptic_accepted_per_check_batch"):
                     if k in kwargs:
                         batch_data[k] = kwargs[k]
                 self._save_json(batch_data, "metrics.jsonl")

    def log_epoch(self):
        # Convert to float and handle NaNs
        clean_losses = [l for l in self.losses if not np.isnan(l)]
        avg_loss = float(np.mean(clean_losses)) if clean_losses else 0.0

        metrics = {
            "epoch": self.epoch_num,
            "type": self.epoch_type,
            "loss": avg_loss
        }
        
        if self.epoch_type in ["validation", "test"]:
            # Use incrementally accumulated stats
            if self.total_count > 0:
                metrics["accuracy"] = self.correct_count / self.total_count
                self.last_accuracy = metrics["accuracy"]
                
                # Per-class accuracy
                if self.real_total > 0:
                    metrics["accuracy_real"] = self.real_correct / self.real_total
                if self.fake_total > 0:
                    metrics["accuracy_fake"] = self.fake_correct / self.fake_total
                if self.real_total > 0 and self.fake_total > 0:
                    metrics["accuracy_balanced"] = (metrics["accuracy_real"] + metrics["accuracy_fake"]) / 2
                    self.last_accuracy_balanced = metrics["accuracy_balanced"]

                # Per-reason recall
                for reason, stats in self.reason_stats.items():
                    if stats["total"] > 0:
                        metrics[f"accuracy_reason_{reason.lower()}"] = stats["tp"] / stats["total"]
                
                # Save sample outputs for visualization
                rank_suffix = ""
                if dist.is_initialized():
                    rank_suffix = f"_rank{dist.get_rank()}"
                os.makedirs(self.log_dir, exist_ok=True)
                samples_path = os.path.join(
                    self.log_dir,
                    f"samples_{self.epoch_type}_epoch_{self.epoch_num}{rank_suffix}.jsonl",
                )
                with open(samples_path, "w") as f:
                    for sample in self.sample_outputs:
                        f.write(json.dumps(sample) + "\n")
                    f.flush()

            # Token-level accuracy from forward
            if hasattr(self, 'token_correct') and self.token_total > 0:
                metrics["token_accuracy"] = float(self.token_correct / self.token_total)
                self.token_correct = 0
                self.token_total = 0
            
            # Confidence metrics
            if self.all_confidences:
                metrics["confidence_mean"] = float(np.mean(self.all_confidences))
                metrics["confidence_std"] = float(np.std(self.all_confidences))
                if self.correct_confidences:
                    metrics["confidence_correct_mean"] = float(np.mean(self.correct_confidences))
                if self.incorrect_confidences:
                    metrics["confidence_incorrect_mean"] = float(np.mean(self.incorrect_confidences))
                if self.real_confidences:
                    metrics["confidence_real_mean"] = float(np.mean(self.real_confidences))
                if self.fake_confidences:
                    metrics["confidence_fake_mean"] = float(np.mean(self.fake_confidences))
                
                # TP/FP/TN/FN confidence metrics
                if self.tp_confidences:
                    metrics["confidence_tp_mean"] = float(np.mean(self.tp_confidences))
                    metrics["confidence_tp_count"] = len(self.tp_confidences)
                if self.fp_confidences:
                    metrics["confidence_fp_mean"] = float(np.mean(self.fp_confidences))
                    metrics["confidence_fp_count"] = len(self.fp_confidences)
                if self.tn_confidences:
                    metrics["confidence_tn_mean"] = float(np.mean(self.tn_confidences))
                    metrics["confidence_tn_count"] = len(self.tn_confidences)
                if self.fn_confidences:
                    metrics["confidence_fn_mean"] = float(np.mean(self.fn_confidences))
                    metrics["confidence_fn_count"] = len(self.fn_confidences)

        print(f"End of Epoch {self.epoch_num} ({self.epoch_type}): {metrics}", flush=True)
        if self.log_freq < 1e8:
            self._save_json(metrics, "metrics.jsonl")
            
            if self.reasons_log:
                with open(os.path.join(self.log_dir, f"reasons_epoch_{self.epoch_num}.json"), "w") as f:
                    json.dump(self.reasons_log, f, indent=2)
                    f.flush()
        
        # Close predictions file if open
        if self._predictions_file:
            self._predictions_file.close()
            self._predictions_file = None
            print(f"All predictions saved to: {self.log_dir}/predictions_{self.epoch_type}_epoch_{self.epoch_num}.jsonl", flush=True)
        
        # Reset for next epoch
        self.reasons_log = []
        self._reset_incremental_stats()