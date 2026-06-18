from typing import Any, Optional
import os
import json
import numpy as np
import torch.distributed as dist
from .base import Logger

class HardLabelLogger(Logger):
    def __init__(self, log_dir: str, log_freq: int, save_all_predictions: bool = False):
        super().__init__(log_dir, log_freq)
        self.save_all_predictions = save_all_predictions
        self._predictions_file = None
        self.audio_ids = []
        self._reset_confidence_stats()

    def reset_epoch(self):
        super().reset_epoch()
        self.audio_ids = []

    def _reset_confidence_stats(self):
        """Reset confidence tracking for new epoch."""
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

    def set_epoch(self, epoch_num: int, epoch_type: str):
        super().set_epoch(epoch_num, epoch_type)
        self._reset_confidence_stats()
        
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

    def log(self, loss: Optional[float] = None, lr: Optional[float] = None, 
            outputs: Optional[Any] = None, gt: Optional[Any] = None, 
            correct: Optional[float] = None, total: Optional[float] = None,
            confidences: Optional[list] = None, audio_ids: Optional[list] = None, **kwargs):
        
        if loss is not None:
            self.losses.append(loss)
        if lr is not None:
            self.lrs.append(lr)
        if outputs is not None:
            self.outputs.extend(outputs if isinstance(outputs, list) else [outputs])
        if gt is not None:
            self.gts.extend(gt if isinstance(gt, list) else [gt])
        if audio_ids is not None:
            self.audio_ids.extend(audio_ids if isinstance(audio_ids, list) else [audio_ids])
        
        # Track confidences
        if confidences and outputs and gt and self.epoch_type in ["validation", "test"]:
            outputs_list = outputs if isinstance(outputs, list) else [outputs]
            gt_list = gt if isinstance(gt, list) else [gt]
            for o, g, conf in zip(outputs_list, gt_list, confidences):
                self.all_confidences.append(conf["confidence"])
                # Extract answers for comparison
                o_ans = self._extract_answer(o).lower()
                g_ans = self._extract_answer(g).lower()
                is_correct = (o_ans == g_ans)
                
                if is_correct:
                    self.correct_confidences.append(conf["confidence"])
                else:
                    self.incorrect_confidences.append(conf["confidence"])
                
                # Per-class confidence
                if g_ans == "real":
                    self.real_confidences.append(conf["confidence"])
                elif g_ans == "fake":
                    self.fake_confidences.append(conf["confidence"])
                
                # TP/FP/TN/FN tracking
                # Note: In antispoofing, "Fake" is the positive class (what we want to detect)
                if o_ans == "fake" and g_ans == "fake":
                    self.tp_confidences.append(conf["confidence"])  # True Positive
                elif o_ans == "fake" and g_ans == "real":
                    self.fp_confidences.append(conf["confidence"])  # False Positive
                elif o_ans == "real" and g_ans == "real":
                    self.tn_confidences.append(conf["confidence"])  # True Negative
                elif o_ans == "real" and g_ans == "fake":
                    self.fn_confidences.append(conf["confidence"])  # False Negative
        
        # Stream predictions to file
        if self._predictions_file and outputs is not None and gt is not None:
            outputs_list = outputs if isinstance(outputs, list) else [outputs]
            gt_list = gt if isinstance(gt, list) else [gt]
            conf_list = confidences if confidences else [None] * len(outputs_list)
            audio_ids_list = (audio_ids if isinstance(audio_ids, list) else [audio_ids] if audio_ids is not None else [])
            audio_ids_list = (audio_ids_list + [None] * len(outputs_list))[:len(outputs_list)]
            for o, g, conf, aid in zip(outputs_list, gt_list, conf_list, audio_ids_list):
                record = {"gt": g, "output": o}
                if aid is not None:
                    record["audio_id"] = aid
                if conf:
                    record.update(conf)
                self._predictions_file.write(json.dumps(record) + "\n")
            
        if correct is not None and total is not None:
            if not hasattr(self, 'token_correct'): self.token_correct = 0
            if not hasattr(self, 'token_total'): self.token_total = 0
            self.token_correct += correct
            self.token_total += total
            
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
                self._save_json(batch_data, "metrics.jsonl")

    def _extract_answer(self, text: str) -> str:
        """Extract answer from hard label format."""
        import re
        text_str = str(text).strip()
        
        # Try to extract from "Final answer: Real/Fake" format
        match = re.search(r"Final\s*[Aa]nswer:\s*(Real|Fake)", text_str, re.IGNORECASE)
        if match:
            return match.group(1).strip().capitalize()
        
        # Fallback: look for Real/Fake anywhere in the text
        text_lower = text_str.lower()
        if "fake" in text_lower:
            return "Fake"
        if "real" in text_lower:
            return "Real"
        
        # If nothing found, return as-is
        return text_str

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
            if self.outputs and self.gts:
                # Extract answers before comparison
                extracted_outputs = [self._extract_answer(o) for o in self.outputs]
                extracted_gts = [self._extract_answer(g) for g in self.gts]
                
                # Text-based accuracy (from generate or extracted tags)
                correct_list = [o.lower() == g.lower() for o, g in zip(extracted_outputs, extracted_gts)]
                acc = sum(correct_list) / len(correct_list)
                metrics["accuracy"] = acc
                self.last_accuracy = acc
                
                # Per-class accuracy
                # Only calculate for samples where we have predictions
                real_indices = [i for i, g in enumerate(extracted_gts[:len(correct_list)]) if g.lower() == "real"]
                fake_indices = [i for i, g in enumerate(extracted_gts[:len(correct_list)]) if g.lower() == "fake"]
                
                if real_indices:
                    real_acc = sum([correct_list[i] for i in real_indices]) / len(real_indices)
                    metrics["accuracy_real"] = real_acc
                
                if fake_indices:
                    fake_acc = sum([correct_list[i] for i in fake_indices]) / len(fake_indices)
                    metrics["accuracy_fake"] = fake_acc
                    
                if real_indices and fake_indices:
                    metrics["accuracy_balanced"] = (real_acc + fake_acc) / 2
                    self.last_accuracy_balanced = metrics["accuracy_balanced"]
            
            # Token-level accuracy from forward (if text-based not possible for all)
            if hasattr(self, 'token_correct') and self.token_total > 0:
                metrics["token_accuracy"] = float(self.token_correct / self.token_total)
                # Reset for next epoch
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
                
            # Save some samples
            if self.outputs and self.gts:
                os.makedirs(self.log_dir, exist_ok=True)
                rank_suffix = ""
                if dist.is_initialized():
                    rank_suffix = f"_rank{dist.get_rank()}"
                os.makedirs(self.log_dir, exist_ok=True)
                samples_path = os.path.join(
                    self.log_dir,
                    f"samples_{self.epoch_type}_epoch_{self.epoch_num}{rank_suffix}.jsonl",
                )
                with open(samples_path, "w") as f:
                    for i in range(min(10, len(self.outputs))):
                        sample = {"gt": self.gts[i], "output": self.outputs[i]}
                        if i < len(self.audio_ids) and self.audio_ids[i] is not None:
                            sample["audio_id"] = self.audio_ids[i]
                        f.write(json.dumps(sample) + "\n")
        
        # Close predictions file if open
        if self._predictions_file:
            self._predictions_file.close()
            self._predictions_file = None
            print(f"All predictions saved to: {self.log_dir}/predictions_{self.epoch_type}_epoch_{self.epoch_num}.jsonl", flush=True)
                
        print(f"End of Epoch {self.epoch_num} ({self.epoch_type}): {metrics}", flush=True)
        if self.log_freq < 1e8: # Only Rank 0 usually has standard log_freq
            self._save_json(metrics, "metrics.jsonl")
