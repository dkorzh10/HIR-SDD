import logging
import random
import re

import torch
import torch.distributed as dist

from src.infrastructure.dist_utils import is_dist_avail_and_initialized, get_rank
from src.infrastructure.logger import MetricLogger
from src.infrastructure.utils import prepare_sample
from src.util_classes.response_parser import ResponseParser

class Validator:
    """Validation logic for SALMON GRPO"""
    
    def __init__(self, config, model_utils, file_utils, output_dir, test_prompt_dict=None):
        self.config = config
        self.model_utils = model_utils
        self.file_utils = file_utils
        self.output_dir = output_dir
        self.test_prompt_dict = test_prompt_dict
        
        self.cuda_enabled = (config.config.run.device == "cuda")
        self.use_amp = config.config.run.amp
        
    @torch.no_grad()
    def valid_epoch(self, epoch, split, model, dataloader, decode=False, save_json=False):
        model = self.model_utils.unwrap_dist_model(model)
        model.eval()

        assert dataloader is not None, "{}_loader does not exist.".format(split)

        metric_logger = MetricLogger(delimiter="  ")
        header = "Eval: data epoch: [{}]".format(epoch)

        results = []
        for samples in metric_logger.log_every(dataloader, self.config.config.run.log_freq, header=header):
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                forward_result = model(samples, verbose=True)
            loss = forward_result.get("loss", 0)
            correct = forward_result.get("correct", 0)
            total = forward_result.get("total", 1)
            res = {
                "id": samples["id"],
                "ground_truth": samples["text"],
                "loss": loss.item(),
                "acc": (correct / total).item(),
                "total": total,
            }

            if decode:
                if model.prompt_dict:
                    if self.test_prompt_dict is None:
                        prompts = None
                    else:
                        # If test_prompt_dict values are lists, randomly select one prompt per task
                        prompts = []
                        for s in samples["task"]:
                            prompt_val = self.test_prompt_dict[s]
                            if isinstance(prompt_val, list):
                                prompts.append(random.choice(prompt_val))
                            else:
                                prompts.append(prompt_val)
                        if "Q" in samples:
                            prompts = [p.format(q) if "{}" in p else p for p, q in zip(prompts, samples["Q"])]
                else:
                    prompts = None

                text = model.generate(samples, self.config.config.run, prompts=prompts)
                res["text"] = text
                res["prompt"] = prompts
                res["task"] = samples["task"]

            results.append(res)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if save_json:
            self.file_utils.save_result(results, self.output_dir, "eval_{}_epoch_{}".format(split, epoch))

        res = {
            "loss": torch.tensor(0).float().cuda(),
            "n_sample": torch.tensor(0).float().cuda(),
            "correct": torch.tensor(0).float().cuda(),
            "n_token": torch.tensor(0).float().cuda(),
            "classification_correct": torch.tensor(0).float().cuda(),  # New: sample-level accuracy
            "classification_total": torch.tensor(0).float().cuda(),
        }
        
        # Accumulate reasons for metrics (local process)
        all_pred_reasons = []
        all_gt_reasons = []

        for item in results:
            item_loss = item["loss"]
            item_n_sample = len(item["id"])
            item_correct = item["acc"] * item["total"]
            item_n_token = item["total"]
            res["loss"] += item_loss * item_n_sample
            res["n_sample"] += item_n_sample
            res["correct"] += item_correct
            res["n_token"] += item_n_token
            
            # Compute classification accuracy if we have generated text
            if decode and "text" in item:
                for i, (pred_text, gt_text) in enumerate(zip(item["text"], item["ground_truth"])):
                    pred_label = ResponseParser.parse_prediction(pred_text)
                    gt_label = ResponseParser.parse_prediction(gt_text)
                    
                    if pred_label is not None and gt_label is not None:
                        res["classification_total"] += 1
                        if pred_label == gt_label:
                            res["classification_correct"] += 1

                    # Parse reasons for balanced accuracy
                    pr = ResponseParser.parse_reasons(pred_text)
                    gr = ResponseParser.parse_reasons(gt_text)
                    all_pred_reasons.append(set(pr))
                    all_gt_reasons.append(set(gr))

        if is_dist_avail_and_initialized():
            dist.all_reduce(res["loss"])
            dist.all_reduce(res["n_sample"])
            dist.all_reduce(res["correct"])
            dist.all_reduce(res["n_token"])
            dist.all_reduce(res["classification_correct"])
            dist.all_reduce(res["classification_total"])

        ret = {"loss": 0, "agg_metrics": 0}
        ret["loss"] = (res["loss"] / res["n_sample"]).item()
        
        # Use classification accuracy if available, otherwise token accuracy
        if decode and res["classification_total"] > 0:
            ret["agg_metrics"] = (res["classification_correct"] / res["classification_total"]).item()
            ret["token_acc"] = (res["correct"] / res["n_token"]).item()
            ret["classification_acc"] = ret["agg_metrics"]
            logging.info(f"Classification Accuracy: {ret['classification_acc']:.4f} ({res['classification_correct']:.0f}/{res['classification_total']:.0f})")
            logging.info(f"Token Accuracy: {ret['token_acc']:.4f}")
            
            # Compute and log balanced reasoning accuracy
            if all_gt_reasons:
                reason_metrics = self._compute_balanced_metrics(all_pred_reasons, all_gt_reasons)
                if reason_metrics:
                    ret.update(reason_metrics)
                    logging.info(f"Reasoning Acc (Balanced): {ret['reason_acc']:.4f}")
                    
        else:
            ret["agg_metrics"] = (res["correct"] / res["n_token"]).item()

        return ret

    def _compute_balanced_metrics(self, pred_reasons, gt_reasons):
        """Compute balanced accuracy for reasoning labels"""
        all_labels = set()
        for r in gt_reasons:
            all_labels.update(r)
        
        if not all_labels:
            return {}
            
        class_accuracies = {}
        for label in all_labels:
            tp = tn = fp = fn = 0
            
            for p_set, g_set in zip(pred_reasons, gt_reasons):
                pred_present = label in p_set
                gt_present = label in g_set
                
                if pred_present and gt_present:
                    tp += 1
                elif not pred_present and not gt_present:
                    tn += 1
                elif pred_present and not gt_present:
                    fp += 1
                elif not pred_present and gt_present:
                    fn += 1
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            if (tp + fn) == 0:
                balanced_acc = specificity
            elif (tn + fp) == 0:
                balanced_acc = sensitivity
            else:
                balanced_acc = 0.5 * (sensitivity + specificity)
                
            class_accuracies[label] = balanced_acc
        
        avg_reason_acc = sum(class_accuracies.values()) / len(class_accuracies)
        
        return {
            "reason_acc": avg_reason_acc,
            "reason_per_class": class_accuracies
        }
