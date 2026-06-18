from typing import Any, Dict, Optional, List
import re
import torch
from tqdm import tqdm
from .base import Epoch
from ..loggers.base import Logger
from .utils.confidence import extract_confidences, get_tokenizer, get_token_ids
from .utils.text_utils import texts_for_log


class TestEpoch(Epoch):
    def __init__(self, model, dataloader, logger: Logger, device=None, 
                 gen_cfg: Optional[Dict[str, Any]] = None,
                 model_format: str = "hard_label",
                 dataset_format: str = "hard_label",
                 log_freq: int = 10,
                 extract_confidence: bool = True):
        super().__init__(model, dataloader, logger, device=device)
        self.gen_cfg = gen_cfg or {"max_new_tokens": 5000, "num_beams": 1, "do_sample": False}
        self.model_format = model_format
        self.dataset_format = dataset_format
        self.log_freq = log_freq
        self.extract_confidence = extract_confidence
        
        # Running stats for intermediate logging
        self.running_correct = 0
        self.running_total = 0
        self.running_parse_errors = 0
        
        # Cache token ID lists for Real/Fake (lists support multi-token words, e.g. SALMON "Fake" -> ["F", "ake"])
        self._real_token_ids = None
        self._fake_token_ids = None

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from either format."""
        # Try reasoning format: <answer>Real/Fake</answer>
        match = re.search(r"<answer>(.*?)</answer>", str(text), re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try hard-label format: Final Answer: Real/Fake
        match = re.search(r"Final\s*Answer:\s*(Real|Fake)", str(text), re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback: look for Real/Fake anywhere
        text_lower = str(text).lower()
        if "fake" in text_lower:
            return "Fake"
        if "real" in text_lower:
            return "Real"
        
        return str(text).strip()

    def _extract_answer_from_gt(self, text: str) -> str:
        """Extract ground truth answer."""
        text_clean = str(text).strip().lower()
        if text_clean in ["real", "fake"]:
            return text_clean.capitalize()
        
        match = re.search(r"<answer>(.*?)</answer>", str(text), re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return str(text).strip()

    def _extract_confidences(self, pred_texts: List[str], logits: torch.Tensor, token_ids: torch.Tensor = None) -> List[Dict[str, float]]:
        """
        Extract Real/Fake probabilities from logits. For multi-token words (e.g. SALMON 'Fake' -> F, ake), uses product of token probs.
        
        Args:
            pred_texts: List of decoded prediction texts
            logits: Logits tensor [batch_size, seq_len, vocab_size]
            token_ids: Optional token IDs tensor [batch_size, seq_len] for more reliable position finding
        
        Returns:
            List of dicts with real_prob, fake_prob, and confidence (where confidence = P(predicted class) for EER)
        """
        tokenizer = get_tokenizer(self.unwrapped_model)
        if tokenizer is None or logits is None:
            return [{"real_prob": 0.0, "fake_prob": 0.0, "confidence": 0.5} for _ in pred_texts]
        
        # Use instance cache for token IDs
        cache = {'_real_token_ids': self._real_token_ids, '_fake_token_ids': self._fake_token_ids}
        real_ids, fake_ids = get_token_ids(tokenizer, cache)
        self._real_token_ids = cache['_real_token_ids']
        self._fake_token_ids = cache['_fake_token_ids']
        
        return extract_confidences(
            pred_texts=pred_texts,
            logits=logits,
            tokenizer=tokenizer,
            real_ids=real_ids,
            fake_ids=fake_ids,
            token_ids=token_ids,
            extract_answer_fn=self._extract_answer
        )

    def run(self, **kwargs):
        self.logger.set_epoch(kwargs.get("epoch_num", 0), "test")
        self.model.eval()
        
        if hasattr(self.dataloader, "sampler") and hasattr(self.dataloader.sampler, "set_epoch"):
            self.dataloader.sampler.set_epoch(kwargs.get("epoch_num", 0))

        
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Test"):
                batch = self._move_to_device(batch, self.device)
                
                outputs = self.model(batch, verbose=True)
                loss_t = outputs["loss"]
                loss = loss_t.mean().item() if loss_t.numel() > 1 else loss_t.item()
                
                # Generate with logits for confidence extraction
                # if self.extract_confidence:
                # Always extract confidences
                pred_texts, completion_ids, logits = self.unwrapped_model.generate(
                    batch, self.gen_cfg, return_outputs=True
                )
                # Per-sample aggregation for conv_audio_classifier when batch is windowed by test dataloader
                if isinstance(pred_texts, dict) and "window_probs" in pred_texts and "windows_per_sample" in pred_texts:
                    probs = pred_texts["window_probs"]
                    windows_per_sample = pred_texts["windows_per_sample"]
                    pred_texts = []
                    offset = 0
                    for n_w in windows_per_sample:
                        mean_prob = probs[offset : offset + n_w].mean().item()
                        pred_texts.append(
                            "Final Answer: Real" if mean_prob >= 0.5 else "Final Answer: Fake"
                        )
                        offset += n_w
                confidences = self._extract_confidences(pred_texts, logits, completion_ids)
                # else:
                #     pred_texts = self.unwrapped_model.generate(batch, self.gen_cfg)
                #     confidences = None
                
                gt_texts = batch.get("text", [])
                audio_ids = batch.get("audio_ids", [])
                
                # Handle format conversion for reasoning model on hard_label dataset
                if self.model_format == "reasoning" and self.dataset_format == "hard_label":
                    pred_answers = [self._extract_answer(p) for p in pred_texts]
                    gt_answers = [self._extract_answer_from_gt(g) for g in gt_texts]
                    self.logger.log(
                        loss=loss,
                        outputs=pred_answers,
                        gt=gt_answers,
                        correct=outputs.get("correct"),
                        total=outputs.get("total"),
                        confidences=confidences,
                        audio_ids=audio_ids
                    )
                    # Update running stats
                    for p, g in zip(pred_answers, gt_answers):
                        if p.lower() not in ("real", "fake"):
                            self.running_parse_errors += 1
                        if p.lower() == g.lower():
                            self.running_correct += 1
                        self.running_total += 1
                else:
                    self.logger.log(
                        loss=loss,
                        outputs=texts_for_log(self.unwrapped_model, pred_texts),
                        gt=gt_texts,
                        correct=outputs.get("correct"),
                        total=outputs.get("total"),
                        confidences=confidences,
                        audio_ids=audio_ids
                    )
                    # Update running stats (extract answers for comparison)
                    for p, g in zip(pred_texts, gt_texts):
                        p_ans = self._extract_answer(p).lower()
                        if p_ans not in ("real", "fake"):
                            self.running_parse_errors += 1
                        g_ans = self._extract_answer_from_gt(g).lower()
                        if p_ans == g_ans:
                            self.running_correct += 1
                        self.running_total += 1
                
                # Print intermediate results
                if (i + 1) % self.log_freq == 0:
                    running_acc = self.running_correct / self.running_total if self.running_total > 0 else 0
                    print(f"  [{i+1}/{len(self.dataloader)}] Running Accuracy: {running_acc:.4f} ({self.running_correct}/{self.running_total})", flush=True)

        # Parsing error metrics (predictions that did not yield valid Real/Fake)
        parse_error_rate = self.running_parse_errors / self.running_total if self.running_total > 0 else 0.0
        print(f"  Test epoch parsing: total={self.running_total}, parse_errors={self.running_parse_errors}, parse_error_rate={parse_error_rate:.4f}", flush=True)

        self.logger.log_epoch()

    def _move_to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._move_to_device(v, device) for v in batch]
        return batch
