from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import torch
from tqdm import tqdm
from ..models.base import Model
from ..loggers.base import Logger
from .base import Epoch
from .utils.confidence import extract_confidences, get_tokenizer, get_token_ids
from .utils.text_utils import texts_for_log


class EvalEpoch(Epoch):
    def __init__(self, model, dataloader, logger, device=None, gen_cfg: Optional[Dict[str, Any]] = None, extract_confidence: bool = True):
        super().__init__(model, dataloader, logger, device=device)
        # Default generation config, can be overridden via constructor
        self.gen_cfg = gen_cfg or {"max_new_tokens": 5000, "num_beams": 1, "do_sample": False}
        self.extract_confidence = extract_confidence
        
        # Cache token ID lists for Real/Fake (lists support multi-token words, e.g. SALMON "Fake" -> ["F", "ake"])
        self._real_token_ids = None
        self._fake_token_ids = None

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from either format."""
        import re
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

    def _extract_confidences(self, pred_texts: List[str], logits: torch.Tensor) -> List[Dict[str, float]]:
        """
        Extract Real/Fake probabilities from logits. For multi-token words (e.g. SALMON 'Fake' -> F, ake), uses product of token probs.
        
        Args:
            pred_texts: List of decoded prediction texts
            logits: Logits tensor [batch_size, seq_len, vocab_size]
        
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
            token_ids=None,  # eval_epoch doesn't pass token_ids
            extract_answer_fn=self._extract_answer
        )

    def run(self, **kwargs):
        self.logger.set_epoch(kwargs.get("epoch_num", 0), "validation")
        self.model.eval()
        
        # If using distributed sampler, set epoch
        if hasattr(self.dataloader, "sampler") and hasattr(self.dataloader.sampler, "set_epoch"):
            self.dataloader.sampler.set_epoch(kwargs.get("epoch_num", 0))

        # Determine autocast dtype - use bfloat16 if supported, else float16
        autocast_dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        
        with torch.no_grad():
            # Use same autocast as training with correct dtype
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=self.device.type == "cuda"):
                for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Eval"):
                    batch = self._move_to_device(batch, self.device)
                    
                    # Call forward with verbose=True to get accuracy-related metrics
                    outputs = self.model(batch, verbose=True)
                    loss_t = outputs["loss"]
                    loss = loss_t.mean().item() if loss_t.numel() > 1 else loss_t.item()
                    
                    gt_texts = batch.get("text", [])
                    audio_ids = batch.get("audio_ids", [])

                    # Generate with logits for confidence extraction
                    if self.extract_confidence:
                        pred_texts, completion_ids, logits = self.unwrapped_model.generate(
                            batch, self.gen_cfg, return_outputs=True
                        )
                        confidences = self._extract_confidences(pred_texts, logits)
                    else:
                        pred_texts = self.unwrapped_model.generate(batch, self.gen_cfg)
                        confidences = None

                    # print(f"pred_texts: {pred_texts}, gt_texts: {gt_texts}")
                    self.logger.log(
                        loss=loss, 
                        outputs=texts_for_log(self.unwrapped_model, pred_texts), 
                        gt=gt_texts,
                        correct=outputs.get("correct"),
                        total=outputs.get("total"),
                        confidences=confidences,
                        audio_ids=audio_ids
                    )
        
        self.logger.log_epoch()

    def _move_to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._move_to_device(v, device) for v in batch]
        return batch



