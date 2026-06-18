from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, List, Union
import torch

class ModelType(Enum):
    SALMON = "salmon"
    SPEECHGPT = "speechgpt"
    FLAMINGO = "flamingo"
    QWEN_AUDIO = "qwen_audio"
    CONV_AUDIO_CLASSIFIER = "conv_audio_classifier"
    DUMMY = "dummy"

class Model(ABC, torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_name = config.get("model_name")
        # Load checkpoint if provided (logic to be implemented in subclasses or here)
        self.ckpt = config.get("ckpt", "")
        
    @abstractmethod
    def forward(self, samples: Dict[str, Any], verbose: bool = False) -> Dict[str, torch.Tensor]:
        """
        Returns {'loss': tensor}; used for SFT/Likelihood training
        """
        pass

    @abstractmethod
    def generate(self, samples: Dict[str, Any], generate_cfg: Dict[str, Any], prompts: Optional[List[str]] = None, return_outputs: bool = False) -> Union[List[str], Any]:
        """
        Returns texts or (texts, completion_ids, logits) if return_outputs=True
        """
        pass

    @abstractmethod
    def compute_logits_for_completions(self, samples: Dict[str, Any], completion_ids: torch.Tensor) -> torch.Tensor:
        """
        Recomputes logits for given completion_ids with gradients; critical for GRPO
        """
        pass

    @abstractmethod
    def generate_text_only(self, prompt_texts: List[str], generate_cfg: Dict[str, Any]) -> List[str]:
        """
        Generates text from text-only prompts; used for unbiased judge evaluations
        """
        pass
    
    def get_tokenizer(self):
        """
        Return tokenizer for length/counting. None if not available (e.g. DummyModel).
        Used by distillation for token-based length filtering.
        """
        return None

    def get_pad_token_id(self) -> Optional[int]:
        """
        Return pad_token_id from tokenizer if available. Used for correct padding across models.
        """
        tok = self.get_tokenizer()
        if tok is None:
            return None
        pid = getattr(tok, "pad_token_id", None)
        if pid is not None and pid >= 0:
            return int(pid)
        eid = getattr(tok, "eos_token_id", None)
        if eid is not None and eid >= 0:
            return int(eid)
        return None

    def count_tokens(self, text: str) -> int:
        """
        Token count for text. Uses tokenizer if available, else fallback to chars//4.
        """
        tok = self.get_tokenizer()
        if tok is not None:
            try:
                ids = tok.encode(text, add_special_tokens=False)
            except Exception:
                ids = getattr(tok(text, add_special_tokens=False), "input_ids", [])
            if isinstance(ids, (list, tuple)):
                return len(ids)
            if hasattr(ids, "shape"):
                return ids.shape[0] if ids.ndim > 0 else 0
        return max(0, len(text) // 4)  # fallback for DummyModel

    def swap_weights(self):
        """
        Swaps trainable weights with reference snapshot. All tensors stay on GPU.
        First call: snapshots current trainable params as ref (on same device).
        Subsequent calls: in-place swap between current and ref.
        """
        trainable = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if not trainable:
            return

        ref = getattr(self, "_ref_trainable_state", None)
        if ref is None:
            self._ref_trainable_state = {n: p.detach().clone() for n, p in trainable.items()}
            return

        for name, param in trainable.items():
            if name not in ref:
                continue
            ref_t = ref[name]
            tmp = param.detach().clone()
            param.data.copy_(ref_t)
            ref_t.copy_(tmp)
