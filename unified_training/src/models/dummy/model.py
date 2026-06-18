from typing import Any, Dict, List, Optional, Union
import torch
from ...models.base import Model
from .tokenizer import DummyTokenizer


class DummyModel(Model):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Dummy parameter to make it a valid torch module with gradients
        self.dummy_param = torch.nn.Parameter(torch.randn(1))
        self._dummy_tokenizer = DummyTokenizer()

    def get_tokenizer(self):
        return self._dummy_tokenizer

    def forward(self, samples: Dict[str, Any], verbose: bool = False) -> Dict[str, torch.Tensor]:
        # Return a dummy loss
        res = {"loss": self.dummy_param * 0.0 + 1.0}
        if verbose:
            res["correct"] = torch.tensor(1.0)
            res["total"] = torch.tensor(1.0)
        return res

    def generate(self, samples: Dict[str, Any], generate_cfg: Dict[str, Any], prompts: Optional[List[str]] = None, return_outputs: bool = False) -> Union[List[str], Any]:
        batch_size = len(samples.get("audio_paths", [])) if "audio_paths" in samples else 1
        dummy_texts = ["Dummy generation"] * batch_size
        
        if return_outputs:
            dummy_completion_ids = torch.randint(0, 100, (batch_size, 10))
            dummy_logits = torch.randn(batch_size, 10, 100)
            return dummy_texts, dummy_completion_ids, dummy_logits
        return dummy_texts

    def compute_logits_for_completions(self, samples: Dict[str, Any], completion_ids: torch.Tensor) -> torch.Tensor:
        # Return dummy logits matching the shape of completion_ids
        batch_size, seq_len = completion_ids.shape
        vocab_size = 100 # Dummy vocab size
        return torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

    def generate_text_only(self, prompt_texts: List[str], generate_cfg: Dict[str, Any]) -> List[str]:
        return ["Dummy text only generation"] * len(prompt_texts)


