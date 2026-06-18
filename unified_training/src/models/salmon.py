from typing import Dict, Any, List, Union, Optional
import torch
import copy
from .base import Model
from .SALMON.salmonn import SALMONN

class SalmonModel(Model):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Prepare config for SALMONN.from_config
        salmon_config = config.get("additional_kwargs", {}).get("salmon", {}).copy()
        
        # Flatten pretrained_ckpts if present
        if "pretrained_ckpts" in salmon_config:
            ckpts = salmon_config.pop("pretrained_ckpts")
            salmon_config.update(ckpts)
        
        # Merge other config params
        # Determine dtype based on hardware or config
        torch_dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            
        salmon_config.update({
            "lora": config.get("lora", {}).get("enabled", True),
            "lora_rank": config.get("lora", {}).get("rank", 8),
            "lora_alpha": config.get("lora", {}).get("alpha", 32),
            "lora_dropout": config.get("lora", {}).get("lora_dropout", 0.1),
            "ckpt": config.get("ckpt", ""),
            "torch_dtype": torch_dtype
        })
        salmon_config.setdefault("use_focal_loss", config.get("use_focal_loss", False))
        salmon_config.setdefault("focal_gamma", config.get("focal_gamma", 2.0))
        
        print("before SALMONN.from_config(salmon_config)")
        self.model = SALMONN.from_config(salmon_config)
        print("after SALMONN.from_config(salmon_config)")

        # Enable gradient checkpointing to reduce memory (trades compute for memory)
        if hasattr(self.model.llama_model, "gradient_checkpointing_enable"):
            self.model.llama_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.model.llama_model, "config"):
                self.model.llama_model.config.use_cache = False

        # Ensure trainable parameters are in float32 for GradScaler stability if not using bfloat16
        # This prevents "ValueError: Attempting to unscale FP16 gradients"
        if torch_dtype == torch.float16:
            for p in self.model.parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)

    def forward(self, samples: Dict[str, Any], verbose: bool = False) -> Dict[str, torch.Tensor]:
        # Adapt prompts: <Audio> -> <SpeechHere>
        if "text" in samples and isinstance(samples["text"], list):
            samples["text"] = [t.replace("<Audio>", "<SpeechHere>") for t in samples["text"]]
        if "prompts" in samples and isinstance(samples["prompts"], list):
            samples["prompts"] = [p.replace("<Audio>", "<SpeechHere>") for p in samples["prompts"]]
            
        return self.model(samples, verbose=verbose)

    def generate(self, samples: Dict[str, Any], generate_cfg: Dict[str, Any], prompts: Optional[List[str]] = None, return_outputs: bool = False) -> Union[List[str], Any]:
        # Adapt prompts
        if prompts:
            prompts = [p.replace("<Audio>", "<SpeechHere>") for p in prompts]
        
        num_return_sequences = generate_cfg.get("num_return_sequences", 1)

        samples = {
            k: (
                v.repeat_interleave(num_return_sequences, dim=0) if isinstance(v, torch.Tensor)
                else [x for x in v for _ in range(num_return_sequences)]
            )
            for k, v in samples.items()
        }

        # Note: return_outputs=True is not fully supported by underlying SALMONN yet for ids/logits
        # We assume for now it returns just text unless we modify SALMONN further
        if return_outputs:
            # We can't easily get ids/logits from the current SALMONN.generate without modifying it to return them.
            # For now, just return text and raise warning or error if strict.
            # Or implement a wrapper that tokenizes the output text to get ids?
            texts = self.model.generate(samples, generate_cfg, prompts=prompts)
            
            # Post-hoc tokenization (approximation)
            completion_ids = []
            for t in texts:
                # Add bos? no, it's completion. Add eos? maybe.
                ids = self.model.llama_tokenizer(t, add_special_tokens=False).input_ids
                completion_ids.append(torch.tensor(ids))
            
            # Pad
            completion_ids = torch.nn.utils.rnn.pad_sequence(completion_ids, batch_first=True, padding_value=self.model.llama_tokenizer.pad_token_id)
            
            # Logits? We can run compute_logits_for_completions to get them!
            # But that requires re-running the model.
            # This is acceptable for some workflows (like PPO where we generate, then eval).
            # But if generate was supposed to return them efficiently, this is slow.
            logits = self.compute_logits_for_completions(samples, completion_ids)
            
            return texts, completion_ids, logits
            
        return self.model.generate(samples, generate_cfg, prompts=prompts)

    def compute_logits_for_completions(self, samples: Dict[str, Any], completion_ids: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(samples, completion_ids)

    def get_tokenizer(self):
        return getattr(self.model, "llama_tokenizer", None)

    def generate_text_only(self, prompt_texts: List[str], generate_cfg: Dict[str, Any]) -> List[str]:
        # Bypass speech encoder, just use LLM
        # Tokenize prompts
        inputs = self.model.llama_tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.model.device)
        
        # Generate
        outputs = self.model.llama_model.generate(
            **inputs,
            max_new_tokens=generate_cfg.get("max_new_tokens", 5000),
            do_sample=generate_cfg.get("do_sample", False),
        )
        
        # Decode
        # outputs includes prompt? usually yes.
        # We should slice if we only want new tokens, but usually we decode all.
        decoded = self.model.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Strip prompt from decoded if needed?
        # Llama generate usually returns full sequence.
        return decoded

