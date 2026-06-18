import torch
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from .llm_as_a_judge import LLMAsAJudge

class LocalLLMJudge(LLMAsAJudge):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        model_name = config.get("model", "Qwen/Qwen2.5-7B-Instruct")
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Load a separate model instance
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map=device,
            local_files_only=False
        )
        self.device = device

    def _generate(self, prompts: List[str]) -> List[str]:
        responses = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get("max_new_tokens", 5000),
                    temperature=self.config.get("temperature", 0.0),
                    do_sample=self.config.get("temperature", 0.0) > 0
                )
            decoded = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(decoded.strip())
        return responses
