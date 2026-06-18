import logging
import re
import torch
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.judges.base_judge import BaseJudge

class NeighborJudge(BaseJudge):
    """
    Judge implementation using a secondary 'neighbor' HuggingFace model 
    loaded locally (e.g., Qwen2.5-3B).
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Neighbor Judge.
        
        Args:
            config: Configuration dict with keys:
                - model_path: HF model ID (default: 'Qwen/Qwen2.5-3B-Instruct')
                - device_map: 'auto', 'cuda:1', etc. (default: 'auto')
                - torch_dtype: 'float16', 'bfloat16', or 'auto' (default: 'auto')
                - load_in_4bit: bool (default: False) - highly recommended for saving VRAM
        """
        self.model_path = config.get('model_path', 'Qwen/Qwen2.5-3B-Instruct')
        self.device_map = config.get('device_map', 'auto')
        self.torch_dtype = config.get('torch_dtype', 'auto')
        load_in_4bit = config.get('load_in_4bit', False)
        
        logging.info(f"Initializing Neighbor Judge with model: {self.model_path}")
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        # Configure tokenizer for batch generation (left padding is usually safer for decoder-only)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load Model
        quantization_config = None
        if load_in_4bit:
            try:
                import bitsandbytes
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
            except (ImportError, ValueError) as e:
                logging.warning(f"Could not load bitsandbytes for 4-bit quantization: {e}. Falling back to full precision.")
                load_in_4bit = False
                quantization_config = None
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        self.model.eval()
        
    def generate_text_only(self, prompt_texts: List[str], generate_cfg: Dict) -> List[str]:
        """
        Generate responses using the neighbor model in batches.
        """
        # Default generation settings
        max_new_tokens = generate_cfg.get("max_new_tokens", 50)
        temperature = generate_cfg.get("temperature", 0.0)
        do_sample = generate_cfg.get("do_sample", False)
        
        # Pre-process all prompts
        processed_inputs = []
        for prompt in prompt_texts:
            # Convert Vicuna-style prompt to the neighbor model's chat template
            clean_prompt = self._clean_vicuna_prompt(prompt)
            
            # Apply chat template
            messages = [
                {"role": "system", "content": "You are a strict and helpful judge."},
                {"role": "user", "content": clean_prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            processed_inputs.append(text)
            
        # Batch tokenization
        inputs = self.tokenizer(
            processed_inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.model.device)
        
        # Batch generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # Batch decode
        # We need to slice off the input tokens to get only the generated part
        # However, because of left padding, direct slicing is tricky if inputs have variable lengths.
        # Standard robust way: decode everything and then strip the input prompt text, 
        # or careful slicing.
        # Given 'inputs' are left-padded, 'outputs' will also be left-padded (usually).
        # But generate() output usually includes the full sequence (input + new tokens).
        
        # A safer approach for batched generation decoding:
        # Calculate the length of input_ids for each sample (excluding padding if possible, 
        # but the model outputs include the padding too).
        # Actually, for left padding, the new tokens are simply appended.
        # We can just decode the new tokens.
        
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[:, input_len:]
        
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
        return responses

    def _clean_vicuna_prompt(self, prompt: str) -> str:
        """
        Strips Vicuna formatting ('USER:', 'ASSISTANT:') to get clean content 
        for the neighbor model's own template.
        """
        # Remove system prompt prefix if present (heuristic)
        if "A chat between a curious user" in prompt:
            prompt = prompt.split("USER:", 1)[-1]
        
        # Remove USER/ASSISTANT tags
        prompt = re.sub(r'^USER:\s*', '', prompt, flags=re.MULTILINE)
        prompt = re.sub(r'\s*ASSISTANT:\s*$', '', prompt, flags=re.MULTILINE)
        
        return prompt.strip()

