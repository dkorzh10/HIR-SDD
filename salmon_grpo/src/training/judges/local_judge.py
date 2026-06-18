"""
Local model judge implementation (wrapper for existing SALMONN model).
"""

import logging
from typing import List, Dict

from src.training.judges.base_judge import BaseJudge


class LocalJudge(BaseJudge):
    """Judge implementation using the local SALMONN model"""
    
    def __init__(self, model):
        """
        Initialize local judge with SALMONN model.
        
        Args:
            model: SALMONN model instance with generate_text_only method
        """
        self.model = model
        logging.info("Initialized local judge using internal SALMONN model")
    
    def generate_text_only(self, prompt_texts: List[str], generate_cfg: Dict) -> List[str]:
        """
        Generate responses using the local model.
        
        Args:
            prompt_texts: List of prompts (in Vicuna format)
            generate_cfg: Generation configuration dict
        
        Returns:
            List of generated text responses
        """
        # Delegate directly to the model's existing method
        return self.model.generate_text_only(prompt_texts, generate_cfg)

