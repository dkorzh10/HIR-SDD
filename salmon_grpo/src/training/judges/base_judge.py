"""
Abstract base class for LLM judges.
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseJudge(ABC):
    """Abstract base class for all judge implementations"""
    
    @abstractmethod
    def generate_text_only(self, prompt_texts: List[str], generate_cfg: Dict) -> List[str]:
        """
        Generate responses for judge prompts.
        
        Args:
            prompt_texts: List of prompt strings to evaluate
            generate_cfg: Generation configuration dict with keys like:
                - max_new_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
                - do_sample: Whether to use sampling
        
        Returns:
            List of generated text responses (one per prompt)
        """
        pass

