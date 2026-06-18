"""
OpenRouter API judge implementation.
"""

import os
import time
import logging
from typing import List, Dict, Optional

import requests

from src.training.judges.base_judge import BaseJudge
from src.training.judges.prompt_formatters import format_prompts


class OpenRouterJudge(BaseJudge):
    """Judge implementation using OpenRouter API"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize OpenRouter judge.
        
        Args:
            config: Configuration dict with keys:
                - api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
                - model: Model identifier (e.g., 'anthropic/claude-3.5-sonnet')
                - max_retries: Maximum retry attempts (default: 3)
                - retry_delay: Initial retry delay in seconds (default: 1.0)
                - timeout: Request timeout in seconds (default: 30)
        """
        config = config or {}
        
        self.api_key = config.get('api_key') or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment "
                "variable or provide 'api_key' in judge.openrouter config."
            )
        
        self.model = config.get('model', 'anthropic/claude-3.5-sonnet')
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.timeout = config.get('timeout', 30)
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        logging.info(f"Initialized OpenRouter judge with model: {self.model}")
        logging.info(f"Retry config: max_retries={self.max_retries}, "
                    f"retry_delay={self.retry_delay}s, timeout={self.timeout}s")
    
    def generate_text_only(self, prompt_texts: List[str], generate_cfg: Dict) -> List[str]:
        """
        Generate responses using OpenRouter API.
        
        Args:
            prompt_texts: List of prompts (in Vicuna format)
            generate_cfg: Generation configuration dict
        
        Returns:
            List of generated text responses
        """
        # Format prompts for the target model
        formatted_prompts = format_prompts(prompt_texts, self.model)
        
        responses = []
        for i, prompt in enumerate(formatted_prompts):
            response = self._generate_single(prompt, generate_cfg, prompt_index=i)
            responses.append(response)
        
        return responses
    
    def _generate_single(self, prompt: str, generate_cfg: Dict, prompt_index: int = 0) -> str:
        """
        Generate a single response with retry logic.
        
        Args:
            prompt: Formatted prompt text
            generate_cfg: Generation configuration
            prompt_index: Index of prompt (for logging)
        
        Returns:
            Generated text response, or "5" (neutral score) if all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                response = self._call_api(prompt, generate_cfg)
                return response
            
            except requests.exceptions.Timeout as e:
                logging.warning(
                    f"OpenRouter API timeout on prompt {prompt_index}, "
                    f"attempt {attempt + 1}/{self.max_retries}: {e}"
                )
            
            except requests.exceptions.RequestException as e:
                logging.warning(
                    f"OpenRouter API request failed on prompt {prompt_index}, "
                    f"attempt {attempt + 1}/{self.max_retries}: {e}"
                )
            
            except Exception as e:
                logging.error(
                    f"Unexpected error calling OpenRouter API on prompt {prompt_index}, "
                    f"attempt {attempt + 1}/{self.max_retries}: {e}"
                )
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logging.info(f"Retrying in {delay}s...")
                time.sleep(delay)
        
        # All retries exhausted, return neutral score
        logging.error(
            f"All {self.max_retries} retry attempts failed for prompt {prompt_index}. "
            f"Returning neutral score (5)."
        )
        return "5"
    
    def _call_api(self, prompt: str, generate_cfg: Dict) -> str:
        """
        Make a single API call to OpenRouter.
        
        Args:
            prompt: Formatted prompt text
            generate_cfg: Generation configuration
        
        Returns:
            Generated text response
        
        Raises:
            requests.exceptions.RequestException: On API errors
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/salmon-grpo",
            "X-Title": "SALMON-GRPO-Judge"
        }
        
        # Build request payload
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": generate_cfg.get("max_new_tokens", 10),
            "temperature": generate_cfg.get("temperature", 0.0),
        }
        
        # Make API request
        response = requests.post(
            self.base_url,
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        if "choices" not in result or len(result["choices"]) == 0:
            raise ValueError(f"Invalid OpenRouter API response: {result}")
        
        generated_text = result["choices"][0]["message"]["content"].strip()
        
        return generated_text

