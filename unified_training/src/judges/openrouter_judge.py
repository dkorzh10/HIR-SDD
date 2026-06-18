import os
import time
import logging
import requests
from typing import List, Dict, Any
from .llm_as_a_judge import LLMAsAJudge

class OpenRouterJudge(LLMAsAJudge):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('token') or os.getenv('OPENROUTER_API_KEY')
        self.model = config.get('model', 'anthropic/claude-3.5-sonnet')
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.timeout = config.get('timeout', 30)
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def _generate(self, prompts: List[str]) -> List[str]:
        responses = []
        for prompt in prompts:
            responses.append(self._generate_with_retry(prompt))
        return responses

    def _generate_with_retry(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0
                }
                response = requests.post(self.base_url, headers=headers, json=data, timeout=self.timeout)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content'].strip()
            except Exception as e:
                logging.warning(f"OpenRouter attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
        return ""
