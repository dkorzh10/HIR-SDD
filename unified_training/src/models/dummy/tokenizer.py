"""
DummyTokenizer for tests. Mimics HuggingFace tokenizer interface (encode, pad_token_id).
Uses simple whitespace+splitting heuristic: ~4 chars per token.
"""
from typing import List, Union, Optional


class DummyTokenizer:
    """Minimal tokenizer for testing. encode() returns token ids (chars // 4 by default)."""

    def __init__(self, chars_per_token: int = 4):
        self.chars_per_token = chars_per_token
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        **kwargs
    ) -> List[int]:
        """Return token ids. Uses len(text) // chars_per_token as approximation."""
        if isinstance(text, list):
            text = text[0] if text else ""
        n = max(0, len(str(text)) // self.chars_per_token)
        return list(range(n))  # Dummy ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        """Decode ids back to placeholder text."""
        return f"<decoded_{len(ids)}_tokens>"
