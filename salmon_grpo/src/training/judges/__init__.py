"""
Judge implementations for LLM-as-a-Judge reward computation.

This package provides different judge implementations:
- BaseJudge: Abstract interface for all judges
- LocalJudge: Uses the internal SALMONN model as judge
- OpenRouterJudge: Uses OpenRouter API with various LLM models
"""

from src.training.judges.base_judge import BaseJudge
from src.training.judges.local_judge import LocalJudge
from src.training.judges.openrouter_judge import OpenRouterJudge
from src.training.judges.neighbor_judge import NeighborJudge

__all__ = ['BaseJudge', 'LocalJudge', 'OpenRouterJudge', 'NeighborJudge']

