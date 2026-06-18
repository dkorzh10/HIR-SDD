"""
Prompt formatters for different LLM models.

Converts Vicuna-style prompts to model-specific formats.
"""

import re
from typing import Dict, List


class PromptFormatter:
    """Base class for prompt formatting"""
    
    @staticmethod
    def format_prompt(prompt: str) -> str:
        """Format a Vicuna-style prompt for the target model"""
        raise NotImplementedError


class ClaudeFormatter(PromptFormatter):
    """Formatter for Anthropic Claude models"""
    
    @staticmethod
    def format_prompt(prompt: str) -> str:
        """
        Convert Vicuna format to Claude's preferred format.
        Vicuna: "USER: {text}\nASSISTANT: "
        Claude: Just the user text without special tokens
        """
        # Remove "USER: " prefix and "ASSISTANT: " suffix
        prompt = re.sub(r'^USER:\s*', '', prompt, flags=re.MULTILINE)
        prompt = re.sub(r'\s*ASSISTANT:\s*$', '', prompt, flags=re.MULTILINE)
        return prompt.strip()


class GPT4Formatter(PromptFormatter):
    """Formatter for OpenAI GPT-4 models"""
    
    @staticmethod
    def format_prompt(prompt: str) -> str:
        """
        Convert Vicuna format to GPT-4's preferred format.
        Similar to Claude, just clean user text.
        """
        # Remove "USER: " prefix and "ASSISTANT: " suffix
        prompt = re.sub(r'^USER:\s*', '', prompt, flags=re.MULTILINE)
        prompt = re.sub(r'\s*ASSISTANT:\s*$', '', prompt, flags=re.MULTILINE)
        return prompt.strip()


class GeminiFormatter(PromptFormatter):
    """Formatter for Google Gemini models"""
    
    @staticmethod
    def format_prompt(prompt: str) -> str:
        """
        Convert Vicuna format to Gemini's preferred format.
        Gemini works well with clean user text.
        """
        # Remove "USER: " prefix and "ASSISTANT: " suffix
        prompt = re.sub(r'^USER:\s*', '', prompt, flags=re.MULTILINE)
        prompt = re.sub(r'\s*ASSISTANT:\s*$', '', prompt, flags=re.MULTILINE)
        return prompt.strip()


class DeepSeekFormatter(PromptFormatter):
    """Formatter for DeepSeek models"""
    
    @staticmethod
    def format_prompt(prompt: str) -> str:
        """
        Convert Vicuna format to DeepSeek's preferred format.
        DeepSeek uses similar chat format to other models.
        """
        # Remove "USER: " prefix and "ASSISTANT: " suffix
        prompt = re.sub(r'^USER:\s*', '', prompt, flags=re.MULTILINE)
        prompt = re.sub(r'\s*ASSISTANT:\s*$', '', prompt, flags=re.MULTILINE)
        return prompt.strip()


class VicunaFormatter(PromptFormatter):
    """Formatter for Vicuna models (no-op, keeps original format)"""
    
    @staticmethod
    def format_prompt(prompt: str) -> str:
        """Keep Vicuna format as-is"""
        return prompt


# Model name to formatter mapping
MODEL_FORMATTERS: Dict[str, PromptFormatter] = {
    'anthropic/claude-3.5-sonnet': ClaudeFormatter,
    'anthropic/claude-3-5-sonnet': ClaudeFormatter,
    'anthropic/claude-3-opus': ClaudeFormatter,
    'anthropic/claude-3-sonnet': ClaudeFormatter,
    'openai/gpt-4o': GPT4Formatter,
    'openai/gpt-4o-mini': GPT4Formatter,
    'openai/gpt-4-turbo': GPT4Formatter,
    'openai/gpt-4': GPT4Formatter,
    'google/gemini-pro-1.5': GeminiFormatter,
    'google/gemini-flash-1.5': GeminiFormatter,
    'google/gemini-pro': GeminiFormatter,
    'deepseek/deepseek-chat': DeepSeekFormatter,
    'deepseek/deepseek-coder': DeepSeekFormatter,
}


def get_formatter(model_name: str) -> PromptFormatter:
    """
    Get the appropriate formatter for a model.
    
    Args:
        model_name: OpenRouter model identifier (e.g., 'anthropic/claude-3.5-sonnet')
    
    Returns:
        PromptFormatter instance for the model
    """
    formatter_class = MODEL_FORMATTERS.get(model_name, VicunaFormatter)
    return formatter_class()


def format_prompts(prompts: List[str], model_name: str) -> List[str]:
    """
    Format a list of prompts for a specific model.
    
    Args:
        prompts: List of Vicuna-style prompts
        model_name: OpenRouter model identifier
    
    Returns:
        List of formatted prompts
    """
    formatter = get_formatter(model_name)
    return [formatter.format_prompt(p) for p in prompts]

