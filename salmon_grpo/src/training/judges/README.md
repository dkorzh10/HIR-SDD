# Judge System for SALMON GRPO

This package provides a flexible judge system for LLM-as-a-Judge reward computation in GRPO training.

## Architecture

The judge system uses a plugin architecture with the following components:

- **BaseJudge**: Abstract base class defining the judge interface
- **LocalJudge**: Uses the internal SALMONN model as judge (default behavior)
- **OpenRouterJudge**: Uses OpenRouter API to access external LLMs (Claude, GPT-4, Gemini, DeepSeek)
- **PromptFormatters**: Automatically converts Vicuna-style prompts to model-specific formats

## Configuration

### Using Internal Model (Default)

```yaml
judge:
  type: "internal"
```

This uses the SALMONN model's LLaMA component as the judge, maintaining backward compatibility with existing behavior.

### Using OpenRouter API

```yaml
judge:
  type: "openrouter"
  openrouter:
    api_key: null  # Uses OPENROUTER_API_KEY env var
    model: "anthropic/claude-3.5-sonnet"
    max_retries: 3
    retry_delay: 1.0
    timeout: 30
```

#### Supported Models

- **Claude**: `anthropic/claude-3.5-sonnet`, `anthropic/claude-3-opus`, `anthropic/claude-3-sonnet`
- **GPT-4**: `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-4-turbo`, `openai/gpt-4`
- **Gemini**: `google/gemini-pro-1.5`, `google/gemini-flash-1.5`, `google/gemini-pro`
- **DeepSeek**: `deepseek/deepseek-chat`, `deepseek/deepseek-coder`

## Environment Setup

To use OpenRouter, set your API key:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Or specify it directly in the config (not recommended for security):

```yaml
judge:
  type: "openrouter"
  openrouter:
    api_key: "sk-or-v1-..."
```

## Error Handling

OpenRouterJudge implements robust error handling:

1. **Retry Logic**: Automatically retries failed API calls up to `max_retries` times
2. **Exponential Backoff**: Delay doubles after each retry (1s, 2s, 4s, ...)
3. **Fallback Scoring**: Returns neutral score (5/10) if all retries fail
4. **Detailed Logging**: All errors are logged for debugging

## Prompt Formatting

The system automatically converts Vicuna-style prompts to model-specific formats:

**Input (Vicuna format):**
```
USER: Rate how well this answer matches the reference on a scale of 1-10.

Reference: This is bonafide speech.

Answer: This is spoof speech.

Your rating (1-10):
ASSISTANT: 
```

**Output (Claude/GPT-4/Gemini format):**
```
Rate how well this answer matches the reference on a scale of 1-10.

Reference: This is bonafide speech.

Answer: This is spoof speech.

Your rating (1-10):
```

## Adding New Judge Implementations

To add a new judge type:

1. Create a new class inheriting from `BaseJudge`
2. Implement the `generate_text_only(prompt_texts, generate_cfg)` method
3. Add the import to `__init__.py`
4. Update the factory logic in `src/core/runner.py`

Example:

```python
from src.training.judges.base_judge import BaseJudge

class CustomJudge(BaseJudge):
    def __init__(self, config):
        # Initialize your judge
        pass
    
    def generate_text_only(self, prompt_texts, generate_cfg):
        # Generate responses
        return responses
```

## Testing

To test the judge system:

1. **Test with no judge** (judge_weight: 0.0):
   ```yaml
   run:
     judge_weight: 0.0
   ```

2. **Test with internal judge**:
   ```yaml
   judge:
     type: "internal"
   run:
     judge_weight: 0.5
   ```

3. **Test with OpenRouter**:
   ```yaml
   judge:
     type: "openrouter"
     openrouter:
       model: "anthropic/claude-3.5-sonnet"
   run:
     judge_weight: 0.5
   ```

## Performance Considerations

- **Internal Judge**: Fast, no API costs, uses GPU memory
- **OpenRouter Judge**: Slower (network latency), API costs, frees GPU memory

For training runs with many iterations, consider:
- Using faster/cheaper models (e.g., `openai/gpt-4o-mini`, `google/gemini-flash-1.5`)
- Reducing `judge_weight` if judge scores are less important
- Using internal judge for quick experiments, OpenRouter for final runs

