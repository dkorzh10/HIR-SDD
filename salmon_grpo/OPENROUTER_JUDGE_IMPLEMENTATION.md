# OpenRouter Judge Implementation - Summary

## Overview

Successfully integrated OpenRouter API as an alternative judge for GRPO reward computation. The implementation maintains full backward compatibility while adding support for external LLM judges (Claude 3.5 Sonnet, GPT-4o, Gemini, DeepSeek).

## Files Created

### 1. Judge Package (`src/training/judges/`)

- **`__init__.py`**: Package initialization and exports
- **`base_judge.py`**: Abstract base class defining the judge interface
- **`local_judge.py`**: Wrapper for existing internal SALMONN model judge
- **`openrouter_judge.py`**: OpenRouter API implementation with retry logic
- **`prompt_formatters.py`**: Model-specific prompt formatting for Claude/GPT-4/Gemini/DeepSeek
- **`README.md`**: Comprehensive documentation for the judge system

## Files Modified

### 2. Reward Computer (`src/training/rewards.py`)

**Changes:**
- Refactored `__init__()` to accept `judge` parameter instead of `model`
- Updated `compute_llm_judge_scores()` to use `self.judge.generate_text_only()`
- Added detailed docstrings
- Maintained all existing reward computation logic

### 3. Runner (`src/core/runner.py`)

**Changes:**
- Added judge factory logic in `__init__()`
- Reads `judge` configuration section
- Instantiates appropriate judge (LocalJudge or OpenRouterJudge) based on config
- Passes judge instance to RewardComputer
- Added logging for judge initialization

### 4. Configuration (`running/configs/salmon_grpo_debug/train_config.yaml`)

**Changes:**
- Added new `judge` section with configuration options:
  - `type`: "internal" or "openrouter"
  - `openrouter`: API key, model, retry settings, timeout

## Key Features

### 1. Backward Compatibility
- Default configuration uses internal model judge (existing behavior)
- Existing configs without `judge` section work unchanged
- No breaking changes to existing code

### 2. Robust Error Handling
- 3 retry attempts with exponential backoff (1s, 2s, 4s)
- Detailed error logging for debugging
- Fallback to neutral score (5/10) if all retries fail
- Timeout protection (30s default)

### 3. Automatic Prompt Formatting
- Converts Vicuna-style prompts to model-specific formats
- Supports Claude, GPT-4, Gemini, DeepSeek
- Extensible for new models

### 4. Flexible Configuration
- API key via environment variable or config
- Configurable retry behavior
- Multiple model options per provider

## Configuration Examples

### Using Internal Judge (Default)

```yaml
judge:
  type: "internal"

run:
  judge_weight: 0.5
```

### Using OpenRouter with Claude 3.5 Sonnet

```yaml
judge:
  type: "openrouter"
  openrouter:
    api_key: null  # Uses OPENROUTER_API_KEY env var
    model: "anthropic/claude-3.5-sonnet"
    max_retries: 3
    retry_delay: 1.0
    timeout: 30

run:
  judge_weight: 0.5
```

### Using OpenRouter with GPT-4o

```yaml
judge:
  type: "openrouter"
  openrouter:
    model: "openai/gpt-4o"

run:
  judge_weight: 0.5
```

### Disabling Judge

```yaml
run:
  judge_weight: 0.0  # Judge is disabled regardless of type
```

## Supported Models

| Provider | Model ID | Notes |
|----------|----------|-------|
| Anthropic | `anthropic/claude-3.5-sonnet` | Recommended, best quality |
| Anthropic | `anthropic/claude-3-opus` | High quality |
| Anthropic | `anthropic/claude-3-sonnet` | Balanced |
| OpenAI | `openai/gpt-4o` | Fast, high quality |
| OpenAI | `openai/gpt-4o-mini` | Fast, cost-effective |
| OpenAI | `openai/gpt-4-turbo` | High quality |
| Google | `google/gemini-pro-1.5` | Good quality |
| Google | `google/gemini-flash-1.5` | Fast, cost-effective |
| DeepSeek | `deepseek/deepseek-chat` | Cost-effective |

## Usage Instructions

### 1. Set API Key

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### 2. Update Configuration

Edit `train_config.yaml`:

```yaml
judge:
  type: "openrouter"
  openrouter:
    model: "anthropic/claude-3.5-sonnet"
```

### 3. Run Training

```bash
python train.py --cfg-path running/configs/salmon_grpo_debug/train_config.yaml
```

### 4. Monitor Logs

The system logs:
- Judge initialization: "Initialized OpenRouter judge with model: ..."
- API failures: "OpenRouter API request failed on prompt X, attempt Y/Z: ..."
- Retry attempts: "Retrying in Xs..."
- Fallback scoring: "All X retry attempts failed for prompt Y. Returning neutral score (5)."

## Testing Checklist

- [x] Test with `judge_weight: 0.0` (no judge) - maintains existing behavior
- [x] Test with `judge.type: "internal"` - uses SALMONN model
- [ ] Test with `judge.type: "openrouter"` - requires API key and live testing
- [ ] Test retry logic with simulated failures - requires live testing
- [ ] Test prompt formatting for each supported model - requires live testing

## Architecture Benefits

1. **Clean Separation**: Judge logic isolated in separate package
2. **Extensibility**: Easy to add new judge implementations
3. **Testability**: Each component can be tested independently
4. **Maintainability**: Clear interfaces and documentation
5. **Flexibility**: Switch between judges via configuration

## Cost Estimation

For a training run with:
- `num_grpo_samples: 4`
- `batch_size_train: 2`
- `iters_per_epoch: 50`
- `max_epoch: 2`

Total iterations: 100
API calls per iteration: 8 (2 aspects × 4 samples)
Total API calls: 800

Estimated tokens per call: ~100 input + ~10 output = 110 tokens
Total tokens: ~88,000 tokens

**Cost estimates:**
- Claude 3.5 Sonnet: ~$0.40
- GPT-4o: ~$0.30
- Gemini Pro 1.5: ~$0.15
- DeepSeek: ~$0.05

## Next Steps

1. Test with actual OpenRouter API key
2. Benchmark performance vs internal judge
3. Compare judge quality across different models
4. Consider implementing batch API calls for efficiency
5. Add metrics tracking for judge API usage and costs

