# Judge System Usage Examples

## Quick Start

### 1. Using Internal Judge (Default)

No changes needed! The system defaults to using the internal SALMONN model as judge.

```yaml
# train_config.yaml
run:
  judge_weight: 0.5  # Enable judge with 50% weight
```

Or explicitly configure:

```yaml
judge:
  type: "internal"

run:
  judge_weight: 0.5
```

### 2. Using OpenRouter with Claude 3.5 Sonnet

**Step 1:** Get an OpenRouter API key from https://openrouter.ai/

**Step 2:** Set the environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

**Step 3:** Update your config:

```yaml
judge:
  type: "openrouter"
  openrouter:
    model: "anthropic/claude-3.5-sonnet"

run:
  judge_weight: 0.5
```

**Step 4:** Run training:

```bash
python train.py --cfg-path running/configs/salmon_grpo_debug/train_config.yaml
```

## Configuration Examples

### Example 1: Cost-Effective Setup (GPT-4o Mini)

```yaml
judge:
  type: "openrouter"
  openrouter:
    model: "openai/gpt-4o-mini"  # Cheaper than GPT-4o
    max_retries: 3
    retry_delay: 1.0
    timeout: 30

run:
  judge_weight: 0.3  # Lower weight to reduce API calls
  num_grpo_samples: 3  # Fewer samples = fewer API calls
```

### Example 2: High-Quality Setup (Claude 3.5 Sonnet)

```yaml
judge:
  type: "openrouter"
  openrouter:
    model: "anthropic/claude-3.5-sonnet"
    max_retries: 5  # More retries for reliability
    retry_delay: 2.0
    timeout: 60

run:
  judge_weight: 0.5
  num_grpo_samples: 4
```

### Example 3: Fast Iteration Setup (Gemini Flash)

```yaml
judge:
  type: "openrouter"
  openrouter:
    model: "google/gemini-flash-1.5"  # Fast and cheap
    max_retries: 2
    timeout: 20

run:
  judge_weight: 0.4
```

### Example 4: A/B Testing (Switch Between Judges)

**Config A - Internal:**
```yaml
judge:
  type: "internal"
run:
  judge_weight: 0.5
  output_dir: /path/to/experiments/internal_judge_run
```

**Config B - OpenRouter:**
```yaml
judge:
  type: "openrouter"
  openrouter:
    model: "anthropic/claude-3.5-sonnet"
run:
  judge_weight: 0.5
  output_dir: /path/to/experiments/openrouter_judge_run
```

Compare results from both runs!

### Example 5: Disabling Judge

```yaml
run:
  judge_weight: 0.0  # Judge disabled, only uses correctness + validity
```

## Custom Judge Prompts

You can customize the evaluation prompts:

```yaml
run:
  judge_weight: 0.5
  judge_prompts:
    - name: "correctness"
      template: |
        USER: Rate how well this answer matches the reference on a scale of 1-10.
        
        Reference: {gt_text}
        
        Answer: {pred_text}
        
        Your rating (1-10):
        ASSISTANT: 
      weight: 0.7  # Increased weight for correctness
    
    - name: "quality"
      template: |
        USER: Rate the quality of this text on a scale of 1-10 (1=gibberish, 10=perfect).
        
        Text: {pred_text}
        
        Your rating (1-10):
        ASSISTANT: 
      weight: 0.3
```

## Monitoring and Debugging

### Check Judge Logs

Judge responses are saved in the output directory:

```bash
# View judge logs for a specific iteration
cat /path/to/output_dir/judge_logs/epoch_0_iter_50.jsonl
```

Example log entry:

```json
{
  "audio_id": "sample_001",
  "ground_truth": "This is bonafide speech.",
  "model_answer": "This is spoof speech.",
  "llm_judge_responses": {
    "correctness": {
      "response": "3",
      "parsed_score": 3.0
    },
    "quality": {
      "response": "7",
      "parsed_score": 7.0
    }
  },
  "judge_score": 0.46,
  "validity_score": 1.0,
  "correctness_score": 0.0,
  "overall_reward": 0.33
}
```

### Monitor Training Logs

```bash
# Watch training progress
tail -f /path/to/output_dir/log.txt
```

Look for:
- `"Initialized OpenRouter judge with model: ..."` - Judge initialized
- `"OpenRouter API request failed..."` - API errors
- `"Retrying in Xs..."` - Retry attempts
- `"All X retry attempts failed..."` - Fallback to neutral score

## Troubleshooting

### Issue: "OpenRouter API key not provided"

**Solution:** Set the environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Or add to config:

```yaml
judge:
  openrouter:
    api_key: "sk-or-v1-..."
```

### Issue: API calls timing out

**Solution:** Increase timeout:

```yaml
judge:
  openrouter:
    timeout: 60  # Increase from default 30s
```

### Issue: Too many API failures

**Solution:** Increase retries and delay:

```yaml
judge:
  openrouter:
    max_retries: 5
    retry_delay: 2.0
```

### Issue: Training is slow with OpenRouter

**Solutions:**
1. Use a faster model (Gemini Flash, GPT-4o Mini)
2. Reduce `num_grpo_samples`
3. Lower `judge_weight`
4. Use internal judge for quick iterations

### Issue: Unexpected judge scores

**Solution:** Check judge logs to see raw responses:

```bash
cat /path/to/output_dir/judge_logs/epoch_0_iter_0.jsonl | jq '.llm_judge_responses'
```

## Cost Optimization Tips

1. **Use cheaper models for experimentation:**
   - `openai/gpt-4o-mini` (~$0.10/M tokens)
   - `google/gemini-flash-1.5` (~$0.075/M tokens)
   - `deepseek/deepseek-chat` (~$0.03/M tokens)

2. **Reduce API calls:**
   - Lower `num_grpo_samples` (e.g., 3 instead of 4)
   - Reduce `batch_size_train` if needed
   - Fewer epochs for testing

3. **Adjust judge weight:**
   - Use `judge_weight: 0.3` instead of `0.5` for less influence

4. **Use internal judge for debugging:**
   - Switch to `type: "internal"` for quick iterations
   - Use OpenRouter only for final runs

## Performance Comparison

| Judge Type | Speed | Cost | GPU Memory | Quality |
|------------|-------|------|------------|---------|
| Internal | Fast | Free | High | Good |
| OpenRouter (Claude) | Slow | $$$ | Low | Excellent |
| OpenRouter (GPT-4o) | Medium | $$ | Low | Excellent |
| OpenRouter (Gemini Flash) | Fast | $ | Low | Good |
| OpenRouter (DeepSeek) | Fast | $ | Low | Good |

## Best Practices

1. **Start with internal judge** for quick experiments
2. **Use OpenRouter** for final training runs or when you need:
   - Better evaluation quality
   - More GPU memory for the main model
   - Consistent scoring across runs
3. **Monitor costs** by checking API usage on OpenRouter dashboard
4. **Test prompts** with a small number of iterations first
5. **Keep logs** for debugging and analysis
6. **A/B test** different judges to find what works best for your task

