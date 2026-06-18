# SALMONN Antispoofing (GRPO)

Standalone pipeline for audio deepfake detection with **SALMONN** (Whisper + BEATs →
Q-Former → Vicuna-7B) using LoRA fine-tuning and **GRPO** with a format / class / judge
reward. Self-contained: it does not depend on `unified_training`.

> Part of [HIR-SDD](../README.md). See the top-level README for the method overview,
> dataset, and results.

## Layout

```
salmon_grpo/
├── train.py                      # entrypoint: python train.py --cfg-path <cfg>
├── running/configs/              # representative train_config.yaml + eval_config.yaml
│   ├── baseline_lora128/         # LoRA-128 hard-label baseline
│   ├── lora_128_balanced/        # balanced-data variant
│   └── salmon_grpo/              # GRPO + judge reward
├── src/
│   ├── core/                     # config, dataset, runner
│   ├── models/                   # SALMONN (Qformer, whisper, llama) + utils
│   ├── training/                 # grpo_trainer, rewards, validator, judges
│   ├── evaluation/               # evaluate_test
│   ├── infrastructure/           # dist utils, logging, optims
│   └── util_classes/             # parsing / file / model helpers
├── JUDGE_USAGE_EXAMPLES.md       # LLM-as-a-judge usage
├── OPENROUTER_JUDGE_IMPLEMENTATION.md
└── prompts/                      # antispoofing prompts (hard-label + reasoning)
```

## Setup

```bash
pip install -r requirements.txt
```

Obtain the base SALMONN checkpoints (Vicuna-7B, Whisper-large-v2, BEATs) from their
original sources and set their paths in the config.

## Configure

Edit a config under `running/configs/<exp>/`. Dataset and checkpoint paths are
placeholders (e.g. `/path/to/data`) — point them at your copy of the
[HIR-SDD dataset](https://hf.co/datasets/marsianin500/HIR-SDD) and checkpoints. Key knobs:
LoRA rank/alpha, learning-rate schedule, audio length, and (for GRPO) the format/class/
judge reward weights.

Each dataset sample is JSON:

```json
{"path": "/path/to/audio.flac", "text": "Final Answer: spoof",
 "task": "antispoofing", "Q": "", "audio_id": "...", "language": "en"}
```

## Train

```bash
python train.py --cfg-path running/configs/salmon_grpo/train_config.yaml
```

For the GRPO judge reward, set `OPENROUTER_API_KEY` (OpenRouter judge) or configure a
local judge — see [JUDGE_USAGE_EXAMPLES.md](./JUDGE_USAGE_EXAMPLES.md) and
[OPENROUTER_JUDGE_IMPLEMENTATION.md](./OPENROUTER_JUDGE_IMPLEMENTATION.md).

## References

SALMONN — https://github.com/bytedance/SALMONN · Vicuna — https://github.com/lm-sys/FastChat
· Whisper — https://github.com/openai/whisper
