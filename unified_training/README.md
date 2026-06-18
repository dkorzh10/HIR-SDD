# Unified Training (HIR-SDD)

A unified framework for training and evaluating speech-deepfake-detection models. In this
release it covers the paper's **SALMONN-7B** pipelines (hard-label SFT, CoT SFT, GRPO +
grounding, LLM-as-a-judge) and the conventional **Wav2Vec2-AASIST** baseline.

> Part of [HIR-SDD](../README.md). See the top-level README for the method overview,
> dataset, and results.

## Layout

```
unified_training/
├── manage/                  # launch helpers (run_locally.sh, setup_env.sh)
├── experiment_configs/      # one config.yaml per experiment (source of truth)
│   ├── salmon_hard_label_sft/
│   ├── salmon_sft/                       # CoT SFT
│   ├── salmon_grpo/                       # GRPO + grounding
│   └── conv_audio_classifier_hard_label_sft/   # Wav2Vec2-AASIST baseline
├── src/
│   ├── runner.py            # entrypoint: python -m src.runner --config <cfg>
│   ├── models/              # salmon, conv_audio_classifier, SALMON (encoders)
│   ├── trainers/            # SFT, GRPO, distillation
│   ├── epochs/              # SFT / GRPO / eval / test loops + grounding
│   ├── judges/              # format + LLM-as-a-judge (OpenRouter / local)
│   ├── dataloaders/         # dataset + samplers
│   └── loggers/, utils/, distributed/
├── config_salmon.yaml       # base SALMONN config
└── requirements.txt
```

## Setup

```bash
./manage/setup_env.sh          # optional: prepares a virtualenv
# Install PyTorch first (torch is unpinned in requirements.txt — pick the CUDA build
# you need from https://pytorch.org/get-started/locally/):
pip install torch torchaudio
pip install -r requirements.txt
```

You also need the base checkpoints (Vicuna-7B, Whisper-large-v2, BEATs for SALMONN) from
their original sources; set their paths in the experiment config.

## Configure

All experiment design lives in `experiment_configs/<name>/config.yaml` — hyperparameters,
model selection, dataset paths, and training arguments. Dataset/checkpoint paths are
placeholders (e.g. `/path/to/data`); edit them to point at your copy of the
[HIR-SDD dataset](https://hf.co/datasets/marsianin500/HIR-SDD) and checkpoints.

## Run

```bash
# via the launcher (auto-detects GPUs from the config)
./manage/run_locally.sh salmon_hard_label_sft
./manage/run_locally.sh salmon_sft
./manage/run_locally.sh salmon_grpo
./manage/run_locally.sh conv_audio_classifier_hard_label_sft

# or directly
python -m src.runner --config experiment_configs/salmon_grpo/config.yaml
```

For the LLM-as-a-judge reward/eval, set `OPENROUTER_API_KEY` (or configure a local judge
in the config's `judge` section).
