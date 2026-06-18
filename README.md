<div align="center">

# HIR-SDD: Towards Robust Speech Deepfake Detection via Human-Inspired Reasoning

**Accepted at INTERSPEECH 2026**

[![Paper](https://img.shields.io/badge/Paper-INTERSPEECH%202026-b31b1b.svg)](https://github.com/dkorzh10/HIR-SDD)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-HIR--SDD-yellow.svg)](https://hf.co/datasets/marsianin500/HIR-SDD)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

</div>

---

HIR-SDD is a framework for **speech deepfake detection (SDD)** that pairs the binary
"bona fide vs. spoof" decision with **human-interpretable, audio-grounded reasoning**.
It combines Large Audio Language Models (LALMs) with chain-of-thought (CoT) supervision
derived from a new **human-annotated reasoning dataset**, plus grounding and
reinforcement learning (GRPO) to keep explanations anchored to real acoustic evidence.

> **Abstract.** Current SDD methods generalize poorly to new audio domains and generators
> and offer little interpretability. We propose HIR-SDD, which fine-tunes a LALM
> (SALMONN) with hard-label and CoT supervision, audio grounding, and GRPO, trained on a
> novel dataset of human-annotated reasoning traces for 41k bona fide and spoof samples.
> The result is competitive detection performance together with reasonable, human-like
> justifications for its predictions.

## What's in this repository

This repo releases the **two training/evaluation codebases** behind the paper, kept
**separate and self-contained**:

| Directory | What it is | Covers |
|-----------|-----------|--------|
| [`unified_training/`](./unified_training) | The unified training framework | Hard-label SFT, CoT SFT, **GRPO + grounding**, LLM-as-a-judge, and the **Wav2Vec2-AASIST** baseline |
| [`salmon_grpo/`](./salmon_grpo) | A standalone SALMONN + GRPO pipeline | SALMONN LoRA fine-tuning and GRPO with the OpenRouter/local judge |

Both pipelines train **SALMONN-7B** (Whisper + BEATs encoders → Q-Former → Vicuna-7B,
LoRA-tuned). The `unified_training` framework additionally implements the conventional
**Wav2Vec2-AASIST** countermeasure used as a baseline in the paper.

> This is a **runnable reference** release: it ships the method code, one representative
> config per training stage, and `requirements`. It does **not** ship datasets, model
> checkpoints, or experiment logs — point the configs at your own data/checkpoints (paths
> in configs are placeholders such as `/path/to/data`). The dataset is on
> [🤗 Hugging Face](https://hf.co/datasets/marsianin500/HIR-SDD).

## Method at a glance

1. **Hard-label SFT** — LoRA fine-tuning of SALMONN to emit `Final Answer: Real/Fake`.
2. **CoT SFT** — train on human-derived reasoning traces to emit structured output:
   ```
   <think> free-form reasoning grounded in acoustic cues </think>
   <reasons>[ detected cue tags from the annotation taxonomy ]</reasons>
   <answer> Real / Fake </answer>
   ```
3. **Audio grounding** — perturb audio deterministically (Gaussian noise, time masking,
   gain) so the model anchors its `<think>`/`<reasons>` to perceptible evidence.
4. **GRPO** — reinforcement learning with reward terms for *format*, *class correctness*,
   and a *judge* score (an LLM rates coverage / relevance / logic / helpfulness).

## The dataset

A human-annotated reasoning corpus released separately on Hugging Face:
[`marsianin500/HIR-SDD`](https://hf.co/datasets/marsianin500/HIR-SDD).

- **41,414** audio samples (32,045 spoof / 9,369 bona fide), **124,410** annotations from
  **37** annotators after filtering, **120,258** free-form comments.
- English & Russian; curated from open SDD datasets plus newly synthesized samples.
- Splits usable for binary classification or CoT training/eval:
  - **Hard-label (HL):** `Train-1-HL`, `Val-1-HL`, `Test-1-HL` (ASVspoof 5 eval subset).
  - **Reasoning (R):** `Train-2-*` / `Test-2-R` — 114k / 8k / 1k train/val/test traces.

## Installation

Each pipeline has its own dependencies. SALMONN/BEATs/Whisper checkpoints must be obtained
from their original sources (see per-pipeline READMEs).

```bash
# unified_training
cd unified_training
pip install -r requirements.txt

# salmon_grpo
cd salmon_grpo
pip install -r requirements.txt
```

## Quickstart

**`unified_training`** — experiments are defined as configs under
`experiment_configs/<name>/config.yaml` (the source of truth for hyperparameters):

```bash
cd unified_training
# Edit experiment_configs/<name>/config.yaml: set data paths + checkpoints, then:
./manage/run_locally.sh salmon_hard_label_sft           # hard-label SFT
./manage/run_locally.sh salmon_sft                      # CoT SFT
./manage/run_locally.sh salmon_grpo                     # GRPO + grounding
./manage/run_locally.sh conv_audio_classifier_hard_label_sft   # Wav2Vec2-AASIST baseline
# equivalently: python -m src.runner --config experiment_configs/<name>/config.yaml
```

**`salmon_grpo`** — single-entrypoint training driven by a YAML config:

```bash
cd salmon_grpo
python train.py --cfg-path running/configs/salmon_grpo/train_config.yaml
```

## Results

Detection performance on `Test-1-HL` (positive class = bona fide; from the paper):

| Model | Train set | Accuracy | Balanced Acc. | F1 |
|-------|-----------|:--------:|:-------------:|:--:|
| Wav2Vec2-AASIST | Train-1-HL | 92.3 | 81.3 | 76.7 |
| Wav2Vec2-AASIST | Train-2-HL | 92.9 | 84.0 | 76.7 |
| SALMONN-7B | Train-1-HL | 93.4 | 89.3 | 84.5 |
| SALMONN-7B | Train-2-HL | 94.5 | 88.6 | 85.7 |
| SALMONN-7B | Train-2-R | 92.9 | 86.7 | 81.4 |
| SALMONN-7B | Train-2-R + Val-1-GRPO | **93.6** | **89.6** | **85.0** |

Binary-classification EER on `Test-1-HL`: **13.2%** (SALMONN-7B) vs **15.7%**
(Wav2Vec2-AASIST). GRPO improves reasoning-trace quality (LLM-as-a-judge: 5.74 vs 5.12
for SFT) and diversifies cues. See the paper for the full tables and reasoning examples.

## Citation

```bibtex
@inproceedings{hirsdd2026,
  title     = {Towards Robust Speech Deepfake Detection via Human-Inspired Reasoning},
  author    = {Dvirniak, Artem and Kushnir, Evgeny and Tarasov, Dmitrii and
               Iudin, Artem and Kiriukhin, Oleg and Pautov, Mikhail and
               Korzh, Dmitrii and Rogov, Oleg Y.},
  booktitle = {Proc. INTERSPEECH 2026},
  year      = {2026}
}
```

## License & acknowledgements

Released under the [MIT License](./LICENSE). The code builds on and vendors components
from upstream projects under their respective licenses — please cite and respect them:
[SALMONN](https://github.com/bytedance/SALMONN), [BEATs](https://github.com/microsoft/unilm/tree/master/beats),
[Whisper](https://github.com/openai/whisper), [Vicuna/FastChat](https://github.com/lm-sys/FastChat),
and [AASIST](https://github.com/clovaai/aasist).

This work was supported by a grant for AI research centers from the Ministry of Economic
Development of the Russian Federation (agreements 000000C313925P4F0002 and №139-10-2025-033).
