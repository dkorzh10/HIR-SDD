# Experiment: baseline_lora64

## Description
Baseline training configuration with LoRA rank 64, serving as the reference point for future experiments.

## Configuration
- **LoRA Rank**: 64
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.1
- **Epochs**: 3
- **Batch Size**: 16 (per GPU)
- **Learning Rate**: 2e-5
- **GPUs**: 4
- **Dataset**: train_salmonn_100k_fixed_targets.json (~74k samples)

## Hypothesis
This baseline should achieve reasonable performance on the antispoofing task. Future experiments will vary hyperparameters to improve upon this baseline.

## Results

### Training
- **Run Date**: TBD
- **Training Time**: TBD
- **Final Loss**: TBD

### Evaluation
- **Accuracy**: TBD
- **Precision**: TBD
- **Recall**: TBD
- **F1 Score**: TBD
- **EER**: TBD
- **AUC-ROC**: TBD

## Notes
- Add observations here after running the experiment
- Document any issues or interesting findings



























