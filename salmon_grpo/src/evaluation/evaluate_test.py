#!/usr/bin/env python3
"""
Evaluate SALMONN antispoofing model on test data
Computes accuracy, precision, recall, F1, EER, and other metrics
Supports multi-GPU evaluation
"""
import sys
import json
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import WhisperFeatureExtractor

from models.salmonn import SALMONN
from dataset import SALMONNDataset


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    
    return rank, world_size, local_rank


def parse_prediction(pred_text):
    """Parse model output to binary prediction
    
    Lenient parsing: accepts any text containing the keywords
    """
    pred_text = pred_text.lower().strip()
    
    # Check for bonafide indicators (check first as it's rarer)
    if 'bonafide' in pred_text or 'genuine' in pred_text:
        return 'bonafide', 1.0
    # Check for spoof/fake indicators
    elif 'spoof' in pred_text or 'fake' in pred_text:
        return 'fake', 0.0
    else:
        # No fallback - return unknown for unparseable predictions
        return 'unknown', -1.0


def compute_metrics(y_true, y_pred, y_scores=None):
    """Compute classification metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, roc_curve
    )
    
    metrics = {}
    
    # Handle empty arrays (no valid predictions)
    if len(y_true) == 0 or len(y_pred) == 0:
        print("\n⚠️  ERROR: No valid predictions to compute metrics")
        metrics['accuracy'] = 0.0
        metrics['balanced_accuracy'] = 0.0
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1'] = 0.0
        metrics['true_negatives'] = 0
        metrics['false_positives'] = 0
        metrics['false_negatives'] = 0
        metrics['true_positives'] = 0
        metrics['specificity'] = 0.0
        metrics['false_positive_rate'] = 0.0
        metrics['false_negative_rate'] = 0.0
        metrics['eer'] = None
        metrics['auc_roc'] = None
        return metrics
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:  # Normal case: 2x2 matrix
        tn, fp, fn, tp = cm.ravel()
    elif cm.size == 1:  # Edge case: only one class predicted
        # If only one class, we need to determine which one
        if cm[0, 0] > 0:
            # Only predicted class 0
            tn = int(cm[0, 0])
            fp = 0
            fn = 0
            tp = 0
        else:
            tn = 0
            fp = 0
            fn = 0
            tp = 0
    else:
        # Unexpected case
        tn = fp = fn = tp = 0
    
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Balanced accuracy = (recall + specificity) / 2
    # Also known as (TPR + TNR) / 2 or (sensitivity + specificity) / 2
    metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
    
    # False rates
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # EER and AUC if scores available
    if y_scores is not None and len(y_scores) > 0:
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.absolute(fnr - fpr))
            metrics['eer'] = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
            metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
        except:
            metrics['eer'] = None
            metrics['auc_roc'] = None
    else:
        metrics['eer'] = None
        metrics['auc_roc'] = None
    
    return metrics


def evaluate(model, dataloader, device, rank, world_size, output_dir=None, prompt_dict=None, generate_cfg=None):
    """Run evaluation with incremental saving"""
    model.eval()
    
    # Get the actual model (unwrap DDP if needed)
    actual_model = model.module if hasattr(model, 'module') else model
    
    all_predictions = []
    all_labels = []
    all_scores = []
    all_texts = []
    
    # Setup incremental save file (only rank 0)
    incremental_file = None
    if rank == 0 and output_dir:
        incremental_file = Path(output_dir) / 'predictions_incremental.jsonl'
        incremental_file.parent.mkdir(parents=True, exist_ok=True)
        # Clear file if it exists
        if incremental_file.exists():
            incremental_file.unlink()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating (GPU {rank})", disable=rank != 0)):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Prepare prompts (exactly like runner.py validation)
            if actual_model.prompt_dict:
                if prompt_dict is None:
                    prompts = None
                else:
                    import random
                    prompts = []
                    for task in batch["task"]:
                        # Use external test_prompt_dict (already formatted with template)
                        prompt_val = prompt_dict[task]
                        if isinstance(prompt_val, list):
                            prompts.append(random.choice(prompt_val))
                        else:
                            prompts.append(prompt_val)
                    if "Q" in batch:
                        prompts = [p.format(q) if "{}" in p else p for p, q in zip(prompts, batch["Q"])]
            else:
                prompts = None
            
            # Generate predictions using config from config file
            # Use provided generate_cfg or create default
            if generate_cfg is None:
                generate_cfg = {
                    "max_new_tokens": 50,
                    "num_beams": 1,
                    "do_sample": False,
                    "min_length": 1,
                    "top_p": 0.9,
                    "repetition_penalty": 1.0,
                    "length_penalty": 1.0,
                    "temperature": 1.0,
                }
            
            outputs = actual_model.generate(batch, generate_cfg, prompts=prompts)
            
            # Parse predictions and save incrementally
            batch_results = []
            for i, output in enumerate(outputs):
                pred_label, pred_score = parse_prediction(output)
                true_label = batch['text'][i].lower().strip()
                
                # Handle unknown predictions (don't fallback, mark as -1)
                if pred_label == 'unknown':
                    pred_binary = -1
                else:
                    pred_binary = 1 if pred_label == 'bonafide' else 0
                    
                label_binary = 1 if 'bonafide' in true_label else 0
                
                all_predictions.append(pred_binary)
                all_labels.append(label_binary)
                all_scores.append(pred_score)
                all_texts.append(output)
                
                # Collect for incremental save
                if rank == 0:
                    batch_results.append({
                        'index': len(all_predictions) - 1,
                        'prediction': pred_label,  # Save as string for clarity
                        'prediction_binary': int(pred_binary),
                        'label': int(label_binary),
                        'score': float(pred_score),
                        'text': output,
                        'correct': bool(pred_binary == label_binary) if pred_binary != -1 else False
                    })
            
            # Save batch results incrementally (only rank 0)
            if rank == 0 and incremental_file and batch_results:
                with open(incremental_file, 'a') as f:
                    for result in batch_results:
                        f.write(json.dumps(result) + '\n')
    
    # Gather results from all GPUs
    if world_size > 1:
        # Convert to tensors for gathering
        pred_tensor = torch.tensor(all_predictions, device=device)
        label_tensor = torch.tensor(all_labels, device=device)
        score_tensor = torch.tensor(all_scores, device=device)
        
        # Gather
        gathered_preds = [torch.zeros_like(pred_tensor) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(label_tensor) for _ in range(world_size)]
        gathered_scores = [torch.zeros_like(score_tensor) for _ in range(world_size)]
        
        dist.all_gather(gathered_preds, pred_tensor)
        dist.all_gather(gathered_labels, label_tensor)
        dist.all_gather(gathered_scores, score_tensor)
        
        # Gather texts using all_gather_object (for strings)
        gathered_texts = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_texts, all_texts)
        
        if rank == 0:
            all_predictions = torch.cat(gathered_preds).cpu().numpy()
            all_labels = torch.cat(gathered_labels).cpu().numpy()
            all_scores = torch.cat(gathered_scores).cpu().numpy()
            # Flatten the gathered texts
            all_texts = [text for texts_list in gathered_texts for text in texts_list]
    else:
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
    
    return all_predictions, all_labels, all_scores, all_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str, required=True, help='Path to config file')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test JSON')
    parser.add_argument('--output', type=str, default='results/evaluation_results.csv', 
                       help='Output CSV file')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print(f"Starting evaluation on {world_size} GPU(s)")
        print(f"Config: {args.cfg_path}")
        print(f"Test data: {args.test_data}")
    
    # Load config
    from omegaconf import OmegaConf
    cfg_dict = OmegaConf.load(args.cfg_path)
    
    # Create a simple namespace to hold config
    class SimpleNamespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    cfg = SimpleNamespace(**cfg_dict)
    
    # Create model
    if rank == 0:
        print("Loading model...")
    
    model = SALMONN(
        llama_path=cfg.model.llama_path,
        whisper_path=cfg.model.whisper_path,
        beats_path=cfg.model.get('beats_path', ''),
        freeze_whisper=cfg.model.get('freeze_whisper', True),
        freeze_beats=cfg.model.get('freeze_beats', True),
        lora=cfg.model.get('lora', True),
        lora_rank=cfg.model.get('lora_rank', 8),
        lora_alpha=cfg.model.get('lora_alpha', 32),
        prompt_path=cfg.model.prompt_path,
        max_txt_len=cfg.model.get('max_txt_len', 128),
    )
    
    # Auto-detect checkpoint if not specified
    ckpt_path = cfg.model.get('ckpt', '')
    if not ckpt_path or ckpt_path == "":
        output_dir_path = cfg.model.get('output_dir', './output_antispoofing')
        checkpoint_strategy = cfg.run.get('checkpoint_strategy', 'best_acc')
        from pathlib import Path as PathLib
        output_path = PathLib(output_dir_path)
        if output_path.exists():
            # Find latest training run (directories starting with 202)
            training_runs = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('202')]
            if training_runs:
                latest_run = max(training_runs, key=lambda d: d.name)
                
                # Determine which checkpoint to use based on strategy
                if checkpoint_strategy == 'last':
                    checkpoint_name = 'checkpoint_last.pth'
                else:  # 'best_acc' or 'best_loss'
                    checkpoint_name = 'checkpoint_best.pth'
                
                checkpoint_path = latest_run / checkpoint_name
                if checkpoint_path.exists():
                    ckpt_path = str(checkpoint_path)
                    if rank == 0:
                        print(f"Auto-detected checkpoint ({checkpoint_strategy}): {ckpt_path}")
                else:
                    if rank == 0:
                        print(f"Warning: No {checkpoint_name} found in {latest_run}")
                        # Fallback to checkpoint_best.pth if checkpoint_last.pth doesn't exist
                        if checkpoint_strategy == 'last':
                            fallback_path = latest_run / 'checkpoint_best.pth'
                            if fallback_path.exists():
                                ckpt_path = str(fallback_path)
                                print(f"Falling back to: {ckpt_path}")
            else:
                if rank == 0:
                    print(f"Warning: No training runs found in {output_dir_path}")
    
    # Load checkpoint if specified
    if ckpt_path:
        if rank == 0:
            print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
        if rank == 0:
            print("✓ Checkpoint loaded successfully")
    
    model = model.to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Create dataset
    if rank == 0:
        print("Loading dataset...")
    
    # Get max_test_samples from config if available
    max_test_samples = None
    if hasattr(cfg, 'datasets') and hasattr(cfg.datasets, 'max_test_samples'):
        max_test_samples = cfg.datasets.max_test_samples
    
    dataset = SALMONNDataset(
        ann_path=args.test_data,
        whisper_path=cfg.model.whisper_path,
        max_samples=max_test_samples,
        seed=42
    )
    
    # Create sampler and dataloader
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=dataset.collater,
        shuffle=False,
    )
    
    # Load test prompts and apply template (like Runner does)
    test_prompt_dict = None
    if hasattr(cfg.model, 'test_prompt_path') and cfg.model.test_prompt_path:
        with open(cfg.model.test_prompt_path, 'r') as f:
            test_prompt_dict = json.load(f)
        # Apply prompt_template to prompts (critical!)
        prompt_template = cfg.model.get('prompt_template', 'USER: {}\nASSISTANT:')
        for k in test_prompt_dict.keys():
            if isinstance(test_prompt_dict[k], list):
                test_prompt_dict[k] = [prompt_template.format(p) for p in test_prompt_dict[k]]
            else:
                test_prompt_dict[k] = prompt_template.format(test_prompt_dict[k])
        if rank == 0:
            print(f"✓ Loaded and formatted test prompts from: {cfg.model.test_prompt_path}")
    
    # Run evaluation
    if rank == 0:
        print(f"Evaluating on {len(dataset)} samples...")
    
    # Get output directory for incremental saves
    if args.output and args.output != 'results/evaluation_results.csv':
        output_dir = Path(args.output).parent
    else:
        # Use output_dir from config if available
        base_output_dir = cfg.model.get('output_dir', './output_antispoofing')
        from datetime import datetime
        output_dir = Path(base_output_dir) / f"eval_{datetime.now().strftime('%Y%m%d%H%M')}"
        args.output = str(output_dir / "results.csv")
    
    # Build generate_cfg from config file
    # OmegaConf DictConfig supports both attribute and dict access
    gen_config = cfg_dict.get('generate', {})
    temperature = float(gen_config.get('temperature', 1.0))
    do_sample = bool(gen_config.get('do_sample', False))
    
    # Validate temperature: must be > 0 when do_sample=True
    if do_sample and temperature <= 0:
        if rank == 0:
            print(f"Warning: temperature={temperature} must be > 0 when do_sample=True. Setting to 1.0")
        temperature = 1.0
    
    generate_cfg = {
        "max_new_tokens": int(gen_config.get('max_new_tokens', 50)),
        "num_beams": int(gen_config.get('num_beams', 1)),
        "do_sample": do_sample,
        "min_length": int(gen_config.get('min_length', 1)),
        "top_p": float(gen_config.get('top_p', 0.9)),
        "repetition_penalty": float(gen_config.get('repetition_penalty', 1.0)),
        "length_penalty": float(gen_config.get('length_penalty', 1.0)),
        "temperature": temperature,
    }
    
    predictions, labels, scores, texts = evaluate(model, dataloader, device, rank, world_size, output_dir=output_dir, prompt_dict=test_prompt_dict, generate_cfg=generate_cfg)
    
    # Compute and save metrics (only on rank 0)
    if rank == 0:
        print("\nComputing metrics...")
        
        # Check for unparseable predictions
        unknown_count = np.sum(predictions == -1)
        valid_predictions = predictions[predictions != -1]
        valid_labels = labels[predictions != -1]
        valid_scores = scores[predictions != -1]
        
        if unknown_count > 0:
            print(f"\n⚠️  WARNING: {unknown_count} predictions were unparseable (marked as 'unknown')")
            print(f"   These will be excluded from metric calculation")
            print(f"   Valid predictions: {len(valid_predictions)} / {len(predictions)}")
        
        metrics = compute_metrics(valid_labels, valid_predictions, valid_scores)
        
        # Save detailed outputs with model answers
        detailed_output_file = args.output.replace('.csv', '_detailed.json')
        # Ensure directory exists
        Path(detailed_output_file).parent.mkdir(parents=True, exist_ok=True)
        
        detailed_outputs = []
        for i in range(len(labels)):
            pred_str = 'unknown' if predictions[i] == -1 else ('bonafide' if predictions[i] == 1 else 'fake')
            detailed_outputs.append({
                'index': i,
                'true_label': 'bonafide' if labels[i] == 1 else 'fake',
                'predicted_label': pred_str,
                'score': float(scores[i]),
                'model_answer': texts[i],
                'correct': bool(labels[i] == predictions[i]) if predictions[i] != -1 else False
            })
        
        with open(detailed_output_file, 'w') as f:
            json.dump(detailed_outputs, f, indent=2)
        print(f"\n✓ Saved detailed outputs to: {detailed_output_file}")
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total samples: {len(labels)}")
        print(f"Valid predictions: {len(valid_predictions)} ({len(valid_predictions)/len(labels)*100:.1f}%)")
        print(f"Unparseable predictions: {unknown_count} ({unknown_count/len(labels)*100:.1f}%)")
        print(f"Bonafide samples: {np.sum(labels)} ({np.sum(labels)/len(labels)*100:.1f}%)")
        print(f"Fake samples: {len(labels) - np.sum(labels)} ({(len(labels)-np.sum(labels))/len(labels)*100:.1f}%)")
        print("\nMetrics:")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision:         {metrics['precision']:.4f}")
        print(f"  Recall:            {metrics['recall']:.4f}")
        print(f"  F1 Score:          {metrics['f1']:.4f}")
        print(f"  Specificity:       {metrics['specificity']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        
        if metrics.get('eer') is not None:
            print(f"\nAdvanced Metrics:")
            print(f"  EER:     {metrics['eer']:.4f}")
            print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        print("="*60)
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_path, index=False)
        print(f"\nMetrics saved to: {output_path}")
        
        # Save detailed results
        detailed_path = output_path.parent / f"{output_path.stem}_detailed.csv"
        try:
            # Ensure all arrays have the same length
            min_len = min(len(labels), len(predictions), len(scores), len(texts))
            results_df = pd.DataFrame({
                'true_label': ['bonafide' if l == 1 else 'fake' for l in labels[:min_len]],
                'predicted_label': ['bonafide' if p == 1 else 'fake' for p in predictions[:min_len]],
                'score': scores[:min_len],
                'model_output': texts[:min_len],
            })
            results_df.to_csv(detailed_path, index=False)
            print(f"Detailed results saved to: {detailed_path}")
        except Exception as e:
            print(f"Warning: Could not save detailed results: {e}")
            print(f"  labels: {len(labels)}, predictions: {len(predictions)}, scores: {len(scores)}, texts: {len(texts)}")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

