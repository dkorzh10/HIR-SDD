import os
import json
import logging

import torch
import torch.nn.functional as F

from src.infrastructure.dist_utils import is_main_process
from src.infrastructure.logger import MetricLogger, SmoothedValue
from src.infrastructure.utils import prepare_sample


class GRPOTrainer:
    """GRPO (Group Relative Policy Optimization) training logic"""
    
    def __init__(self, model, config, optimizer, scheduler, scaler, train_loader, 
                 output_dir, log_writter, reward_computer, epsilon=1e-8, initial_state_dict=None, prompt_dict=None, scheduler_epoch_offset=0):
        self.model = model
        self._model = model.module if hasattr(model, "module") else model  # Unwrapped model for parameter access
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.train_loader = train_loader
        self.output_dir = output_dir
        self.log_writter = log_writter
        self.reward_computer = reward_computer
        self.epsilon = epsilon
        self.initial_state_dict = initial_state_dict
        self.prompt_dict = prompt_dict
        self.scheduler_epoch_offset = scheduler_epoch_offset
        
        self.use_amp = config.config.run.amp
        self.cuda_enabled = (config.config.run.device == "cuda")
        self.iters_per_epoch = config.config.run.iters_per_epoch
        
    def _generate_completions(self, samples, num_grpo_samples):
        """Generate multiple completions for each sample"""
        all_texts = []
        all_completion_ids = []
        
        # Read generation config from config file with defaults
        gen_config = self.config.config.get("generate", {})
        temperature = float(gen_config.get("temperature", 1.0))
        do_sample = bool(gen_config.get("do_sample", True))
        
        # Debug: log the actual values being used (once per trainer instance)
        if not hasattr(self, '_logged_gen_config'):
            logging.info(f"Generation config - temperature: {temperature}, do_sample: {do_sample}, "
                        f"max_new_tokens: {gen_config.get('max_new_tokens', 50)}")
            self._logged_gen_config = True
        
        # Validate temperature: must be > 0 when do_sample=True
        if do_sample and temperature <= 0:
            logging.warning(f"temperature={temperature} must be > 0 when do_sample=True. Setting to 1.0")
            temperature = 1.0
        
        generate_cfg = {
            "max_new_tokens": int(gen_config.get("max_new_tokens", 2000)),
            "num_beams": int(gen_config.get("num_beams", 1)),
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": float(gen_config.get("top_p", 0.9)),
            "min_length": int(gen_config.get("min_length", 1)),
        }

        # Prepare prompts for generation (same as validation)
        prompts = None
        if self.model.prompt_dict and self.prompt_dict:
            import random
            prompts = []
            for task in samples["task"]:
                prompt_val = self.prompt_dict[task]
                if isinstance(prompt_val, list):
                    prompts.append(random.choice(prompt_val))
                else:
                    prompts.append(prompt_val)
        
        # Switch to eval mode for generation to enable use_cache and avoid gradient checkpointing conflicts
        was_training = self.model.training
        self.model.eval()
        
        try:
            with torch.no_grad():  # Generation doesn't need gradients
                for i in range(num_grpo_samples):
                    text, completion_ids, _ = self.model.generate(
                        samples, generate_cfg, prompts=prompts, return_outputs=True
                    )
                    all_texts.extend(text)
                    all_completion_ids.append(completion_ids)
        finally:
            # Restore original mode
            if was_training:
                self.model.train()
                
        return all_texts, all_completion_ids

    def _process_completions(self, all_texts, all_completion_ids):
        """Pad completions, create masks, and clean text"""
        # Pad all completion_ids to the same length before concatenating
        max_len = max(ids.size(1) for ids in all_completion_ids)
        padded_completion_ids = []
        completion_masks = []  # Track which tokens are real vs padded
        
        # end_sym is usually "</s>", we need its ID. Assuming tokenizer handles it correctly.
        # For LLaMA, EOS ID is typically 2.
        eos_token_id = self.model.llama_tokenizer.eos_token_id
        
        for ids in all_completion_ids:
            seq_len = ids.size(1)
            # Create mask: 1 for real tokens, 0 for padding
            mask = torch.ones(ids.size(0), max_len, dtype=torch.float32, device=ids.device)
            
            if seq_len < max_len:
                # Pad with pad_token_id (typically 0 for LLaMA)
                padding = torch.zeros(ids.size(0), max_len - seq_len, dtype=ids.dtype, device=ids.device)
                ids = torch.cat([ids, padding], dim=1)
                # Mark padded positions in mask
                mask[:, seq_len:] = 0
            
            # Apply EOS masking logic to the mask tensor
            for i in range(ids.size(0)):
                eos_indices = (ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    first_eos_idx = eos_indices[0].item()
                    # Mask positions after the first EOS
                    if first_eos_idx + 1 < max_len:
                        mask[i, first_eos_idx + 1:] = 0
            
            padded_completion_ids.append(ids)
            completion_masks.append(mask)
        
        # Stack tensors
        all_completion_ids = torch.cat(padded_completion_ids, dim=0)  # (batch*num_samples, seq_len)
        completion_mask = torch.cat(completion_masks, dim=0)  # (batch*num_samples, seq_len)
        
        # Clean texts for reward computation (strip </s> and subsequent garbage)
        cleaned_texts = []
        for text in all_texts:
            # Truncate at first </s>
            if "</s>" in text:
                text = text.split("</s>")[0] + "</s>"
            cleaned_texts.append(text)
            
        return all_completion_ids, completion_mask, cleaned_texts

    def _compute_rewards_and_advantages(self, cleaned_texts, samples, num_grpo_samples, batch_size):
        """Compute rewards and normalize to advantages"""
        samples_repeated = {
            'text': samples['text'] * num_grpo_samples,
            'id': samples.get('id', [None] * batch_size) * num_grpo_samples
        }
        # Use cleaned texts for rewards
        rewards, reward_details = self.reward_computer.compute_rewards(cleaned_texts, samples_repeated, return_details=True)
        
        # Compute group-relative advantages
        rewards_grouped = rewards.view(batch_size, num_grpo_samples)
        mean_reward = rewards_grouped.mean(dim=1, keepdim=True)
        std_reward = rewards_grouped.std(dim=1, keepdim=True)
        
        # Add minimum std threshold to prevent near-zero advantages when all samples are identical
        min_std = 0.01  # Minimum std to ensure meaningful advantages
        std_reward = torch.maximum(std_reward, torch.tensor(min_std, device=std_reward.device))
        
        advantages = (rewards_grouped - mean_reward) / (std_reward + self.epsilon)
        advantages = advantages.view(-1)  # Flatten back
        
        return rewards, advantages, reward_details

    def _compute_losses(self, samples, all_completion_ids, completion_mask, rewards, advantages, num_grpo_samples):
        """Compute forward pass, KL divergence, and final losses"""
        # Repeat samples for each generation
        samples_repeated_full = {
            'spectrogram': samples['spectrogram'].repeat_interleave(num_grpo_samples, dim=0),
        }
        if samples.get('raw_wav') is not None:
            samples_repeated_full['raw_wav'] = samples['raw_wav'].repeat_interleave(num_grpo_samples, dim=0)
        if samples.get('padding_mask') is not None:
            samples_repeated_full['padding_mask'] = samples['padding_mask'].repeat_interleave(num_grpo_samples, dim=0)
        
        # Debug: Check all_completion_ids shape
        if not hasattr(self, '_logged_completion_shape'):
            logging.info(f"all_completion_ids shape: {all_completion_ids.shape}")
            logging.info(f"completion_mask shape: {completion_mask.shape}")
            logging.info(f"Sample completion_ids[0]: {all_completion_ids[0][:10] if all_completion_ids.size(1) > 0 else 'EMPTY'}")
            self._logged_completion_shape = True
        
        with torch.no_grad():
            ref_log_probs = self.get_reference_log_probs(samples_repeated_full, all_completion_ids)
        
        all_logits = self.model.compute_logits_for_completions(samples_repeated_full, all_completion_ids)
        
        log_probs = self.get_log_probs(all_logits, all_completion_ids)
        
        # Mask out padding tokens when computing KL divergence and policy loss
        masked_log_probs = log_probs * completion_mask
        masked_ref_log_probs = ref_log_probs * completion_mask
        
        # Compute per-token KL divergence
        per_token_kl = (masked_log_probs - masked_ref_log_probs) * completion_mask
        kl_div = per_token_kl.sum(dim=1)
        
        # Compute log probability ratio (current / reference) for policy gradient
        # This is the standard GRPO/PPO approach
        log_ratio = (masked_log_probs - masked_ref_log_probs).sum(dim=1)
        
        # Policy loss: -E[log_ratio * advantages]
        # Negative sign because we want to maximize expected advantage
        beta = self.config.config.run.get("kl_coef", 0.1)
        policy_loss = -(log_ratio * advantages).mean()
        kl_penalty = beta * kl_div.mean()
        loss = policy_loss + kl_penalty
        
        return loss, kl_div, policy_loss, kl_penalty, log_probs, ref_log_probs, log_ratio

    def _log_debug_info(self, log_probs, ref_log_probs, completion_mask, log_ratio, advantages, policy_loss, kl_penalty, loss, rewards):
        """Log debug information to file and console"""
        if not hasattr(self, '_logged_loss_debug'):
            # Compute means only over non-masked positions
            num_valid_tokens = completion_mask.sum().item()
            masked_sum_log_probs = (log_probs * completion_mask).sum().item()
            masked_sum_ref_log_probs = (ref_log_probs * completion_mask).sum().item()
            
            debug_data = {
                "iteration": "first",
                "log_ratio_mean": log_ratio.mean().item(),
                "log_ratio_std": log_ratio.std().item(),
                "log_ratio_min": log_ratio.min().item(),
                "log_ratio_max": log_ratio.max().item(),
                "advantages_mean": advantages.mean().item(),
                "advantages_std": advantages.std().item(),
                "log_probs_shape": list(log_probs.shape),
                "completion_mask_shape": list(completion_mask.shape),
                "num_valid_tokens": num_valid_tokens,
                "log_probs_mean_valid": masked_sum_log_probs / num_valid_tokens if num_valid_tokens > 0 else 0.0,
                "ref_log_probs_mean_valid": masked_sum_ref_log_probs / num_valid_tokens if num_valid_tokens > 0 else 0.0,
                "log_probs_has_nan": torch.isnan(log_probs).any().item(),
                "ref_log_probs_has_nan": torch.isnan(ref_log_probs).any().item(),
                "policy_loss": policy_loss.item(),
                "kl_penalty": kl_penalty.item(),
                "total_loss": loss.item(),
                "rewards_mean": rewards.mean().item(),
                "rewards_std": rewards.std().item()
            }
            
            # Write to debug log file
            if is_main_process():
                debug_log_file = os.path.join(self.output_dir, "grpo_debug.jsonl")
                with open(debug_log_file, "a", encoding='utf-8') as f:
                    f.write(json.dumps(debug_data, ensure_ascii=False) + "\n")
                
                # Also print to stdout for visibility
                print(f"\n{'='*60}")
                print(f"GRPO LOSS DEBUG - First Iteration")
                print(f"{'='*60}")
                print(f"log_ratio: mean={debug_data['log_ratio_mean']:.6f}, std={debug_data['log_ratio_std']:.6f}")
                print(f"advantages: mean={debug_data['advantages_mean']:.6f}, std={debug_data['advantages_std']:.6f}")
                print(f"policy_loss: {debug_data['policy_loss']:.6f}, kl_penalty: {debug_data['kl_penalty']:.6f}")
                print(f"total_loss: {debug_data['total_loss']:.6f}")
                print(f"Debug log saved to: {debug_log_file}")
                print(f"{'='*60}\n")
            
            self._logged_loss_debug = True

    def get_grpo_loss(self, samples):
        """Compute GRPO loss with group-relative advantages"""
        # Get num_grpo_samples from config, default to 4
        num_grpo_samples = int(self.config.config.run.get("num_grpo_samples", 4))
        batch_size = samples["spectrogram"].shape[0]
        
        # 1. Generate completions
        all_texts, all_completion_ids = self._generate_completions(samples, num_grpo_samples)
        
        # 2. Process completions (padding, masking, cleaning)
        all_completion_ids, completion_mask, cleaned_texts = self._process_completions(all_texts, all_completion_ids)
        
        # 3. Compute Rewards and Advantages
        rewards, advantages, reward_details = self._compute_rewards_and_advantages(
            cleaned_texts, samples, num_grpo_samples, batch_size
        )
        
        # 4. Compute Losses
        loss, kl_div, policy_loss, kl_penalty, log_probs, ref_log_probs, log_ratio = self._compute_losses(
            samples, all_completion_ids, completion_mask, rewards, advantages, num_grpo_samples
        )
        
        # 5. Log Debug Info (only once per training)
        self._log_debug_info(
            log_probs, ref_log_probs, completion_mask, log_ratio, advantages, 
            policy_loss, kl_penalty, loss, rewards
        )
        
        return loss, rewards.mean(), kl_div.mean(), policy_loss, kl_penalty, reward_details

    def get_log_probs(self, logits, labels):
        """
        Compute log softmax for selected tokens only.
        
        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len) - token IDs
        
        Returns:
            log_probs: (batch_size, seq_len) - log probabilities of selected tokens
        """
        # Compute log softmax over vocabulary
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for the actual tokens
        # labels shape: (batch_size, seq_len)
        # We need to select log_probs[i, j, labels[i, j]] for each i, j
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for gathering
        # log_probs: (batch_size * seq_len, vocab_size)
        log_probs_flat = log_probs.view(-1, vocab_size)
        
        # labels: (batch_size * seq_len)
        labels_flat = labels.view(-1)
        
        # Gather: (batch_size * seq_len)
        selected_log_probs = log_probs_flat.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
        
        # Reshape back: (batch_size, seq_len)
        selected_log_probs = selected_log_probs.view(batch_size, seq_len)
        
        return selected_log_probs
    
    def get_reference_log_probs(self, samples, completion_ids):
        """
        Compute log probabilities using the reference (initial) model weights.
        Temporarily swaps trainable parameters to initial state.
        
        Args:
            samples: Input samples with audio
            completion_ids: Generated token IDs
        
        Returns:
            ref_log_probs: Log probabilities from reference model
        """
        if self.initial_state_dict is None:
            logging.warning("No reference model available, using current model for KL computation")
            ref_logits = self.model.compute_logits_for_completions(samples, completion_ids)
            return self.get_log_probs(ref_logits, completion_ids)
        
        current_state = {}
        # Use unwrapped model to match parameter names with initial_state_dict
        for name, param in self._model.named_parameters():
            if name in self.initial_state_dict:
                current_state[name] = param.data.clone()
                param.data.copy_(self.initial_state_dict[name])
        
        # Debug: Log once to verify reference model is being used
        if not hasattr(self, '_logged_ref_model'):
            ref_model_data = {
                "parameters_swapped": len(current_state),
                "parameters_stored": len(self.initial_state_dict),
                "swap_successful": len(current_state) == len(self.initial_state_dict),
                "parameter_names": list(current_state.keys())[:5]  # First 5 names as sample
            }
            
            # Write to debug log file
            if is_main_process():
                debug_log_file = os.path.join(self.output_dir, "grpo_debug.jsonl")
                with open(debug_log_file, "a", encoding='utf-8') as f:
                    f.write(json.dumps({"reference_model_check": ref_model_data}, ensure_ascii=False) + "\n")
                
                # Also print to stdout
                print(f"\n>>> Reference model: Swapped {len(current_state)}/{len(self.initial_state_dict)} parameters")
                print(f">>> Swap successful: {ref_model_data['swap_successful']}")
                print(f">>> Debug log: {debug_log_file}\n")
            
            self._logged_ref_model = True
        
        ref_logits = self.model.compute_logits_for_completions(samples, completion_ids)
        ref_log_probs = self.get_log_probs(ref_logits, completion_ids)
        
        # Restore current model weights
        for name, param in self._model.named_parameters():
            if name in current_state:
                param.data.copy_(current_state[name])
        
        return ref_log_probs
        
    def grpo_step(self, samples):
        """Single GRPO training step"""
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss, avg_reward, kl_div, policy_loss, kl_penalty, reward_details = self.get_grpo_loss(samples)
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss, avg_reward, kl_div, policy_loss, kl_penalty, reward_details

    def train_epoch(self, epoch):
        self.model.train()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, self.iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        log_freq = self.config.config.run.log_freq

        for i in metric_logger.log_every(range(self.iters_per_epoch), log_freq, header=header, logger=self.log_writter, start_step=epoch*self.iters_per_epoch):
            if i >= self.iters_per_epoch:
                break

            samples = next(self.train_loader)
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)

            # Apply scheduler offset if any (for cases where we skip data but want fresh schedule)
            effective_epoch = epoch - self.scheduler_epoch_offset
            if effective_epoch < 0:
                effective_epoch = 0
                
            self.scheduler.step(cur_epoch=effective_epoch, cur_step=i)

            try:
                loss, avg_reward, kl_div, policy_loss, kl_penalty, reward_details = self.grpo_step(samples)

                if (i + 1) % self.config.config.run.accum_grad_iters == 0:
                    # Gradient clipping
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.config.run.get("max_grad_norm", 1.0))

                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
            except torch.OutOfMemoryError as e:
                logging.error(f"OOM caught in training loop at epoch {epoch} iter {i}. Skipping batch. Error: {e}")
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                continue

            metric_logger.update(loss=loss.item())
            metric_logger.update(reward=avg_reward.item())
            metric_logger.update(kl_div=kl_div.item())
            metric_logger.update(policy_loss=policy_loss.item())
            metric_logger.update(kl_penalty=kl_penalty.item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            
            # Console logging at log_freq intervals
            if is_main_process() and i % log_freq == 0:
                logging.info(
                    f"Epoch [{epoch}][{i}/{self.iters_per_epoch}] "
                    f"Loss: {loss.item():.4f} (policy: {policy_loss.item():.4f}, kl: {kl_penalty.item():.4f}) "
                    f"Reward: {avg_reward.item():.4f} KL-div: {kl_div.item():.4f}"
                )
            
            # Judge logs written EVERY iteration (not just at log_freq)
            if is_main_process():
                judge_log_dir = os.path.join(self.output_dir, "judge_logs")
                os.makedirs(judge_log_dir, exist_ok=True)
                judge_log_file = os.path.join(judge_log_dir, f"epoch_{epoch}_iter_{i}.jsonl")
                
                with open(judge_log_file, "w", encoding='utf-8') as f:
                    for detail in reward_details:
                        f.write(json.dumps(detail, ensure_ascii=False) + "\n")
            
            # Log to file every log_freq iterations
            if is_main_process() and (i % log_freq == 0 or i == self.iters_per_epoch - 1):
                log_data = {
                    "epoch": epoch,
                    "iteration": i,
                    "train_loss": loss.item(),
                    "train_reward": avg_reward.item(),
                    "train_kl_div": kl_div.item(),
                    "train_policy_loss": policy_loss.item(),
                    "train_kl_penalty": kl_penalty.item(),
                    "train_lr": self.optimizer.param_groups[0]["lr"]
                }
                with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_data) + "\n")
                    f.flush()

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }
