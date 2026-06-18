import torch
import re
import numpy as np


from src.util_classes.response_parser import ResponseParser

class RewardComputer:
    """Reward computation for GRPO training with LLM-as-a-Judge"""
    
    def __init__(self, device, validator, judge=None, config=None):
        """
        Initialize reward computer.
        
        Args:
            device: Torch device
            validator: Validator instance for parsing predictions
            judge: Judge instance (BaseJudge subclass) for LLM evaluation
            config: Dictionary containing reward configuration (mode, weights, etc.)
        """
        self.device = device
        self.validator = validator
        self.judge = judge
        
        # Load config with defaults
        self.config = config or {}
        self.mode = self.config.get("mode", "label_guided") # Default to label_guided
        self.judge_weight = self.config.get("judge_weight", 0.0)
        
        # Load appropriate prompts based on mode
        custom_prompts = self.config.get("judge_prompts", None)
        if custom_prompts:
            self.judge_prompts = custom_prompts
        elif self.mode == "reasoning_match":
            self.judge_prompts = self._default_reasoning_match_prompts()
        else:
            self.judge_prompts = self._default_label_guided_prompts()

    def _default_reasoning_match_prompts(self):
        """
        Regime 1: (gt_reasoning, pred_reasoning)
        Compares prediction against a ground truth reasoning text.
        """
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        
        return [
            {
                "name": "correctness",
                "template": (
                    f"{system_prompt} "
                    "USER: You are a strict judge. Rate how well the provided Answer matches the Reference conclusion. "
                    "Output ONLY a single integer from 1 to 10.\n\n"
                    "Reference: {gt_text}\n\n"
                    "Answer: {pred_text}\n\n"
                    "Score (1-10):\n"
                    "ASSISTANT: "
                ),
                "weight": 0.25
            },
            {
                "name": "reasoning_quality",
                "template": (
                    f"{system_prompt} "
                    "USER: You are a strict judge. Rate the reasoning quality. "
                    "Good answers provide detailed analysis matching the reference. "
                    "Output ONLY a single integer from 1 to 10.\n\n"
                    "Reference: {gt_text}\n\n"
                    "Answer: {pred_text}\n\n"
                    "Score (1-10):\n"
                    "ASSISTANT: "
                ),
                "weight": 0.25
            },
            {
                "name": "comprehensiveness",
                "template": (
                    f"{system_prompt} "
                    "USER: You are a strict judge. Did the Answer cover all the key points mentioned in the Reference? "
                    "Output ONLY a single integer from 1 to 10.\n\n"
                    "Reference: {gt_text}\n\n"
                    "Answer: {pred_text}\n\n"
                    "Score (1-10):\n"
                    "ASSISTANT: "
                ),
                "weight": 0.25
            },
            {
                "name": "logic",
                "template": (
                    f"{system_prompt} "
                    "USER: You are a strict judge. Assess the logic of the answer. "
                    "Output ONLY a single integer from 1 to 10.\n\n"
                    "Answer: {pred_text}\n\n"
                    "Score (1-10):\n"
                    "ASSISTANT: "
                ),
                "weight": 0.25
            }
        ]

    def _default_label_guided_prompts(self):
        """
        Regime 2: (gt_hard_label, pred_reasoning)
        Evaluates reasoning quality using a single unified prompt that outputs multiple metrics.
        """
        system_prompt = "You are an expert audio forensics analyst acting as a judge. You evaluate explanations for why an audio clip is real (bonafide) or deepfake (spoof)."
        
        strict_instruction = "Output scores in a structured format. Do not output any other text. Do not output explanation."
        
        # Few-shot examples to calibrate the judge
        few_shot_examples = (
            "Example 1:\n"
            "Answer: <think>electronic sound</think><reasons>['STRANGE_VOICE']</reasons><answer>spoof</answer>\n"
            "Detail: 2\nRelevance: 4\nLogic: 5\nHelpfulness: 3\n\n"
            "Example 2:\n"
            "Answer: <think>The voice sounds slightly robotic with unnatural pauses.</think><reasons>['STRANGE_VOICE']</reasons><answer>spoof</answer>\n"
            "Detail: 5\nRelevance: 8\nLogic: 7\nHelpfulness: 6\n\n"
            "Example 3:\n"
            "Answer: <think>The audio contains high-frequency artifacts around 16kHz and the breathing pattern is inconsistent with natural speech, suggesting a VC model.</think><reasons>['STRANGE_VOICE', 'ARTIFACTS']</reasons><answer>spoof</answer>\n"
            "Detail: 9\nRelevance: 10\nLogic: 9\nHelpfulness: 10\n\n"
        )

        return [
            {
                "name": "combined_metrics",
                "template": (
                    f"{system_prompt} "
                    "USER: Evaluate the provided analysis on four criteria:\n"
                    "1. Detail: Depth of observation (timestamps, artifacts, etc.)\n"
                    "2. Relevance: Focus on valid audio characteristics (not hallucinations)\n"
                    "3. Logic: Consistency between evidence and conclusion\n"
                    "4. Helpfulness: Clear structure and utility for the user\n\n"
                    f"{few_shot_examples}"
                    "Now evaluate this Answer:\n"
                    "Answer: {pred_text}\n\n"
                    "Output scores (1-10) for each aspect in the EXACT format (do not explain yourself, do not write any extra symbols):\n"
                    "Detail: X\nRelevance: Y\nLogic: Z\nHelpfulness: W\n\n"
                    f"{strict_instruction}\n"
                    "ASSISTANT: "
                ),
                "weight": 1.0 # This weight is just a placeholder; individual weights are applied in parsing
            }
        ]
    
    def parse_score(self, text):
        """
        Extract numeric score from LLM response.
        Handles: "8", "8/10", "Rating: 7", "I give it a 9", etc.
        Returns score in range [1, 10], defaults to 1 for unparseable responses.
        """
        # Clean the text
        text = text.strip().lower()
        
        if not text:
            return 1.0
            
        # Strategy 1: Exact match "Score: X" or "Rating: X"
        match = re.search(r'(?:score|rating|grade)\s*[:=]\s*(\d+)', text)
        if match:
            score = float(match.group(1))
            return max(1.0, min(score, 10.0))
            
        # Strategy 2: X/10 format
        match = re.search(r'(\d+)\s*/\s*10', text)
        if match:
            score = float(match.group(1))
            return max(1.0, min(score, 10.0))

        # Strategy 3: Find all numbers, prefer the one that is isolated
        # This regex looks for numbers 1-10 that are their own words
        numbers = re.findall(r'\\b([1-9]|10)\\b', text)
        if numbers:
            # If multiple numbers, naive heuristic: take the last one as it's often the conclusion
            return float(numbers[-1])
            
        # Try to find a number 1-10 at the start of the response (fallback)
        match = re.match(r'^(\d+)', text)
        if match:
            score = float(match.group(1))
            return max(1.0, min(score, 10.0))
        
        # Default to 1 (worst score) for unparseable/gibberish responses
        return 1.0

    def parse_combined_scores(self, text):
        """
        Parse multiple scores from a single combined response.
        Expected format:
        Detail: 8
        Relevance: 7
        Logic: 6
        Helpfulness: 9
        """
        scores = {
            "detail": 1.0,
            "relevance": 1.0,
            "logic": 1.0,
            "helpfulness": 1.0
        }
        
        if not text:
            return scores
            
        text_lower = text.lower()
        
        # Regex to find "Aspect: Score" pattern
        patterns = {
            "detail": r"detail\s*[:=]\s*(\d+)",
            "relevance": r"relevance\s*[:=]\s*(\d+)",
            "logic": r"logic\s*[:=]\s*(\d+)",
            "helpfulness": r"helpfulness\s*[:=]\s*(\d+)"
        }
        
        for aspect, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                val = float(match.group(1))
                scores[aspect] = max(1.0, min(val, 10.0))
                
        return scores

    def compute_llm_judge_scores(self, pred_texts, gt_texts, return_responses=False):
        """
        Use LLM as judge to evaluate text quality on multiple aspects.
        
        Args:
            pred_texts: List of predicted texts
            gt_texts: List of ground truth texts
            return_responses: If True, return raw LLM responses for logging
        
        Returns:
            scores: Tensor of normalized scores OR dict of scores per aspect if using internal gating
            responses: (optional) List of dicts with responses per aspect
        """
        if self.judge is None or self.judge_weight == 0:
            scores = torch.zeros(len(pred_texts), device=self.device)
            if return_responses:
                return scores, [{}] * len(pred_texts)
            return scores
        
        # Store scores per aspect for gating logic
        # Shape: {aspect_name: [score_0, score_1, ...]}
        aspect_scores = {}
        all_responses = [{} for _ in range(len(pred_texts))] if return_responses else None
        
        # Check if we are using the combined prompt
        is_combined = (len(self.judge_prompts) == 1 and self.judge_prompts[0]["name"] == "combined_metrics")
        
        if is_combined:
            prompt_config = self.judge_prompts[0]
            eval_prompts = []
            for pred, gt in zip(pred_texts, gt_texts):
                prompt = prompt_config["template"].format(
                    pred_text=pred,
                    gt_text=gt
                )
                eval_prompts.append(prompt)
            
            with torch.no_grad():
                responses = self.judge.generate_text_only(
                    eval_prompts,
                    generate_cfg={
                        "max_new_tokens": 100,
                        "do_sample": True,
                        "temperature": 0.4,
                    }  # Need more tokens for multi-line output
                )
            
            # Initialize lists for each aspect
            aspect_scores = {
                "detail": [],
                "relevance": [],
                "logic": [],
                "helpfulness": []
            }
            
            for i, resp in enumerate(responses):
                parsed = self.parse_combined_scores(resp)
                for k, v in parsed.items():
                    aspect_scores[k].append(v)
                
                if return_responses:
                    all_responses[i]["combined_metrics"] = {
                        "response": resp,
                        "parsed_score": parsed
                    }
                    
        else:
            # Original logic for separate prompts
            for prompt_config in self.judge_prompts:
                eval_prompts = []
                for pred, gt in zip(pred_texts, gt_texts):
                    prompt = prompt_config["template"].format(
                        pred_text=pred,
                        gt_text=gt
                    )
                    eval_prompts.append(prompt)
                
                with torch.no_grad():
                    responses = self.judge.generate_text_only(
                        eval_prompts,
                        generate_cfg={
                            "max_new_tokens": 50,
                            "do_sample": True,
                            "temperature": 0.4,
                        }  # Judge only needs to output a number 1-10
                    )
                
                scores = [self.parse_score(resp) for resp in responses]
                
                aspect_name = prompt_config["name"]
                aspect_scores[aspect_name] = scores
                
                if return_responses:
                    for i, (resp, score) in enumerate(zip(responses, scores)):
                        all_responses[i][aspect_name] = {
                            "response": resp,
                            "parsed_score": score
                        }
        
        # Return the raw aspect scores so compute_rewards can gate them
        # compute_rewards will handle the weighting and aggregation
        if return_responses:
            return aspect_scores, all_responses
        return aspect_scores

    def compute_rewards(self, completion, samples, return_details=False):
        """
        Compute rewards based on the selected mode.
        """
        if self.mode == "reasoning_match":
            return self._compute_reasoning_match_rewards(completion, samples, return_details)
        else:
            return self._compute_label_guided_rewards(completion, samples, return_details)

    def _compute_label_guided_rewards(self, completion, samples, return_details=False):
        """
        Regime 2 Implementation:
        1. Hard Correctness (Binary): pred_label == gt_label
        2. Hard Format (Binary): valid XML tags
        3. LLM Judge (Continuous): Quality of reasoning (detail, logic, etc.)
        """
        rewards = []
        details = [] if return_details else None
        
        # Weights from config or defaults
        weights = self.config.get("weights", {})
        w_correctness = weights.get("correctness", 1.0)
        w_format = weights.get("format", 0.1)
        w_judge = self.judge_weight


        # 1. Compute Judge Scores (if enabled)
        # Note: These prompts are reference-free, so gt_text is unused by the template
        # but passed for compatibility.
        if w_judge > 0:
            aspect_scores, judge_responses = self.compute_llm_judge_scores(
                completion, samples['text'], return_responses=return_details
            )
        else:
            aspect_scores = {}
            judge_responses = [None] * len(completion) if return_details else None

        # Pre-calculate aspect weights
        if len(self.judge_prompts) == 1 and self.judge_prompts[0]["name"] == "combined_metrics":
            # Manually set weights for the combined aspects
            aspect_weights = {
                "detail": 0.25,
                "relevance": 0.25,
                "logic": 0.25,
                "helpfulness": 0.25
            }
        else:
            aspect_weights = {p["name"]: p["weight"] for p in self.judge_prompts}
            
        total_judge_weight_norm = sum(aspect_weights.values()) or 1.0

        for i in range(len(completion)):
            # A. Hard Logic: Correctness
            pred = ResponseParser.parse_prediction(completion[i])
            gt = ResponseParser.parse_prediction(samples['text'][i])
            
            is_correct = 1.0 if (pred == gt and pred is not None) else 0.0
            
            # B. Hard Logic: Format & Structure
            text = completion[i]
            has_think = '<think>' in text and '</think>' in text
            has_reasons = '<reasons>' in text and '</reasons>' in text
            has_answer = '<answer>' in text and '</answer>' in text
            
            is_format_valid = 0.33 if (has_answer) else 0.0
            is_format_valid = 0.66 if (has_think and has_answer) else is_format_valid
            is_format_valid = 1.0 if (has_think and has_reasons and has_answer) else is_format_valid

            # D. LLM Judge Score Aggregation
            judge_score = 0.0
            current_aspect_scores = {}
            
            if w_judge > 0:
                for aspect, scores in aspect_scores.items():
                    score = scores[i]
                    
                    # Logic Gating: If format is broken, reasoning logic score is penalized
                    if not has_think and aspect in ['logic', 'detail']:
                        score = score * 0.5
                        
                    current_aspect_scores[aspect] = score
                    judge_score += score * aspect_weights.get(aspect, 0.0)
                
                judge_score = judge_score / (total_judge_weight_norm * 10.0) # Normalize to [0,1]

            # E. Final Reward Combination
            reward = (
                is_correct * w_correctness +
                is_format_valid * w_format +
                judge_score * w_judge
            )
            
            # Small noise
            reward += np.random.uniform(-0.01, 0.01)
            rewards.append(reward)

            if return_details:
                detail = {
                    "audio_id": samples.get('id', [None]*len(completion))[i],
                    "ground_truth_label": gt,
                    "model_answer": completion[i],        # Keep old key for compatibility
                    "model_prediction": pred,
                    "is_correct": is_correct,
                    "judge_score": judge_score,
                    "llm_judge_responses": judge_responses[i] if judge_responses else {},
                    "overall_reward": reward,
                    "format_valid": is_format_valid,
                }
                details.append(detail)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        return (rewards_tensor, details) if return_details else rewards_tensor

    def _compute_reasoning_match_rewards(self, completion, samples, return_details=False):
        """
        Regime 1 Implementation:
        Uses the original logic where we compare against a full text reference.
        Includes hard check for reasoning tags similarity.
        """
        rewards = []
        details = [] if return_details else None
        
        if self.judge_weight > 0:
            aspect_scores, judge_responses = self.compute_llm_judge_scores(completion, samples['text'], return_responses=return_details)
        else:
            aspect_scores = {}
            judge_responses = [None] * len(completion) if return_details else None
        
        # Map aspect names to weights for easy access
        aspect_weights = {p["name"]: p["weight"] for p in self.judge_prompts}
        total_weight = sum(aspect_weights.values())
        
        for i in range(len(completion)):
            # Use ResponseParser via static methods for parsing
            pred = ResponseParser.parse_prediction(completion[i])
            gt = ResponseParser.parse_prediction(samples['text'][i])
            
            # Parse reasons list
            pred_reasons = ResponseParser.parse_reasons(completion[i])
            gt_reasons = ResponseParser.parse_reasons(samples['text'][i])
            
            # Compute reasons score: (TP / GT) - 0.5 * (FP / GT)
            # Normalize to lower case for comparison
            pred_reasons_set = set(r.lower().strip() for r in pred_reasons)
            gt_reasons_set = set(r.lower().strip() for r in gt_reasons)
            
            if not gt_reasons_set:
                reasons_score = 1.0 if not pred_reasons_set else 0.0
            else:
                # True Positives: Predicted reasons that are in GT
                tp = len(pred_reasons_set.intersection(gt_reasons_set))
                # False Positives: Predicted reasons that are NOT in GT
                fp = len(pred_reasons_set - gt_reasons_set)
                
                num_gt = len(gt_reasons_set)
                
                # Formula: (TP / GT) - 0.5 * (FP / GT)
                reasons_score = (tp / num_gt) - 0.5 * (fp / num_gt)
                
                # Optional: Clamp to avoid extremely negative rewards if desired, 
                # or keep as is for strong negative signal.
                # User requested this specific formula, so we keep it raw but maybe bounded at -1.0?
                # Usually rewards are preferred to be somewhat bounded.
                # Let's keep it raw as requested, but bounded at -1.0 to prevent explosion.
                reasons_score = max(-1.0, reasons_score)
            
            is_correct = 1.0 if (pred == gt and pred is not None) else 0.0
            is_valid = 1.0 if pred is not None else 0.0
            
            # Calculate judge score with gating logic
            judge_score = 0.0
            
            if self.judge_weight > 0:
                current_scores = {}
                for aspect, scores in aspect_scores.items():
                    score = scores[i]
                    
                    # Gating: If binary correctness check failed, force correctness-related scores to min (1.0)
                    # Correctness and Comprehensiveness depend on getting the answer right
                    if is_correct == 0.0 and aspect in ['correctness', 'comprehensiveness']:
                        score = 1.0
                    
                    # Logic and Attempt to Reason can still get points even if answer is wrong
                    
                    # Length-based gating for reasoning: short answers cannot be good reasoning
                    # Soft penalty: scale score by length if < 50 chars
                    # This avoids the hard cliff where 49 chars = 1.0 score
                    answer_len = len(completion[i].strip())
                    
                    # Apply length penalty only to reasoning aspects
                    if aspect in ['reasoning_quality', 'logic', 'attempt_to_reason']:
                        if answer_len < 50:
                            penalty_factor = max(0.2, answer_len / 50.0)
                            score = score * penalty_factor
                    
                    current_scores[aspect] = score
                    judge_score += score * aspect_weights[aspect]
                
                # Normalize to [0, 1]
                judge_score = judge_score / (total_weight * 10.0)
            
            # Structure bonus: reward explicit XML tags
            is_format_valid = 0.0
            text = completion[i]
            has_think = '<think>' in text and '</think>' in text
            has_reasons = '<reasons>' in text and '</reasons>' in text
            has_answer = '<answer>' in text and '</answer>' in text
            
            is_format_valid = 0.33 if (has_answer) else 0.0
            is_format_valid = 0.66 if (has_think and has_answer) else is_format_valid
            is_format_valid = 1.0 if (has_think and has_reasons and has_answer) else is_format_valid

            correctness_weight = 0.3
            validity_weight = 0.1
            w_format = 0.1
            w_reasons = 0.2  # Weight for reasons match
            
            reward = (is_correct * correctness_weight + 
                     is_valid * validity_weight + 
                     is_format_valid * w_format +
                     reasons_score * w_reasons +
                     judge_score * self.judge_weight)
            
            # Add small random noise to prevent identical rewards for same outputs
            noise = np.random.uniform(-0.025, 0.025)
            reward = reward + noise
            
            rewards.append(reward)
            
            if return_details:
                audio_id = samples.get('id', [None] * len(completion))[i]
                gt_text = samples['text'][i]
                
                # Update logs to show gated scores if changed
                if judge_responses[i]:
                    for aspect, score in current_scores.items():
                        if aspect in judge_responses[i]:
                            judge_responses[i][aspect]['gated_score'] = score
                
                detail = {
                    "audio_id": audio_id,
                    "ground_truth": gt_text,
                    "model_answer": completion[i],
                    "llm_judge_responses": judge_responses[i] if judge_responses else {},
                    "judge_score": judge_score,
                    "validity_score": is_valid,
                    "correctness_score": is_correct,
                    "format_valid": is_format_valid,
                    "reasons_score": reasons_score,
                    "overall_reward": reward
                }
                details.append(detail)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        if return_details:
            return rewards_tensor, details
        return rewards_tensor
