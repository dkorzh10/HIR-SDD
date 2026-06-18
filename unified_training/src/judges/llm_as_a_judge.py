import re
import json
from abc import abstractmethod
from typing import Any, Dict, List, Optional
from .base import Judge

class LLMAsAJudge(Judge):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.w_correctness = config.get("weights", {}).get("correctness", 1.0)
        self.w_format = config.get("weights", {}).get("format", 0.1)
        self.w_judge = config.get("weights", {}).get("judge", 1.0)
        
        # Aspects weights (normalized internally)
        self.aspect_weights = config.get("aspect_weights", {
            "detail": 0.25,
            "relevance": 0.25,
            "logic": 0.25,
            "helpfulness": 0.25
        })
        
        # Load prompt template from config or default
        self.template = config.get("prompt_template", self._default_template())

    def _default_template(self) -> str:
        return (
            "You are an expert audio forensics analyst acting as a judge. "
            "Evaluate the provided analysis on four criteria:\n"
            "1. Detail: Depth of observation (timestamps, artifacts, etc.)\n"
            "2. Relevance: Focus on valid audio characteristics\n"
            "3. Logic: Consistency between evidence and conclusion\n"
            "4. Helpfulness: Clear structure and utility\n\n"
            "Answer: {output}\n\n"
            "Output scores (1-10) for each aspect in the EXACT format:\n"
            "Detail: X\nRelevance: Y\nLogic: Z\nHelpfulness: W\n"
        )

    @abstractmethod
    def _generate(self, prompts: List[str]) -> List[str]:
        """Subclasses implement this to call their respective LLMs."""
        pass

    def score(self, inputs: List[str], outputs: List[str], gt: List[str], meta: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        # 1. Hard Logic: Format & Correctness
        format_scores = []
        correctness_scores = []
        for out, g in zip(outputs, gt):
            # Format
            has_think = '<think>' in out and '</think>' in out
            has_reasons = '<reasons>' in out and '</reasons>' in out
            has_answer = '<answer>' in out and '</answer>' in out
            
            is_format_valid = 0.33 if has_answer else 0.0
            is_format_valid = 0.66 if (has_think and has_answer) else is_format_valid
            is_format_valid = 1.0 if (has_think and has_reasons and has_answer) else is_format_valid
            format_scores.append(is_format_valid)
            
            # Correctness
            ans_match = re.search(r"<answer>(.*?)</answer>", out, re.DOTALL)
            g_ans_match = re.search(r"<answer>(.*?)</answer>", str(g), re.DOTALL)
            
            ans = ans_match.group(1).strip().lower() if ans_match else None
            g_ans = g_ans_match.group(1).strip().lower() if g_ans_match else str(g).lower()
            
            is_correct = 1.0 if (ans and (ans == g_ans or (ans in ["real", "bonafide"] and g_ans in ["real", "bonafide"]) or (ans in ["fake", "spoof"] and g_ans in ["fake", "spoof"]))) else 0.0
            correctness_scores.append(is_correct)

        # 2. LLM Judge scores
        prompts = [self.template.format(output=out) for out in outputs]
        llm_responses = self._generate(prompts)
        
        llm_scores = []
        for resp in llm_responses:
            parsed = self._parse_combined_scores(resp)
            # Aggregate aspects
            weighted_score = 0.0
            total_w = sum(self.aspect_weights.values())
            for aspect, val in parsed.items():
                weighted_score += (val / 10.0) * self.aspect_weights.get(aspect, 0.0)
            llm_scores.append(weighted_score / total_w if total_w > 0 else 0.0)

        # 3. Combine
        final_scores = []
        for c, f, l in zip(correctness_scores, format_scores, llm_scores):
            reward = (c * self.w_correctness + 
                      f * self.w_format + 
                      l * self.w_judge)
            final_scores.append(reward)
            
        avg_score = sum(final_scores) / len(final_scores) if final_scores else 0.0
        return {
            "score": avg_score, 
            "meta": {
                "raw_scores": final_scores,
                "correctness": correctness_scores,
                "format": format_scores,
                "llm_scores": llm_scores,
                "llm_responses": llm_responses
            }
        }

    def _parse_combined_scores(self, text: str) -> Dict[str, float]:
        scores = {"detail": 1.0, "relevance": 1.0, "logic": 1.0, "helpfulness": 1.0}
        text_lower = text.lower()
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


