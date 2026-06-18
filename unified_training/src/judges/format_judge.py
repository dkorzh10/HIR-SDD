import re
from typing import Any, Dict, List, Set
from .base import Judge


def _parse_reasons(text: str) -> Set[str]:
    """Extract reasons set from <reasons>[...]</reasons> tag."""
    match = re.search(r'<reasons>\s*(\[.*?\])\s*</reasons>', str(text), re.DOTALL)
    if not match:
        return set()
    try:
        raw = match.group(1)
        parsed = eval(raw)
        if isinstance(parsed, list):
            result = set()
            for item in parsed:
                if isinstance(item, str):
                    if item.startswith('['):
                        inner = eval(item)
                        result.update(r.upper() for r in inner if isinstance(r, str))
                    else:
                        result.add(item.upper())
            return result
    except Exception:
        pass
    return set()


def _compute_correct_reasons_overlap(gt_reasons: Set[str], out_reasons: Set[str]) -> float:
    """
    Metric: (num_correct / len(gt_reasons)) * (num_correct / len(out_reasons)).
    Returns 0 if either set is empty.
    """
    if not gt_reasons or not out_reasons:
        return 0.0
    num_correct = len(gt_reasons & out_reasons)
    recall = num_correct / len(gt_reasons)
    precision = num_correct / len(out_reasons)
    return recall * precision


class FormatJudge(Judge):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        weights = config.get("weights", {})
        self.w_format = weights.get("format", 0.5)
        self.w_correctness = weights.get("correctness", 0.5)
        self.w_reasons_correctness = weights.get("reasons_correctness", 0.0)

    def score(self, inputs, outputs, gt, meta=None):
        results = []

        for out, g in zip(outputs, gt):
            format_ok = False
            is_correct = False
            correct_reasons_overlap = 0.0
            ans = None

            has_think = "<think>" in out and "</think>" in out
            has_reasons = "<reasons>" in out and "</reasons>" in out
            has_answer = "<answer>" in out and "</answer>" in out

            g_ans = None
            if has_think and has_reasons and has_answer:
                format_ok = True

                ans_match = re.search(r"<answer>(.*?)</answer>", out, re.DOTALL)
                g_ans_match = re.search(r"<answer>(.*?)</answer>", str(g), re.DOTALL)
                g_ans = g_ans_match.group(1).strip().lower() if g_ans_match else str(g).lower()
                if g_ans in ("real", "bonafide"):
                    g_ans = "real"
                elif g_ans in ("fake", "spoof"):
                    g_ans = "fake"

                if ans_match:
                    raw_ans = ans_match.group(1).strip().lower()
                    ans = "real" if raw_ans == "real" else ("fake" if raw_ans == "fake" else None)
                    if ans is not None and ans == g_ans:
                        is_correct = True

            # correct_reasons_overlap: only when gt has reasons (e.g. reasoning dataset, not hard-label)
            gt_reasons = _parse_reasons(str(g))
            if gt_reasons and self.w_reasons_correctness > 0:
                if format_ok:
                    out_reasons = _parse_reasons(out)
                    correct_reasons_overlap = _compute_correct_reasons_overlap(gt_reasons, out_reasons)
                # else: 0.0 (format wrong -> no overlap score)
            elif not gt_reasons and self.w_reasons_correctness > 0 and format_ok and is_correct:
                if g_ans == "real":
                    correct_reasons_overlap = 1.0

            score = 0.0
            score += self.w_format * (1.0 if format_ok else 0.0)
            score += self.w_correctness * (1.0 if is_correct else 0.0)
            score += self.w_reasons_correctness * correct_reasons_overlap

            results.append({
                "score": score,
                "format_ok": format_ok,
                "is_correct": is_correct,
                "correct_reasons_overlap": correct_reasons_overlap,
                "answer": ans,  # "real" | "fake" | None (for skeptic filtering)
            })

        return {
            "score": sum(r["score"] for r in results) / len(results),
            "meta": {"per_sample": results},
        }

