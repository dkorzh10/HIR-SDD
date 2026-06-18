"""
Text utilities for epoch outputs (e.g. stripping [PAD] for Salmon log files).
"""
import re
from typing import Any, List, Optional


def _get_salmon_pad_strings(model: Any) -> List[str]:
    """Get all string forms of the pad token from Salmon tokenizer (literal + decoded)."""
    strings_to_strip = ["[PAD]"]
    try:
        tok = getattr(model, "model", None) and getattr(model.model, "llama_tokenizer", None)
        if tok is not None and getattr(tok, "pad_token_id", None) is not None:
            pt = getattr(tok, "pad_token", None)
            if pt is not None:
                s = (pt if isinstance(pt, str) else str(pt)).strip()
                if s and s not in strings_to_strip:
                    strings_to_strip.append(s)
            decoded = tok.decode([tok.pad_token_id]).strip()
            if decoded and decoded not in strings_to_strip:
                strings_to_strip.append(decoded)
    except Exception:
        pass
    return strings_to_strip


def strip_pad_tokens_from_texts(texts: List[str], pad_strings: Optional[List[str]] = None) -> List[str]:
    """
    Remove pad token strings from decoded text (e.g. Salmon).
    Use only for display/logging; do not use for completion_ids, logits, or confidence calculations.
    """
    if pad_strings is None:
        pad_strings = ["[PAD]"]
    out = []
    for s in texts:
        if not isinstance(s, str):
            out.append(s)
            continue
        t = s
        for pad_str in pad_strings:
            if pad_str:
                t = t.replace(pad_str, " ")
        t = re.sub(r"\s+", " ", t).strip()
        out.append(t)
    return out


def texts_for_log(model: Any, texts: List[str]) -> List[str]:
    """
    Return texts with [PAD] stripped when model is Salmon, otherwise return texts unchanged.
    Use only for writing to log files / judge input; do not use for completion_ids, logits, or confidences.
    """
    if type(model).__name__ != "SalmonModel":
        return texts
    pad_strings = _get_salmon_pad_strings(model)
    return strip_pad_tokens_from_texts(texts, pad_strings=pad_strings)
