"""
Shared utilities for confidence extraction from model logits.

This module provides functions to extract Real/Fake probabilities from model logits,
supporting multi-token words (e.g., SALMON's "Fake" tokenized as ["F", "ake"]).
"""

import re
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F


def get_tokenizer(model):
    """
    Get tokenizer from model.
    
    Args:
        model: The unwrapped model instance
        
    Returns:
        Tokenizer instance or None if not found
    """
    if hasattr(model, 'processor'):
        return model.processor.tokenizer
    if hasattr(model, 'tokenizer'):
        return model.tokenizer
    if hasattr(model, 'model') and hasattr(model.model, 'llama_tokenizer'):
        return model.model.llama_tokenizer
    return None


def get_token_ids(tokenizer, cache: Optional[Dict] = None) -> tuple:
    """
    Get token ID lists for Real and Fake.
    
    Returns lists to support multi-token words (e.g., SALMON: Fake -> [F, ake]).
    Uses cache dictionary if provided to avoid repeated tokenization.
    
    Args:
        tokenizer: Tokenizer instance
        cache: Optional dict to cache token IDs (should have keys '_real_token_ids' and '_fake_token_ids')
        
    Returns:
        Tuple of (real_token_ids, fake_token_ids) as lists
    """
    if cache is not None:
        if '_real_token_ids' not in cache or cache['_real_token_ids'] is None:
            cache['_real_token_ids'] = tokenizer.encode("Real", add_special_tokens=False)
            cache['_fake_token_ids'] = tokenizer.encode("Fake", add_special_tokens=False)
        return cache['_real_token_ids'], cache['_fake_token_ids']
    else:
        real_ids = tokenizer.encode("Real", add_special_tokens=False)
        fake_ids = tokenizer.encode("Fake", add_special_tokens=False)
        return real_ids, fake_ids


def find_answer_position(text: str, tokenizer, token_ids: Optional[List[int]] = None, 
                         real_ids: Optional[List[int]] = None, fake_ids: Optional[List[int]] = None) -> int:
    """
    Find position of answer token in generated sequence.
    
    If token_ids provided, searches directly in token sequence (more reliable).
    Otherwise falls back to re-tokenizing the decoded text.
    
    Args:
        text: Decoded text string
        tokenizer: Tokenizer instance
        token_ids: Optional list of token IDs for the generated sequence
        real_ids: Optional pre-computed token IDs for "Real"
        fake_ids: Optional pre-computed token IDs for "Fake"
        
    Returns:
        Position index of the answer token
    """
    # If we have the actual token IDs, search for Real/Fake tokens directly
    if token_ids is not None and real_ids is not None and fake_ids is not None:
        # Search for Real or Fake token sequence in the generated tokens
        for i in range(len(token_ids) - max(len(real_ids), len(fake_ids)) + 1):
            # Check if Real tokens match at position i
            if token_ids[i:i+len(real_ids)] == real_ids:
                return i
            # Check if Fake tokens match at position i
            if token_ids[i:i+len(fake_ids)] == fake_ids:
                return i
        
        # Fallback: search backwards from end (answer is usually at the end)
        for i in range(len(token_ids) - 1, -1, -1):
            if i + len(real_ids) <= len(token_ids) and token_ids[i:i+len(real_ids)] == real_ids:
                return i
            if i + len(fake_ids) <= len(token_ids) and token_ids[i:i+len(fake_ids)] == fake_ids:
                return i
    
    # Fallback: re-tokenize decoded text (less reliable with PAD tokens)
    # Reasoning format: after <answer>
    match = re.search(r"<answer>", text)
    if match:
        prefix = text[:match.end()]
        return len(tokenizer.encode(prefix, add_special_tokens=False))
    
    # Hard-label format: after "Final Answer: "
    match = re.search(r"Final Answer:\s*", text)
    if match:
        prefix = text[:match.end()]
        return len(tokenizer.encode(prefix, add_special_tokens=False))
    
    return 0  # First token as fallback


def extract_confidences(pred_texts: List[str], logits: torch.Tensor, 
                       tokenizer, real_ids: List[int], fake_ids: List[int],
                       token_ids: Optional[torch.Tensor] = None,
                       extract_answer_fn=None) -> List[Dict[str, float]]:
    """
    Extract Real/Fake probabilities from logits.
    
    For multi-token words (e.g., SALMON 'Fake' -> F, ake), uses product of token probabilities
    in log-space for numerical stability.
    
    Args:
        pred_texts: List of decoded prediction texts
        logits: Logits tensor [batch_size, seq_len, vocab_size]
        tokenizer: Tokenizer instance
        real_ids: Token IDs for "Real"
        fake_ids: Token IDs for "Fake"
        token_ids: Optional token IDs tensor [batch_size, seq_len] for more reliable position finding
        extract_answer_fn: Optional function to extract answer from text (e.g., _extract_answer method)
    
    Returns:
        List of dicts with real_prob, fake_prob, and confidence (where confidence = P(predicted class) for EER)
    """
    if tokenizer is None or logits is None:
        return [{"real_prob": 0.0, "fake_prob": 0.0, "confidence": 0.5} for _ in pred_texts]
    
    confidences = []
    seq_len = logits.shape[1]
    vocab_size = logits.shape[2]

    def seq_logprob(ids: list, start: int, batch_idx: int) -> float:
        """Compute log-probability of token sequence for numerical stability."""
        if start + len(ids) > seq_len:
            return -float('inf')
        logp = 0.0
        for k, tid in enumerate(ids):
            pos = start + k
            if pos >= seq_len or tid >= vocab_size:
                return -float('inf')
            token_logits = logits[batch_idx, pos].float()
            logprobs = F.log_softmax(token_logits, dim=-1)
            logp += logprobs[tid].item()
        return logp

    for i, text in enumerate(pred_texts):
        # Use token IDs if available for more reliable position finding
        token_list = token_ids[i].tolist() if token_ids is not None else None
        answer_pos = find_answer_position(text, tokenizer, token_list, real_ids, fake_ids)

        real_prob = 0.0
        fake_prob = 0.0

        if answer_pos < seq_len:
            log_real = seq_logprob(real_ids, answer_pos, i)
            log_fake = seq_logprob(fake_ids, answer_pos, i)

            # Normalize probabilities to sum to 1 using logsumexp for numerical stability
            log_total = torch.logsumexp(torch.tensor([log_real, log_fake]), dim=0)
            real_prob = torch.exp(log_real - log_total).item()
            fake_prob = torch.exp(log_fake - log_total).item()
            
            # Confidence = P(predicted class) for proper EER computation
            # Determine which class was predicted by checking the text
            if extract_answer_fn is not None:
                pred_answer = extract_answer_fn(text).lower()
            else:
                # Simple fallback: check if "fake" or "real" appears in text
                text_lower = text.lower()
                if "fake" in text_lower:
                    pred_answer = "fake"
                elif "real" in text_lower:
                    pred_answer = "real"
                else:
                    pred_answer = ""
            
            if pred_answer == "fake":
                confidence = fake_prob
            elif pred_answer == "real":
                confidence = real_prob
            else:
                # If we can't determine prediction, use the higher probability
                confidence = max(real_prob, fake_prob)
        else:
            real_prob = fake_prob = 0.5
            confidence = 0.5

        confidences.append({
            "real_prob": real_prob,
            "fake_prob": fake_prob,
            "confidence": confidence
        })

    return confidences
