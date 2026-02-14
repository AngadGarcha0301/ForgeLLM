"""
Evaluation metrics for LLM fine-tuning.
"""

import math
from typing import List, Dict, Any, Optional
import torch
from collections import Counter


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.
    
    Perplexity = e^loss
    Lower is better.
    """
    return math.exp(loss)


def calculate_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4
) -> Dict[str, float]:
    """
    Calculate BLEU score.
    
    A simple implementation of BLEU for evaluation.
    """
    def get_ngrams(text: str, n: int) -> Counter:
        tokens = text.lower().split()
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    total_precision = []
    
    for n in range(1, max_n + 1):
        total_matches = 0
        total_count = 0
        
        for pred, ref in zip(predictions, references):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            
            for ngram, count in pred_ngrams.items():
                total_matches += min(count, ref_ngrams.get(ngram, 0))
                total_count += count
        
        if total_count > 0:
            precision = total_matches / total_count
        else:
            precision = 0
        
        total_precision.append(precision)
    
    # Calculate geometric mean
    if all(p > 0 for p in total_precision):
        bleu = math.exp(sum(math.log(p) for p in total_precision) / len(total_precision))
    else:
        bleu = 0
    
    return {
        "bleu": bleu,
        "bleu_1": total_precision[0] if len(total_precision) > 0 else 0,
        "bleu_2": total_precision[1] if len(total_precision) > 1 else 0,
        "bleu_3": total_precision[2] if len(total_precision) > 2 else 0,
        "bleu_4": total_precision[3] if len(total_precision) > 3 else 0,
    }


def calculate_rouge_l(prediction: str, reference: str) -> float:
    """
    Calculate ROUGE-L (Longest Common Subsequence) F1 score.
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # LCS length using dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(ref_tokens)
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    
    return f1


def calculate_rouge(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Calculate ROUGE scores for a batch.
    """
    rouge_l_scores = [
        calculate_rouge_l(pred, ref) 
        for pred, ref in zip(predictions, references)
    ]
    
    return {
        "rouge_l": sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0
    }


def calculate_exact_match(
    predictions: List[str],
    references: List[str]
) -> float:
    """
    Calculate exact match accuracy.
    """
    matches = sum(
        1 for pred, ref in zip(predictions, references)
        if pred.strip().lower() == ref.strip().lower()
    )
    return matches / len(predictions) if predictions else 0


def calculate_token_accuracy(
    predictions: List[str],
    references: List[str],
    tokenizer
) -> float:
    """
    Calculate token-level accuracy.
    """
    total_tokens = 0
    matching_tokens = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenizer.encode(pred)
        ref_tokens = tokenizer.encode(ref)
        
        min_len = min(len(pred_tokens), len(ref_tokens))
        total_tokens += len(ref_tokens)
        
        for i in range(min_len):
            if pred_tokens[i] == ref_tokens[i]:
                matching_tokens += 1
    
    return matching_tokens / total_tokens if total_tokens > 0 else 0
