import torch
from torch import nn
from typing import Callable, Optional
from hyperparameters import hyperparameters
from beam_search import AutoregressiveInferenceResults

# TODO: fix this type hint
GPT_search_method = Callable[
    [nn.Module, torch.Tensor, int, int], 'AutoregressiveInferenceResults'
]

def greedy_search_gpt(
    model: nn.Module,
    tgt_tokens: torch.Tensor,
    vocab_size: int,
    max_len:int,
) -> AutoregressiveInferenceResults:
    with torch.no_grad():
        total_len = tgt_tokens.size(1) + max_len
        batch_size = tgt_tokens.size(0)
        tgt_tokens = torch.cat([tgt_tokens, torch.zeros(batch_size, max_len, dtype=torch.long).to(hyperparameters.device)], dim=1)
        softmax_probs = torch.zeros(batch_size, total_len, vocab_size).to(hyperparameters.device)

        for t in range(total_len-max_len,total_len):
            output,_ = model(tgt_tokens)
            assert output.shape == (batch_size, total_len, vocab_size)
            logits = output[:, t - 1, :]
            assert logits.shape == (batch_size, vocab_size)
            probs = torch.softmax(logits, dim=-1)
            assert probs.shape == (batch_size, vocab_size)
            softmax_probs[:, t, :] = probs
            predicted_tokens = torch.argmax(probs, dim=-1)
            tgt_tokens[:, t] = predicted_tokens
    assert tgt_tokens.shape == (batch_size, total_len)
    return AutoregressiveInferenceResults(tgt_tokens, softmax_probs)

def topk_sampling_gpt(
    model: nn.Module,
    tgt_tokens: torch.Tensor,
    vocab_size: int,
    max_len:int,
    k:int = 5,
) -> AutoregressiveInferenceResults:
    with torch.no_grad():
        total_len = tgt_tokens.size(1) + max_len
        batch_size = tgt_tokens.size(0)
        tgt_tokens = torch.cat([tgt_tokens, torch.zeros(batch_size, max_len, dtype=torch.long).to(hyperparameters.device)], dim=1)
        softmax_probs = torch.zeros(batch_size, total_len, vocab_size).to(hyperparameters.device)

        for t in range(total_len-max_len,total_len):
            output,_ = model(tgt_tokens)
            assert output.shape == (batch_size, total_len, vocab_size)
            logits = output[:, t - 1, :]
            assert logits.shape == (batch_size, vocab_size)
            probs = torch.softmax(logits, dim=-1)
            assert probs.shape == (batch_size, vocab_size)
            topk_probs, topk_tokens = torch.topk(probs, k=k, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            predicted_tokens = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)
            tgt_tokens[:, t] = predicted_tokens
            softmax_probs[:, t, :] = probs
    assert tgt_tokens.shape == (batch_size, total_len)
    return AutoregressiveInferenceResults(tgt_tokens, softmax_probs)