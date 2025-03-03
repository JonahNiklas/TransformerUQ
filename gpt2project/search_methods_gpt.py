import torch
from torch import nn
from typing import Callable, Optional
from hyperparameters import hyperparameters
from beam_search import AutoregressiveInferenceResults

# TODO: fix this type hint
GPT_search_method = Callable[
    [nn.Module, torch.Tensor, int, int], "AutoregressiveInferenceResults"
]


def greedy_search_gpt(
    model: nn.Module,
    tgt_tokens: torch.Tensor,
    vocab_size: int,
    max_len: int,
) -> AutoregressiveInferenceResults:
    with torch.no_grad():
        total_len = tgt_tokens.size(1) + max_len
        batch_size = tgt_tokens.size(0)
        tgt_tokens = torch.cat(
            [
                tgt_tokens,
                torch.zeros(batch_size, max_len, dtype=torch.long).to(
                    hyperparameters.device
                ),
            ],
            dim=1,
        )
        softmax_probs = torch.zeros(batch_size, total_len, vocab_size).to(
            hyperparameters.device
        )

        for t in range(total_len - max_len, total_len):
            output, _ = model(tgt_tokens)
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
    max_len: int,
    k: int = 8,
    temperature: float = 0.3,
) -> AutoregressiveInferenceResults:
    with torch.no_grad():
        total_len = tgt_tokens.size(1) + max_len
        batch_size = tgt_tokens.size(0)
        assert tgt_tokens.shape == (batch_size, tgt_tokens.size(1))
        softmax_probs = torch.ones(batch_size, tgt_tokens.size(1), vocab_size).to(
            hyperparameters.device
        )

        sample_rng = torch.Generator(device=hyperparameters.device)
        sample_rng.manual_seed(40)

        for t in range(max_len):
            output, _ = model(tgt_tokens)
            assert output.shape == (batch_size, tgt_tokens.size(1), vocab_size)
            logits = output[:, -1, :]
            assert logits.shape == (batch_size, vocab_size)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            assert probs.shape == (batch_size, vocab_size)
            topk_probs, topk_tokens = torch.topk(probs, k=k, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
            predicted_tokens = torch.gather(topk_tokens, dim=-1, index=ix)
            assert predicted_tokens.shape == (batch_size, 1)
            tgt_tokens = torch.cat([tgt_tokens, predicted_tokens], dim=1)
            softmax_probs = torch.cat([softmax_probs, probs.unsqueeze(1)], dim=1)

        # TODO: Missing clean_inference_results see beam_search.py
    assert tgt_tokens.shape == (batch_size, total_len)
    assert softmax_probs.shape == (batch_size, total_len, vocab_size)
    return AutoregressiveInferenceResults(tgt_tokens, softmax_probs)