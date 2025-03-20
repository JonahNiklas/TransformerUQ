from dataclasses import dataclass
from tracemalloc import stop
import tiktoken
import torch
from torch import nn
from typing import Callable, List, Tuple
from gpt2project.constants import PADDING_TOKEN_ID
from hyperparameters import hyperparameters

GPT_search_method = Callable[
    [nn.Module, torch.Tensor, int, int, bool, bool], "AutoregressiveInferenceResultsGPT"
]


def greedy_search_gpt(
    model: nn.Module,
    tgt_tokens: torch.Tensor,
    vocab_size: int,
    max_generated_len: int,
    break_on_newline: bool,
    only_first_word: bool,
) -> "AutoregressiveInferenceResultsGPT":
    with torch.no_grad():
        prompt_len = tgt_tokens.size(1)
        total_len = prompt_len + max_generated_len
        batch_size = tgt_tokens.size(0)
        softmax_probs = torch.empty((batch_size, 0, vocab_size)).to(
            hyperparameters.device
        )

        assert tgt_tokens.shape == (batch_size, prompt_len)

        for t in range(max_generated_len):
            output, _ = model(tgt_tokens)
            assert output.shape == (batch_size, tgt_tokens.size(1), vocab_size)
            logits = output[:, -1, :]
            assert logits.shape == (batch_size, vocab_size)
            probs = torch.softmax(logits, dim=-1)
            softmax_probs = torch.cat([softmax_probs, probs.unsqueeze(1)], dim=1)
            predicted_tokens = torch.argmax(probs, dim=-1)
            assert predicted_tokens.shape == (batch_size,)
            tgt_tokens = torch.cat([tgt_tokens, predicted_tokens.unsqueeze(1)], dim=1)

    assert tgt_tokens.shape == (batch_size, total_len)
    assert softmax_probs.shape == (batch_size, max_generated_len, vocab_size)

    tgt_tokens = tgt_tokens[:, prompt_len:]
    assert tgt_tokens.shape == (batch_size, max_generated_len)

    tgt_tokens, softmax_probs = _clean_inference_results(
        tgt_tokens, softmax_probs, break_on_newline, only_first_word
    )

    return AutoregressiveInferenceResultsGPT(tgt_tokens, softmax_probs)


def topk_sampling_gpt(
    model: nn.Module,
    tgt_tokens: torch.Tensor,
    vocab_size: int,
    max_generated_len: int,
    break_on_newline: bool,
    only_first_word: bool,
    k: int = 10,
    temperature: float = 0.5,
) -> "AutoregressiveInferenceResultsGPT":
    with torch.no_grad():
        prompt_len = tgt_tokens.size(1)
        total_len = prompt_len + max_generated_len
        batch_size = tgt_tokens.size(0)
        softmax_probs = torch.empty((batch_size, 0, vocab_size)).to(
            hyperparameters.device
        )

        assert tgt_tokens.shape == (batch_size, prompt_len)

        for t in range(max_generated_len):
            output, _ = model(tgt_tokens)
            assert output.shape == (batch_size, tgt_tokens.size(1), vocab_size)
            logits = output[:, -1, :]
            assert logits.shape == (batch_size, vocab_size)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            assert probs.shape == (batch_size, vocab_size)
            topk_probs, topk_tokens = torch.topk(probs, k=k, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1)
            predicted_tokens = torch.gather(topk_tokens, dim=-1, index=ix)
            assert predicted_tokens.shape == (batch_size, 1)
            tgt_tokens = torch.cat([tgt_tokens, predicted_tokens], dim=1)
            softmax_probs = torch.cat([softmax_probs, probs.unsqueeze(1)], dim=1)

    assert tgt_tokens.shape == (batch_size, total_len)
    assert softmax_probs.shape == (batch_size, max_generated_len, vocab_size)

    tgt_tokens = tgt_tokens[:, prompt_len:]
    assert tgt_tokens.shape == (batch_size, max_generated_len)
    tgt_tokens, softmax_probs = _clean_inference_results(
        tgt_tokens, softmax_probs, break_on_newline, only_first_word
    )

    return AutoregressiveInferenceResultsGPT(tgt_tokens, softmax_probs)


tokenizer = tiktoken.get_encoding("gpt2")


def _clean_inference_results(
    tgt_tokens: torch.Tensor,
    softmax_probs: torch.Tensor,
    break_on_newline: bool,
    only_first_word: bool,
    padding_token_id: int = PADDING_TOKEN_ID,
    eos_token_id: int = tokenizer.eot_token,
    newline_token_id: int = tokenizer.encode("\n")[0],
) -> Tuple[torch.Tensor, torch.Tensor]:
    stop_token_ids: List[int] = [eos_token_id] + (
        [newline_token_id] if break_on_newline else []
    )
    if only_first_word:
        stop_token_ids = stop_token_ids + tokenizer.encode(".")
        for i in range(tgt_tokens.size(0)):
            for j in range(1, tgt_tokens.size(1)):
                current_token_id = tgt_tokens[i, j]
                current_token = tokenizer.decode([int(current_token_id.item())])
                if " " in current_token or current_token_id in stop_token_ids:
                    tgt_tokens[i, j:] = padding_token_id
                    softmax_probs[i, j:, padding_token_id] = 1.0
                    break
        return tgt_tokens, softmax_probs

    for i in range(tgt_tokens.size(0)):
        for j in range(1, tgt_tokens.size(1)):
            if tgt_tokens[i, j] in stop_token_ids:
                tgt_tokens[i, j:] = padding_token_id
                softmax_probs[i, j:, padding_token_id] = 1.0
                break

    return tgt_tokens, softmax_probs


# used for type checking
_all_search_methods: List[GPT_search_method] = [
    greedy_search_gpt,
    topk_sampling_gpt,
]

@dataclass
class AutoregressiveInferenceResultsGPT:
    """
    Results of autoregressive inference (batch_size, max_len)
    """

    token_ids: torch.Tensor

    """
    Softmax probabilities for each token (batch_size, max_len, vocab_size)
    """
    softmax_probs: torch.Tensor

    def get_softmax_probs_for_selected_token(self) -> torch.Tensor:
        """
        Get the softmax probability for each token in token_ids by indexing into softmax_probs.
        For each batch and timestep, this returns the probability corresponding to the predicted token.

        Returns:
            A tensor of shape (batch_size, max_len) containing the probabilities for the selected tokens.
        """
        batch_size = self.token_ids.size(0)
        max_len = self.token_ids.size(1)
        token_ids_without_padding = torch.where(
            self.token_ids == PADDING_TOKEN_ID,
            0,
            self.token_ids,
        )
        assert token_ids_without_padding.shape == (batch_size, max_len)
        selected_probs = self.softmax_probs.gather(
            dim=2, index=token_ids_without_padding.unsqueeze(2)
        )
        assert selected_probs.shape == (batch_size, max_len, 1)
        selected_probs = selected_probs.squeeze(2)
        selected_probs = torch.where(
            self.token_ids == PADDING_TOKEN_ID,
            1.0,
            selected_probs,
        )
        assert selected_probs.shape == (batch_size, max_len)
        return selected_probs
