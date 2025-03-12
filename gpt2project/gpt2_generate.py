import torch
import tiktoken
import numpy as np
from tqdm import tqdm
from torch import nn
from typing import List, Tuple

from gpt2project.gpt2model import GPT
from gpt2project.search_methods_gpt import (
    AutoregressiveInferenceResultsGPT,
    GPT_search_method,
    topk_sampling_gpt,
)
from gpt2project.uq.gpt_aq_funcs import AcquisitionFunctionGPT
from gpt2project.utils.decode import decode_token_id_batch, decode_token_list
from hyperparameters import hyperparameters
from torch.functional import F

from uq.acquisition_func import BLEU_mean_output_batch

enc = tiktoken.get_encoding("gpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"


def generate_autoregressivly_gpt2(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    tgt_tokens: torch.Tensor,
    search_method: GPT_search_method,
    break_on_newline: bool,
    max_tokens: int = 32,
) -> AutoregressiveInferenceResultsGPT:
    model.eval()
    tgt_tokens = tgt_tokens.to(hyperparameters.device)
    vocab_size = tokenizer.n_vocab
    output = search_method(model, tgt_tokens, vocab_size, max_tokens, break_on_newline)
    return output


def generate_autoregressivly_gpt2_with_uq(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    tgt_tokens: torch.Tensor,
    search_method: GPT_search_method,
    break_on_newline: bool,
    aq_funcs: List[AcquisitionFunctionGPT],
    max_tokens: int = 32,
) -> Tuple[List[List[str]], torch.Tensor]:
    model.eval()
    tgt_tokens = tgt_tokens.to(hyperparameters.device)
    vocab_size = tokenizer.n_vocab
    batch_size = tgt_tokens.size(0)

    hypothesis: List[List[str]] = [[] for _ in range(batch_size)]
    inference_results: List[AutoregressiveInferenceResultsGPT] = []

    uqs = torch.empty(batch_size, len(aq_funcs)).to(hyperparameters.device)

    for n in range(hyperparameters.uq.num_inferences):
        output = search_method(
            model, tgt_tokens, vocab_size, max_tokens, break_on_newline
        )
        assert output.token_ids.shape == (batch_size, max_tokens)

        inference_results.append(output)
        for b in range(batch_size):
            hypothesis[b].append(
                decode_token_list(output.token_ids[b].tolist(), tokenizer)
            )

    output_hypothesis = []
    for i, aq_func in enumerate(aq_funcs):
        uq = aq_func(hypothesis, inference_results)
        uqs[:, i] = uq
        if aq_func.multiple_inference:
            hyp = BLEU_mean_output_batch(hypothesis)
        else:
            hyp = [hypothesis[b][0] for b in range(batch_size)]
        output_hypothesis.append(hyp)

    return output_hypothesis, uqs
