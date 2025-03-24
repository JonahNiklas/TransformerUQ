from __future__ import annotations
import torch
import tiktoken
from tqdm import tqdm
from torch import nn
from typing import List, Tuple
from gpt2project.bayesformer_gpt import BayesformerGPT
from gpt2project.data_processing.abstract_evaluation_dataset import (
    AbstractEvaluationDataset,
)
from gpt2project.gpt2model import GPT
from gpt2project.search_methods_gpt import (
    AutoregressiveInferenceResultsGPT,
    GPT_search_method,
)
from gpt2project.uq.evaluation_run_config import EvaluationRunConfig
from gpt2project.uq.gpt_aq_funcs import AcquisitionFunctionGPT
from gpt2project.utils.cache_function_return import cache_evaluation_run_return
from gpt2project.utils.decode import decode_token_list
from hyperparameters import hyperparameters

from uq.acquisition_func import BLEU_mean_output_batch

enc = tiktoken.get_encoding("gpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"


def generate_for_entire_dataset(
    model: GPT,
    dataset: AbstractEvaluationDataset,
    tokenizer: tiktoken.Encoding,
    search_method: GPT_search_method,
) -> List[str]:
    generated_texts: List[str] = []
    for example in tqdm(
        dataset, desc=f"Doing inference on {dataset.__class__.__name__}"
    ):
        prompt = example.prompt
        tokens = (
            torch.tensor(tokenizer.encode(prompt))
            .unsqueeze(0)
            .to(hyperparameters.device)
        )
        generated_token_ids = generate_autoregressivly_gpt2(
            model,
            tokenizer,
            tokens,
            search_method=search_method,
            break_on_newline=dataset.break_on_newline,
            only_first_word=dataset.only_first_word,
            max_tokens=dataset.max_tokens,
        )
        generated_texts.append(
            decode_token_list(generated_token_ids.token_ids[0].tolist(), tokenizer)
        )
    return generated_texts


def generate_autoregressivly_gpt2(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    tgt_tokens: torch.Tensor,
    search_method: GPT_search_method,
    break_on_newline: bool,
    only_first_word: bool = False,
    max_tokens: int = 32,
) -> AutoregressiveInferenceResultsGPT:
    tgt_tokens = tgt_tokens.to(hyperparameters.device)
    vocab_size = tokenizer.n_vocab
    output = search_method(
        model, tgt_tokens, vocab_size, max_tokens, break_on_newline, only_first_word
    )
    return output


@cache_evaluation_run_return()
def generate_with_uq_for_entire_dataset(
    evaluation_run_config: EvaluationRunConfig,
) -> Tuple[List[List[str]], torch.Tensor]:
    dataset = evaluation_run_config.dataset
    dataset_size = len(dataset)

    all_output_texts: List[List[str]] = [
        [] for _ in range(len(evaluation_run_config.aq_funcs))
    ]
    all_uqs: torch.Tensor = torch.empty((0, len(evaluation_run_config.aq_funcs))).to(
        hyperparameters.device
    )

    for i, dataset_example in enumerate(dataset):
        prompt = dataset_example.prompt
        assert isinstance(prompt, str)
        prompt_token_ids = (
            torch.tensor(evaluation_run_config.tokenizer.encode(prompt))
            .unsqueeze(0)
            .to(hyperparameters.device)
        )
        output_texts, uq = generate_autoregressivly_gpt2_with_uq(
            evaluation_run_config.model,
            evaluation_run_config.tokenizer,
            prompt_token_ids,
            evaluation_run_config.search_method,
            evaluation_run_config.enable_mcdo,
            break_on_newline=dataset.break_on_newline,
            only_first_word=dataset.only_first_word,
            max_tokens=dataset.max_tokens,
            aq_funcs=evaluation_run_config.aq_funcs,
        )
        for aq in range(len(evaluation_run_config.aq_funcs)):
            all_output_texts[aq].extend(output_texts[aq])
        all_uqs = torch.cat((all_uqs, uq), dim=0)

    return all_output_texts, all_uqs


def generate_autoregressivly_gpt2_with_uq(
    model: GPT | BayesformerGPT,
    tokenizer: tiktoken.Encoding,
    tgt_tokens: torch.Tensor,
    search_method: GPT_search_method,
    enable_mcdo: bool,
    aq_funcs: List[AcquisitionFunctionGPT],
    break_on_newline: bool,
    only_first_word: bool,
    max_tokens: int,
) -> Tuple[List[List[str]], torch.Tensor]:
    if enable_mcdo:
        _enable_test_time_dropout(model)
    tgt_tokens = tgt_tokens.to(hyperparameters.device)
    vocab_size = model.config.vocab_size
    batch_size = tgt_tokens.size(0)

    hypothesis: List[List[str]] = [[] for _ in range(batch_size)]
    inference_results: List[AutoregressiveInferenceResultsGPT] = []

    uqs = torch.empty(batch_size, len(aq_funcs)).to(hyperparameters.device)

    for n in range(hyperparameters.uq.num_inferences):
        output = search_method(
            model, tgt_tokens, vocab_size, max_tokens, break_on_newline, only_first_word
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
            hyp = BLEU_mean_output_batch(
                hypothesis, use_effective_order=only_first_word
            )
        else:
            hyp = [hypothesis[b][0] for b in range(batch_size)]
        output_hypothesis.append(hyp)

    return output_hypothesis, uqs


def _enable_test_time_dropout(model: nn.Module) -> None:
    assert any(
        isinstance(module, nn.Dropout) for module in model.modules()
    ), "No dropout layer found in model"
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
