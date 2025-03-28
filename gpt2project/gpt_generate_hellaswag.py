from __future__ import annotations
from typing import List, Tuple
from tqdm import tqdm
import torch
import tiktoken
from gpt2project.bayesformer_gpt import BayesformerGPT
from gpt2project.data_processing.abstract_evaluation_dataset import DatasetExample
from gpt2project.data_processing.abstract_evaluation_dataset import (
    DatasetExampleMultipleChoice,
)
from gpt2project.gpt2_generate import _enable_test_time_dropout
from gpt2project.gpt2model import GPT
from gpt2project.uq.gpt_aq_funcs import AcquisitionFunctionGPT
from gpt2project.uq.evaluation_run_config import EvaluationRunConfig
from gpt2project.utils.cache_function_return import cache_evaluation_run_return
from hyperparameters import hyperparameters
from torch.nn import functional as F
from collections import Counter


@cache_evaluation_run_return()
def eval_with_uq_for_entire_hellaswag_dataset(
    evaluation_run_config: EvaluationRunConfig,
) -> Tuple[List[List[str]], torch.Tensor]:

    if evaluation_run_config.enable_mcdo:
        _enable_test_time_dropout(evaluation_run_config.model)

    dataset = evaluation_run_config.dataset
    dataset_size = len(dataset)
    aq_method_count = 1

    all_output_texts: List[List[str]] = [[] for _ in range(aq_method_count)]
    all_uqs: torch.Tensor = torch.empty((0, aq_method_count)).to(hyperparameters.device)

    for i, dataset_example in tqdm(
        enumerate(dataset),
        desc="Doing inference with UQ",
        total=(dataset_size),
    ):
        output_class, uqs = _eval_hellaswag_example(
            evaluation_run_config.model,
            evaluation_run_config.tokenizer,
            dataset_example,
            evaluation_run_config.aq_funcs,
        )

        for aq in range(aq_method_count):
            all_output_texts[aq].extend(output_class[aq])
        all_uqs = torch.cat((all_uqs, uqs.unsqueeze(0)), dim=0)

    return all_output_texts, all_uqs


def _eval_hellaswag_example(
    model: GPT | BayesformerGPT,
    tokenizer: tiktoken.Encoding,
    example: DatasetExample,
    aq_funcs: List[AcquisitionFunctionGPT],
) -> Tuple[List[str], torch.Tensor]:
    assert isinstance(
        example, DatasetExampleMultipleChoice
    ), "HellaSwags needs dataset of type DatasetExampleMultipleChoice"
    prompt, targets, choice_options = (
        example.prompt,
        example.targets,
        example.choice_options,
    )
    full_prompts, mask, full_len = _encode_full_prompt(
        tokenizer, prompt, choice_options
    )

    n = hyperparameters.uq.num_inferences
    class_prediction_strs = []
    avg_scores = torch.zeros(n, len(choice_options)).to(hyperparameters.device)
    for i in range(n):
        logits, _ = model(full_prompts)
        logits = logits.view(len(choice_options), full_len, -1)
        most_likely_row, avg_score = _get_most_likely_row(full_prompts, mask, logits)
        avg_scores[i] = avg_score
        class_prediction_strs.append(str(most_likely_row))

    uqs = torch.tensor([_hellaSwag_UQ(avg_scores)]).to(hyperparameters.device)
    final_class_prediction_str = [_majority_vote(class_prediction_strs)]

    return final_class_prediction_str, uqs


def _encode_full_prompt(
    tokenizer: tiktoken.Encoding, prompt: str, choice_options: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, int]:

    # encode prompt
    prompt_token_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).to(
        hyperparameters.device
    )

    # encode choice options. Add space because " word" is a different token than "word"
    choice_options_token_ids = [
        torch.tensor(tokenizer.encode(" " + choice), dtype=torch.long).to(
            hyperparameters.device
        )
        for choice in choice_options
    ]

    prompt_len = prompt_token_ids.size(-1)
    option_len = max([token_ids.size(-1) for token_ids in choice_options_token_ids])
    full_len = option_len + prompt_len
    full_prompts = torch.zeros(len(choice_options), full_len, dtype=torch.long).to(
        hyperparameters.device
    )

    mask = torch.zeros(len(choice_options), full_len, dtype=torch.long).to(
        hyperparameters.device
    )
    for i, choice_token_ids in enumerate(choice_options_token_ids):
        full_prompts[i, :prompt_len] = prompt_token_ids
        full_prompts[i, prompt_len : prompt_len + choice_token_ids.size(-1)] = (
            choice_token_ids
        )
        mask[i, prompt_len : prompt_len + choice_token_ids.size(-1)] = 1

    return full_prompts, mask, full_len


def _get_most_likely_row(
    tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor
) -> Tuple[int, torch.Tensor]:

    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (
        mask[..., 1:]
    ).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    most_likely_row = avg_loss.argmin().item()

    return int(most_likely_row), avg_loss


def _majority_vote(class_predictions: List[str]) -> str:
    return Counter(class_predictions).most_common(1)[0][0]


def _hellaSwag_UQ(avg_cross_entropy_score: torch.Tensor) -> torch.Tensor:
    softmax_probs_for_classes = torch.softmax(
        avg_cross_entropy_score, dim=-1
    )  # (num_inferences, class_size)
    class_averaged_softmax_probs = torch.mean(
        softmax_probs_for_classes, dim=0
    )  # (class_size)
    class_averaged_variance = torch.var(class_averaged_softmax_probs)  # (class_size)

    return torch.mean(class_averaged_variance, dim=0)  # (1)
