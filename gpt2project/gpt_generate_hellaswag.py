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
    avg_probs = torch.zeros(n, len(choice_options)).to(hyperparameters.device)
    for i in range(n):
        logits, _ = model(full_prompts)
        logits = logits.view(len(choice_options), full_len, -1)
        most_likely_row, _, avg_prob = _get_most_likely_row(full_prompts, mask, logits)
        avg_probs[i] = avg_prob
        class_prediction_strs.append(str(most_likely_row))

    final_class_prediction = _majority_vote(class_prediction_strs)
    uqs = torch.tensor([_hellaSwag_UQ_selected_class_only(avg_probs, int(final_class_prediction))]).to(hyperparameters.device)

    return [final_class_prediction], uqs


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
) -> Tuple[int, torch.Tensor, torch.Tensor]:

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

    # Calculate average probability of the correct next token in the completion region
    probs = F.softmax(shift_logits, dim=-1)
    actual_token_probs = torch.gather(probs, 2, shift_tokens.unsqueeze(-1)).squeeze(-1)
    actual_token_log_probs = torch.log(actual_token_probs)
    masked_actual_token_log_probs = actual_token_log_probs * shift_mask
    sum_prob = masked_actual_token_log_probs.sum(dim=1)
    avg_prob = sum_prob / shift_mask.sum(dim=1)

    # TODO: remove cross entropy loss part

    assert int(most_likely_row) == avg_prob.argmax().item()

    return int(most_likely_row), avg_loss, avg_prob


def _majority_vote(class_predictions: List[str]) -> str:
    return Counter(class_predictions).most_common(1)[0][0]


def _hellaSwag_UQ(sequence_probabilities: torch.Tensor) -> torch.Tensor:
    # avg_token_probabilities is (num_inferences, class_size)
    softmax_probs = F.softmax(sequence_probabilities, dim=1)
    variance_of_probs = torch.var(softmax_probs, dim=0)  # (class_size)
    class_averaged_variance = torch.mean(variance_of_probs)  # (1)

    return class_averaged_variance  # (1)


def _hellaSwag_UQ_selected_class_only(
    sequence_probabilities: torch.Tensor, selected_class: int
) -> torch.Tensor:
    # avg_token_probabilities is (num_inferences, class_size)
    softmax_probs = F.softmax(sequence_probabilities, dim=1)
    variance_of_probs = torch.var(softmax_probs, dim=0)  # (class_size)
    output = variance_of_probs[selected_class]
    assert output.numel() == 1, "Should be a scalar"
    return output


def _hellaSwag_UQ_beam_score(
    sequence_probabilities: torch.Tensor, selected_class: int
) -> torch.Tensor:
    # avg_token_probabilities is (num_inferences, class_size)
    avg_token_probabilities_across_inferences = torch.mean(
        sequence_probabilities, dim=0
    )  # (class_size)
    output = avg_token_probabilities_across_inferences[selected_class]  # (1)
    assert output.numel() == 1, "Should be a scalar"
    return output


def _hellaSwag_BALD_UQ(sequence_log_probabilities: torch.Tensor) -> torch.Tensor:
    # Convert log probabilities to probabilities via softmax along candidate dimension.
    dropout_probs = F.softmax(sequence_log_probabilities, dim=1)  # shape: (T, num_choices)

    # Compute the predictive distribution as the average over the dropout samples.
    predictive_distribution = dropout_probs.mean(dim=0)  # shape: (num_choices)

    # Compute the entropy of the predictive distribution.
    predictive_entropy = -(
        predictive_distribution * (predictive_distribution + 1e-8).log()
    ).sum()

    # Compute the entropy for each dropout sample.
    sample_entropies = -(dropout_probs * (dropout_probs + 1e-8).log()).sum(
        dim=1
    )  # shape: (T,)
    expected_entropy = sample_entropies.mean()

    # BALD score: the mutual information between predictions and the model parameters.
    bald_score = predictive_entropy - expected_entropy
    return bald_score
