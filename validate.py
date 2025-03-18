from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.utils.data as data
from sacrebleu import corpus_bleu
from tqdm import tqdm

from beam_search import (
    beam_search_batched,
    beam_search_unbatched,
    greedy_search,
    top_k_sampling,
)
from constants import constants
from data_processing.vocab import load_vocab, output_to_text
from generate import generate_autoregressivly
from hyperparameters import hyperparameters

logger = logging.getLogger(__name__)


def validate(
    model: nn.Module,
    test_data: data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    save_hypotheses_to_file: bool = False,
    num_batches_to_validate_on: int | None = None,
) -> float:
    all_references: list[str] = []
    all_hypotheses: list[str] = []

    src_vocab = load_vocab(constants.file_paths.src_vocab)
    tgt_vocab = load_vocab(constants.file_paths.tgt_vocab)

    logger.debug("Started validating models")
    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(test_data), desc="Running validation", total=len(test_data)
        ):
            src_tokens, ground_truth = batch
            src_tokens, ground_truth = src_tokens.to(
                hyperparameters.device
            ), ground_truth.to(hyperparameters.device)
            output = generate_autoregressivly(
                model,
                src_tokens,
                ground_truth,
                beam_search_batched,
                tgt_vocab,
                print_ex=1,
            )
            all_hypotheses.extend(output)
            all_references.extend(
                [output_to_text(ref) for ref in ground_truth.tolist()]
            )

            if (
                num_batches_to_validate_on is not None
                and i + 1 >= num_batches_to_validate_on
            ):
                logger.info(
                    f"Only  validating on {num_batches_to_validate_on} batches, stopping"
                )
                break

    if save_hypotheses_to_file:
        logger.info("Saving hypotheses to file")
        with open("local/data/test/hypotheses.en", "w") as f:
            for hyp in all_hypotheses:
                f.write(hyp + "\n")

    bleu_score = corpus_bleu(all_hypotheses, [all_references]).score
    logger.info(f"Validation BLEU Score: {bleu_score}")
    return bleu_score
