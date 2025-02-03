from __future__ import annotations

import logging
from typing import Tuple
import torch
import torch.nn as nn
import torch.utils.data as data
from sacrebleu import corpus_bleu
from tqdm import tqdm

from data_processing.vocab import output_to_text
from uq.generate_with_uq import generate_autoregressivly_with_uq
from uq.acquisition_func import AcquisitionFunction, BLEUVariance

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_uq(
    model: nn.Module,
    test_data: data.DataLoader,
    save_hypotheses_to_file: bool = False,
    aq_func: AcquisitionFunction = BLEUVariance(),
    num_batches_to_validate_on: int | None = None,
) -> Tuple[float, torch.Tensor, list[Tuple[str,str, float]]]:
    total_loss = 0
    all_references: list[str] = []
    all_hypotheses: list[str] = []
    all_uq: list[torch.Tensor] = []

    logger.debug("Started validating models")

    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(test_data), desc="Running validation", total=len(test_data)
        ):
            src_tokens, ground_truth = batch
            src_tokens, ground_truth = src_tokens.to(device), ground_truth.to(device)
            
            output,uq = generate_autoregressivly_with_uq(model, src_tokens, ground_truth, print_ex=1, aq_func=aq_func)
            all_uq.extend(uq)
            all_hypotheses.extend(output)
            all_references.extend([output_to_text(ref) for ref in ground_truth.tolist()])

            if num_batches_to_validate_on is not None and i + 1 >= num_batches_to_validate_on:
                logger.info(f"Only  validating on {num_batches_to_validate_on} batches, stopping")
                break

    if save_hypotheses_to_file:
        logger.info("Saving hypotheses to file")
        with open("local/data/test/hypotheses.en", "w") as f:
            for hyp in all_hypotheses:
                f.write(hyp + "\n")

    bleu_score = corpus_bleu(all_hypotheses, [all_references]).score

    logger.info(f"Validation BLEU Score: {bleu_score}")

    avg_uq = torch.stack(all_uq).mean()
    
    flattened_uq: list[float] = [uq.item() for uq in all_uq]
    hyp_ref_uq_pair = list(zip(all_hypotheses,all_references, flattened_uq))
    return bleu_score, avg_uq, hyp_ref_uq_pair


if __name__ == "__main__":
    dummy_hyptheses = [
        "This is a string for BLEU metric computation",
        "Banana is nice for health",
        "cat makes sounds"
    ]
    dummy_references = [
        "This is a test sentence for BLEU score calculation",
        "Banana is good for a long life",
        "cat is meowing",
        "house is big",
    ]

    bleu_score = corpus_bleu(dummy_hyptheses, [dummy_references]).score
    print(f"Dummy BLEU Score: {bleu_score}")