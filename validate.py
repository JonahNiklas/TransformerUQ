from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
from sacrebleu import corpus_bleu
from tqdm import tqdm

from generate import generate_autoregressivly
from vocab import output_to_text
from acquisition_func import BLEUVariance

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(
    model: nn.Module,
    test_data: data.DataLoader,
    criterion: nn.Module | None,
    save_hypotheses_to_file: bool = False,
) -> float:
    model.eval()
    total_loss = 0
    all_references: list[str] = []
    all_hypotheses: list[str] = []

    logger.debug("Started validating models")

    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(test_data), desc="Running validation", total=len(test_data)
        ):
            src_tokens, ground_truth = batch
            src_tokens, ground_truth = src_tokens.to(device), ground_truth.to(device)
            
            output = generate_autoregressivly(model, src_tokens, print_ex=1)

            all_hypotheses.extend(output)
            all_references.extend([output_to_text(ref) for ref in ground_truth.tolist()])
            # loss = criterion(output, tgt_tokens) # cannot calculate loss after taking argmax
            # total_loss += loss.item()
            # logger.warning("Validation on only one batch for now")
            # break

    if save_hypotheses_to_file:
        logger.info("Saving hypotheses to file")
        with open("local/data/test/hypotheses.en", "w") as f:
            for hyp in all_hypotheses:
                f.write(hyp + "\n")

    # avg_loss = total_loss # / len(test_data) TODO: change this when running on more than one batch
    bleu_score = corpus_bleu(all_hypotheses, [all_references]).score

    # print(f"Validation Loss: {avg_loss} | BLEU Score: {bleu_score}")
    logger.info(f"Validation BLEU Score: {bleu_score}")

    # return bleu_score, avg_loss
    return bleu_score


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
