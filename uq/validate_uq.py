from __future__ import annotations

import logging
from typing import Tuple
import torch
import torch.nn as nn
import torch.utils.data as data
from sacrebleu import corpus_bleu
from tqdm import tqdm

from beam_search import beam_search_batched, beam_search_unbatched, greedy_search
from generate import generate_autoregressivly
from uq.generate_with_uq import generate_autoregressivly_with_uq
from data_processing.vocab import output_to_text
from uq.acquisition_func import AcquisitionFunction, BLEUVariance

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_uq(
    model: nn.Module,
    test_data: data.DataLoader,
    save_hypotheses_to_file: bool = False,
    aq_func: AcquisitionFunction = BLEUVariance(),
) -> Tuple[float, torch.Tensor]:
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

    if save_hypotheses_to_file:
        logger.info("Saving hypotheses to file")
        with open("local/data/test/hypotheses.en", "w") as f:
            for hyp in all_hypotheses:
                f.write(hyp + "\n")

    bleu_score = corpus_bleu(all_hypotheses, [all_references]).score
    logger.info(f"Validation BLEU Score: {bleu_score}")

    avg_uq = torch.stack(all_uq).mean()
    return bleu_score, avg_uq

