import logging
from typing import Any, List

import matplotlib.pyplot as plt
import torch

from gpt2project.gpt2_commongen import CommongenEval
from gpt2project.gpt2_squad import SquadEval

logger = logging.getLogger(__name__)


def plot_retention_curve_cg(
    output_texts: List[str],
    concepts: List[List[str]],
    targets: List[List[str]],
    uq: torch.Tensor,
    eval_function: CommongenEval,
    aq_func_name: str,
    filepath: str,
) -> None:
    # Sort the results based on UQ
    sorted_indices = sorted(range(len(uq)), key=lambda i: uq[i].item())
    assert sorted_indices != list(range(len(uq))), "UQ is not working"

    sorted_outputs = [output_texts[i] for i in sorted_indices]
    sorted_targets = [targets[i] for i in sorted_indices]
    sorted_concepts = [concepts[i] for i in sorted_indices]

    cutoffs = range(1, len(sorted_outputs) + 1)
    retention_scores = []

    for cutoff in cutoffs:
        selected_outputs = sorted_outputs[:cutoff]
        selected_targets = sorted_targets[:cutoff]
        selected_concepts = sorted_concepts[:cutoff]
        score = eval_function(selected_outputs, selected_concepts, selected_targets)
        retention_scores.append(score)

    plt.figure()
    plt.plot(cutoffs, retention_scores)
    plt.xlabel("Number of Samples")
    plt.ylabel("Evaluation Score")
    plt.title(f"Retention Curve for {aq_func_name}")
    plt.savefig(filepath)
    logger.info("Saved retention curve to %s", filepath)
    plt.show()


def plot_retention_curve_squad(
    output_texts: List[str],
    targets: List[List[str]],
    uq: torch.Tensor,
    eval_function_squad: SquadEval,
    aq_func_name: str,
    filepath: str,
) -> None:
    # Sort the results based on UQ
    sorted_indices = sorted(range(len(uq)), key=lambda i: uq[i].item())
    sorted_outputs = [output_texts[i] for i in sorted_indices]
    sorted_targets = [targets[i] for i in sorted_indices]

    cutoffs = range(1, len(output_texts) + 1, 10)
    retention_scores = []

    for cutoff in cutoffs:
        selected_outputs = (
            sorted_outputs[:cutoff] if cutoff > 1 else [sorted_outputs[0]]
        )
        selected_targets = (
            sorted_targets[:cutoff] if cutoff > 1 else [sorted_targets[0]]
        )
        score = eval_function_squad(selected_outputs, selected_targets)
        retention_scores.append(score)

    plt.figure()
    plt.plot(cutoffs, retention_scores)
    plt.xlabel("Number of Samples")
    plt.ylabel(f"Evaluation Score: {eval_function_squad.__class__.__name__}")
    plt.title(f"Retention Curve for {aq_func_name}")
    plt.savefig(filepath)
    logger.info("Saved retention curve to %s", filepath)
    plt.show()
