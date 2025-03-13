from __future__ import annotations
from typing import List, Any, Union
import torch
from utils.general_plotter import PlotData, cache_plot_data
from gpt2project.utils.benchmark_eval_funcs import (
    KeywordEval,
    MultipleTargetEval,
    SingleTargetEval,
)


def calc_retention_curve_commongen(
    output_texts: List[List[str]],
    concepts: List[List[str]],
    targets: List[List[str]],
    uqs: torch.Tensor,
    eval_function: MultipleTargetEval | KeywordEval,
    aq_func_names: List[str],
    stepsize: int,
    benchmark_name: str,
    model_name: str,
    folder: str,
    filename: str,
) -> None:
    assert len(output_texts) == len(aq_func_names) == uqs.size(1)
    retention_scores: List[List[float]] = [[] for _ in range(len(aq_func_names))]

    for aq, aq_func_name in enumerate(aq_func_names):
        # Sort the results based on UQ
        uq = uqs[:, aq]
        o_text = output_texts[aq]
        assert len(uq) == len(o_text), "UQ and output texts are not the same length"

        sorted_indices = sorted(range(len(uq)), key=lambda i: abs(uq[i].item()))
        assert sorted_indices != list(range(len(uq))), "UQ is not working"

        sorted_outputs = [o_text[i] for i in sorted_indices]
        sorted_targets = [targets[i] for i in sorted_indices]
        sorted_concepts = [concepts[i] for i in sorted_indices]

        cutoffs = range(stepsize, len(sorted_outputs) + 1, stepsize)

        for cutoff in cutoffs:
            selected_outputs = sorted_outputs[:cutoff]
            selected_targets = sorted_targets[:cutoff]
            selected_concepts = sorted_concepts[:cutoff]
            if isinstance(eval_function, KeywordEval):
                score = eval_function(selected_outputs, selected_concepts)
            else:
                score = eval_function(selected_outputs, selected_targets)
            retention_scores[aq].append(score)

    cache_plot_data(
        PlotData(
            eval_method=eval_function.__class__.__name__,
            model_name=model_name,
            benchmark=benchmark_name,
            uq_methods=aq_func_names,
            eval_scores=retention_scores,
            x_points=list(cutoffs),
        ),
        folder,
        filename,
    )


def calc_retention_curve(
    output_texts: List[List[str]],
    targets: List[List[str]],
    uqs: torch.Tensor,
    eval_function: MultipleTargetEval,
    aq_func_names: List[str],
    stepsize: int,
    benchmark_name: str,
    model_name: str,
    folder: str,
    filename: str,
) -> None:
    assert len(output_texts) == len(aq_func_names) == uqs.size(1)
    retention_scores: List[List[float]] = [[] for _ in range(len(aq_func_names))]

    for aq, aq_func_name in enumerate(aq_func_names):
        # Sort the results based on UQ
        uq = uqs[:, aq]
        o_text = output_texts[aq]
        assert len(uq) == len(o_text), "UQ and output texts are not the same length"

        sorted_indices = sorted(range(len(uq)), key=lambda i: abs(uq[i].item()))
        assert sorted_indices != list(range(len(uq))), "UQ is not working"

        sorted_outputs = [o_text[i] for i in sorted_indices]
        sorted_targets = [targets[i] for i in sorted_indices]

        cutoffs = range(stepsize, len(sorted_outputs) + 1, stepsize)

        for cutoff in cutoffs:
            selected_outputs = (
                sorted_outputs[:cutoff] if cutoff > 1 else [sorted_outputs[0]]
            )
            selected_targets = (
                sorted_targets[:cutoff] if cutoff > 1 else [sorted_targets[0]]
            )
            score = eval_function(selected_outputs, selected_targets)
            retention_scores[aq].append(score)
    cache_plot_data(
        PlotData(
            eval_method=eval_function.__class__.__name__,
            model_name=model_name,
            benchmark=benchmark_name,
            uq_methods=aq_func_names,
            eval_scores=retention_scores,
            x_points=list(cutoffs),
        ),
        folder,
        filename,
    )
