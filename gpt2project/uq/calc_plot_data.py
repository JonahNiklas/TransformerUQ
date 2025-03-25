from __future__ import annotations
from typing import List
import torch
from gpt2project.uq.evaluation_run_config import EvaluationRunConfig
from utils.general_plotter import PlotData, cache_plot_data, get_gpt_plot_data_path


def calc_retention_curve(
    output_texts: List[List[str]],
    uqs: torch.Tensor,
    evaluation_run_config: EvaluationRunConfig,
) -> PlotData:
    assert len(output_texts) == len(evaluation_run_config.aq_funcs) == uqs.size(1)
    assert len(output_texts[0]) == len(evaluation_run_config.dataset)

    dataset = evaluation_run_config.dataset

    retention_scores: List[List[float]] = [
        [] for _ in range(len(evaluation_run_config.aq_funcs))
    ]
    for aq_func_idx in range(len(evaluation_run_config.aq_funcs)):
        # Sort the results based on UQ
        uq = uqs[:, aq_func_idx]
        o_text = output_texts[aq_func_idx]
        assert len(uq) == len(o_text), "UQ and output texts are not the same length"

        sorted_indices = sorted(range(len(uq)), key=lambda i: abs(uq[i].item()))
        assert sorted_indices != list(range(len(uq))), "UQ is not working"

        sorted_outputs = [o_text[i] for i in sorted_indices]
        sorted_dataset = [dataset[i] for i in sorted_indices]

        cutoffs = range(
            evaluation_run_config.stepsize,
            len(sorted_outputs) + 1,
            evaluation_run_config.stepsize,
        )

        for cutoff in cutoffs:
            selected_outputs = (
                sorted_outputs[:cutoff] if cutoff > 1 else [sorted_outputs[0]]
            )
            selected_dataset_examples = (
                sorted_dataset[:cutoff] if cutoff > 1 else [sorted_dataset[0]]
            )
            score = evaluation_run_config.eval_function(
                selected_outputs, selected_dataset_examples
            )
            retention_scores[aq_func_idx].append(score)

    plot_data = PlotData(
        eval_method=evaluation_run_config.eval_function.__class__.__name__,
        search_method_type=evaluation_run_config.search_method.__name__,
        enable_mcdo=evaluation_run_config.enable_mcdo,
        model_name=evaluation_run_config.model.__class__.__name__,
        benchmark=evaluation_run_config.dataset.__class__.__name__,
        aq_func_name=[
            aq_func.__class__.__name__ for aq_func in evaluation_run_config.aq_funcs
        ],
        eval_scores=retention_scores,
        x_points=list(cutoffs),
    )
    cache_plot_data(
        plot_data,
        get_gpt_plot_data_path(evaluation_run_config),
    )
    return plot_data
