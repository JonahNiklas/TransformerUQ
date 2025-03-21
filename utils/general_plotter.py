import logging
import os
from typing import List
import matplotlib.pyplot as plt
from pydantic import BaseModel

from gpt2project.uq.evaluation_run_config import EvaluationRunConfig


logger = logging.getLogger(__name__)


class PlotData(BaseModel):
    eval_method: str
    search_method_type: str
    enable_mcdo: bool
    model_name: str
    benchmark: str
    uq_methods: List[str]
    eval_scores: List[List[float]]
    x_points: List[float]


def get_gpt_evaluation_path(
    evaluation_run_config: EvaluationRunConfig,
) -> str:
    folder = _get_gpt_evaluation_cache_folder(evaluation_run_config)
    os.makedirs(folder, exist_ok=True)
    filename = _get_gpt_evaluation_cache_filename(evaluation_run_config)
    return os.path.join(folder, filename)


def _get_gpt_evaluation_cache_folder(
    evaluation_run_config: EvaluationRunConfig,
) -> str:
    return f"local/gpt-cache/{evaluation_run_config.benchmark_name}/{evaluation_run_config.model.__class__.__name__}/mcdo{evaluation_run_config.enable_mcdo}/{evaluation_run_config.search_method.__name__}"


def _get_gpt_evaluation_cache_filename(
    evaluation_run_config: EvaluationRunConfig,
) -> str:
    return f"{evaluation_run_config.benchmark_name}_outputs_{evaluation_run_config.model.__class__.__name__}_{evaluation_run_config.run_name}_n{evaluation_run_config.n_batches_to_validate}.pt"


def get_gpt_plot_data_path(
    evaluation_run_config: EvaluationRunConfig,
) -> str:
    folder = _get_gpt_plot_data_folder(evaluation_run_config)
    os.makedirs(folder, exist_ok=True)
    filename = _get_gpt_plot_data_filename(evaluation_run_config)
    return os.path.join(folder, filename)


def _get_gpt_plot_data_folder(
    evaluation_run_config: EvaluationRunConfig,
) -> str:
    return f"local/gpt-results/{evaluation_run_config.benchmark_name.lower()}/{evaluation_run_config.model.__class__.__name__}/mcdo{evaluation_run_config.enable_mcdo}/{evaluation_run_config.search_method.__name__}"


def _get_gpt_plot_data_filename(
    evaluation_run_config: EvaluationRunConfig,
) -> str:
    return f"plot_data_{evaluation_run_config.run_name}_{evaluation_run_config.eval_function.__class__.__name__}_n{evaluation_run_config.n_batches_to_validate}_step{evaluation_run_config.stepsize}.pt"


def get_gpt_plot_data_wmt_path(
    evaluation_run_config: EvaluationRunConfig,
) -> str:
    folder = _get_gpt_plot_data_folder(evaluation_run_config)
    os.makedirs(folder, exist_ok=True)
    filename = _get_gpt_plot_data_filename(evaluation_run_config)
    return os.path.join(folder, filename)


def cache_plot_data(plot_data: PlotData, filepath: str) -> None:
    with open(filepath, "w") as f:
        f.write(plot_data.model_dump_json())


def cache_plot_data_wmt(plot_data: PlotData, filepath: str) -> None:
    with open(filepath, "w") as f:
        f.write(plot_data.model_dump_json())


def load_plot_data(plot_data_paths: List[str]) -> List[PlotData]:
    plot_data_list = []
    for plot_data_path in plot_data_paths:
        with open(plot_data_path, "r") as f:
            plot_data = PlotData.model_validate_json(f.read())
            plot_data_list.append(plot_data)
    return plot_data_list


def plot_ret_curve(plot_data_paths: List[str], title: str, save_filepath: str) -> None:
    plot_data = load_plot_data(plot_data_paths)
    assert all(
        plot_datum.benchmark == plot_data[0].benchmark for plot_datum in plot_data
    ), "All benchmarks must be the same"
    assert all(
        plot_datum.eval_method == plot_data[0].eval_method for plot_datum in plot_data
    ), "All evaluation methods must be the same"

    plt.figure()
    for plot_datum in plot_data:
        for aq in range(len(plot_datum.uq_methods)):
            plt.plot(
                plot_datum.x_points,
                plot_datum.eval_scores[aq],
                label=f"{plot_datum.model_name} {plot_datum.search_method_type} {"MCDO" if plot_datum.enable_mcdo else ""} {plot_datum.uq_methods[aq]}",
            )
    plt.xlabel("Number of Samples")
    plt.ylabel("Evaluation Score")
    plt.title(title + "| " + plot_data[0].eval_method)
    plt.legend()
    plt.savefig(save_filepath)
    logger.info(f"Saved plot to {save_filepath}")
    plt.show()
