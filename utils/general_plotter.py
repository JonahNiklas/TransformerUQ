from __future__ import annotations
import logging
import os
from typing import Any, List
import matplotlib.pyplot as plt
from pydantic import AliasChoices, BaseModel, Field

from gpt2project.uq.evaluation_run_config import EvaluationRunConfig


logger = logging.getLogger(__name__)


class PlotData(BaseModel):
    eval_method: str
    search_method_type: str
    enable_mcdo: bool
    model_name: str
    benchmark: str
    aq_func_names: List[str] = Field(
        validation_alias=AliasChoices("aq_func_names", "uq_methods")
    )
    eval_scores: List[List[float]]
    x_points: List[float]
    auc: List[float] = Field(default_factory=list)

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Normalize x_points to be between 0 and 1
        self.x_points = [x / self.x_points[-1] for x in self.x_points]

        # Extract AUC values from aq_func_names if present
        cleaned_names = []
        auc_values = []
        for name in self.aq_func_names:
            if "(AUC =" in name:
                base_name, auc_part = name.split(" (AUC =")
                auc_value = float(auc_part.rstrip(")"))
                cleaned_names.append(base_name)
                auc_values.append(auc_value)
            else:
                cleaned_names.append(name)
                auc_values.append(0.0)
        self.aq_func_names = cleaned_names
        self.auc = auc_values

        if self.model_name == "own":
            self.model_name = "Transformer"
        elif self.model_name == "bayesformer":
            self.model_name = "BayesFormer"
        elif self.model_name == "pytorch":
            self.model_name = "PyTorch"
        

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
    return f"local/gpt-evaluation-cache/{evaluation_run_config.run_name}"


def _get_gpt_evaluation_cache_filename(
    evaluation_run_config: EvaluationRunConfig,
) -> str:
    return f"{evaluation_run_config.dataset.__class__.__name__}_{evaluation_run_config.model.__class__.__name__}_mcdo{evaluation_run_config.enable_mcdo}_{evaluation_run_config.search_method.__name__}.pt"


def get_gpt_plot_data_path(
    evaluation_run_config: EvaluationRunConfig,
    file_extension: str = ".json",
) -> str:
    folder = _get_gpt_plot_data_folder(evaluation_run_config)
    os.makedirs(folder, exist_ok=True)
    filename = _get_gpt_plot_data_filename(evaluation_run_config, file_extension)
    return os.path.join(folder, filename)


def _get_gpt_plot_data_folder(
    evaluation_run_config: EvaluationRunConfig,
) -> str:
    return f"local/gpt-plot-data/{evaluation_run_config.run_name}"


def _get_gpt_plot_data_filename(
    evaluation_run_config: EvaluationRunConfig,
    file_extension: str,
) -> str:
    return f"{evaluation_run_config.dataset.__class__.__name__}_{evaluation_run_config.model.__class__.__name__}_mcdo{evaluation_run_config.enable_mcdo}_{evaluation_run_config.search_method.__name__}_{evaluation_run_config.eval_function.__class__.__name__}{file_extension}"


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


def plot_ret_curve(
    plot_data: PlotData | List[PlotData], title: str, save_filepath: str
) -> None:
    if isinstance(plot_data, PlotData):
        plot_data = [plot_data]

    assert all(
        plot_datum.benchmark == plot_data[0].benchmark for plot_datum in plot_data
    ), "All benchmarks must be the same"
    assert all(
        plot_datum.eval_method == plot_data[0].eval_method for plot_datum in plot_data
    ), "All evaluation methods must be the same"

    plt.figure()
    for plot_datum in plot_data:
        for aq in range(len(plot_datum.aq_func_names)):
            plt.plot(
                plot_datum.x_points,
                plot_datum.eval_scores[aq],
                label=f"{plot_datum.model_name} {plot_datum.search_method_type} {"MCDO" if plot_datum.enable_mcdo else ""} {plot_datum.aq_func_names[aq]}",
            )
    plt.xlabel("Number of Samples")
    plt.ylabel("Evaluation Score")
    plt.title(title + "| " + plot_data[0].eval_method)
    plt.legend()
    plt.savefig(save_filepath)
    logger.info(f"Saved plot to {save_filepath}")
    plt.show()
