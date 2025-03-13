import os
from typing import List
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pydantic import BaseModel
from regex import P


class PlotData(BaseModel):
    eval_method: str
    model_name: str
    benchmark: str
    uq_methods: List[str]
    eval_scores: List[List[float]]
    x_points: List[int]


def cache_plot_data(plot_data: PlotData, folder: str, filename: str) -> None:
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
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
    plt.figure()
    for plot_datum in plot_data:
        for aq in range(len(plot_datum.uq_methods)):
            plt.plot(
                plot_datum.x_points,
                plot_datum.eval_scores[aq],
                label=f"{plot_datum.model_name} {plot_datum.eval_method} {plot_datum.uq_methods[aq]}",
            )
    plt.xlabel("Number of Samples")
    plt.ylabel("Evaluation Score")
    plt.title(title)
    plt.legend()
    plt.savefig(save_filepath)
    plt.show()
