from typing import List
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class PlotData:
    eval_method: str
    model: str
    benchmark: str
    uq_methods: List[str]
    eval_scores: List[float]
    x_points: List[int]


def plot_ret_curve(plot_data: List[PlotData], title: str, filepath: str) -> None:
    assert all(
        plot_data.benchmark == plot_data[0].benchmark for plot_data in plot_data
    ), "All benchmarks must be the same"
    plt.figure()
    for plot_data in plot_data:
        plt.plot(
            plot_data.x_points,
            plot_data.eval_scores,
            label=f"{plot_data.model} {plot_data.eval_method}",
        )
    plt.xlabel("Number of Samples")
    plt.ylabel("Evaluation Score")
    plt.title(title)
    plt.legend()
    plt.savefig(filepath)
    plt.show()
