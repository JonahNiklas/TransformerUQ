from typing import List, Tuple
import logging
import matplotlib.pyplot as plt
import numpy as np
from sacrebleu import corpus_bleu
from sklearn.metrics import roc_curve, auc

from uq.validate_uq import ValidationResult
from utils.general_plotter import PlotData, cache_plot_data_wmt

logger = logging.getLogger(__name__)

label_fontsize = 12
plot_figsize = (6.4 * 0.85, 4.8 * 0.85)


def calc_ret_curve_plot_data_wmt(
    validationResults: List[ValidationResult],
    aq_func_names: List[str],
    model_name: str,
    eval_method: str,
    search_method: str,
    enable_mcdo: bool,
    benchmark_name: str,
    save_path: str,
) -> None:
    # Sort the hypothesis-UQ pairs by UQ value
    bleu_scores: List[List[float]] = [[] for _ in range(len(validationResults))]
    interval = 0.025
    for idx, val_result in enumerate(validationResults):
        hyp_ref_uq_pair = [
            (
                val_result.hypothesis[i],
                val_result.reference[i],
                val_result.uncertainty[i].item(),
            )
            for i in range(len(val_result.hypothesis))
        ]

        hyp_ref_uq_pair.sort(key=lambda x: abs(x[2]))

        for i in range(
            0, len(hyp_ref_uq_pair), max(int(interval * len(hyp_ref_uq_pair)), 1)
        ):
            interval_pairs = hyp_ref_uq_pair[
                : i + max(int(interval * len(hyp_ref_uq_pair)), 1)
            ]
            hypothesis_in_interval = [pair[0] for pair in interval_pairs]
            reference_in_interval = [pair[1] for pair in interval_pairs]
            interval_bleu_scores = corpus_bleu(
                hypothesis_in_interval, [reference_in_interval]
            ).score
            bleu_scores[idx].append(interval_bleu_scores)

    # Calculate the area under the retention curves
    aq_funcs_and_auc = aq_func_names
    for idx, scores in enumerate(bleu_scores):
        x = [i * interval for i in range(len(scores))]
        auc_score = auc(x, scores)
        aq_funcs_and_auc[idx] = f"{aq_funcs_and_auc[idx]} (AUC = {auc_score:.2f})"

    logger.info(f"Saving plot data to {save_path}")
    cache_plot_data_wmt(
        PlotData(
            eval_method=eval_method,
            search_method_type=search_method,
            enable_mcdo=enable_mcdo,
            model_name=model_name,
            benchmark=benchmark_name,
            aq_func_names=aq_funcs_and_auc,
            eval_scores=bleu_scores,
            x_points=[i * interval for i in range(len(bleu_scores[0]))],
        ),
        save_path,
    )


def plot_data_retained_curve(
    validationResults: List[ValidationResult],
    methods: List[str],
    save_path: str,
    run_name: str,
) -> None:
    # Sort the hypothesis-UQ pairs by UQ value
    bleu_scores: List[List[float]] = [[] for _ in range(len(validationResults))]
    interval = 0.025
    for idx, val_result in enumerate(validationResults):

        hyp_ref_uq_pair = [
            (
                val_result.hypothesis[i],
                val_result.reference[i],
                val_result.uncertainty[i].item(),
            )
            for i in range(len(val_result.hypothesis))
        ]

        hyp_ref_uq_pair.sort(key=lambda x: abs(x[2]))

        for i in range(
            0, len(hyp_ref_uq_pair), max(int(interval * len(hyp_ref_uq_pair)), 1)
        ):
            interval_pairs = hyp_ref_uq_pair[
                : i + max(int(interval * len(hyp_ref_uq_pair)), 1)
            ]
            hypothesis_in_interval = [pair[0] for pair in interval_pairs]
            reference_in_interval = [pair[1] for pair in interval_pairs]
            interval_bleu_scores = corpus_bleu(
                hypothesis_in_interval, [reference_in_interval]
            ).score
            bleu_scores[idx].append(interval_bleu_scores)

    # Calculate the area under the retention curves
    auc_scores = []
    correlations = []
    for scores in bleu_scores:
        x = [i * interval for i in range(len(scores))]
        auc_score = auc(x, scores)
        auc_scores.append(auc_score)

        corr = np.corrcoef(x, scores)[0, 1]
        correlations.append(corr)

    print(save_path, "=====")
    # Plot the data retained curve
    plt.figure(figsize=plot_figsize)
    for i in range(len(bleu_scores)):
        if methods[i] == "mpnet_dot":
            continue
        plt.plot(
            [i * interval for i in range(len(bleu_scores[i]))],
            bleu_scores[i],
            label=f"{methods[i]} (AUC = {auc_scores[i]:.2f})",
        )
        print(f"{methods[i]} (AUC = {auc_scores[i]:.2f}), Corr = {correlations[i]:.5f}")
    print("=====")
    plt.legend()
    plt.xlabel("Data retained", fontsize=label_fontsize)
    plt.ylabel("BLEU Score", fontsize=label_fontsize)
    plt.savefig(save_path)
    plt.show()
    logger.info(f"Data retained curve saved at: {save_path}")


def plot_uq_histogram(
    validation_result_id: ValidationResult,
    validation_result_ood: ValidationResult,
    method: str,
    save_path: str,
    run_name: str,
) -> None:
    """
    Plot the histogram of the given UQ values.

    Args:
    uq_values: list of UQ values.
    """

    test_uq_values_id = validation_result_id.uncertainty.tolist()
    test_uq_values_ood = validation_result_ood.uncertainty.tolist()

    min_length = min(len(test_uq_values_id), len(test_uq_values_ood))
    test_uq_values_id = test_uq_values_id[:min_length]
    test_uq_values_ood = test_uq_values_ood[:min_length]

    assert len(test_uq_values_id) == len(test_uq_values_ood)

    plt.figure(figsize=plot_figsize)
    plt.hist(test_uq_values_id, bins=20, alpha=0.5, label="In-distribution")
    plt.hist(test_uq_values_ood, bins=20, alpha=0.5, label="Out-of-distribution")
    plt.xlabel("UQ Value", fontsize=label_fontsize)
    plt.ylabel("Frequency", fontsize=label_fontsize)
    plt.legend(loc="upper left")
    plt.savefig(save_path)
    plt.show()
    logger.info(f"UQ histogram saved at: {save_path}")


def plot_combined_roc_curve(
    validation_result_id: List[ValidationResult],
    validation_result_ood: List[ValidationResult],
    methods: List[str],
    save_path: str,
) -> None:
    """
    Plot the all the ROC curves of the different uq_methods
    Args:
    validationResults: list of ValidationResults
    """
    plt.figure(figsize=plot_figsize)
    assert len(validation_result_id) == len(validation_result_ood) == len(methods)
    for val_id, val_ood, method in zip(
        validation_result_id, validation_result_ood, methods
    ):
        if method == "mpnet_dot" or method == "mpnet_cosine" or method == "mpnet_norm":
            continue

        test_uq_values_id = val_id.uncertainty.tolist()
        test_uq_values_ood = val_ood.uncertainty.tolist()

        min_length = min(len(test_uq_values_id), len(test_uq_values_ood))
        test_uq_values_id = test_uq_values_id[:min_length]
        test_uq_values_ood = test_uq_values_ood[:min_length]

        assert len(test_uq_values_id) == len(test_uq_values_ood)

        # Generate true labels (0 for in-distribution, 1 for out-of-distribution)
        true_labels = [0] * len(test_uq_values_id) + [1] * len(test_uq_values_ood)
        uq_values = test_uq_values_id + test_uq_values_ood

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(true_labels, uq_values)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{method} (area = {roc_auc:.2f})")

        # # Find TPR at FPR=0.95
        # idx = (fpr >= 0.05).nonzero()[0][0]
        # plt.plot([0.05], [tpr[idx]], 'o', label=f"{method} TPR at FPR=0.05: {tpr[idx]:.2f}")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=label_fontsize)
    plt.ylabel("True Positive Rate", fontsize=label_fontsize)
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    logger.info(f"ROC curve saved at: {save_path}")