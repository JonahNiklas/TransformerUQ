from typing import List, Tuple
import matplotlib.pyplot as plt
from sacrebleu import corpus_bleu
from sklearn.metrics import roc_curve, auc

from uq.validate_uq import ValidationResult

def plot_data_retained_curve(validationResults: List[ValidationResult],methods: List[str],save_path: str, run_name:str) -> None:
    # Sort the hypothesis-UQ pairs by UQ value
    bleu_scores: List[List[float]] = [[] for _ in range(len(validationResults))]
    interval = 0.025
    for idx,val_result in enumerate(validationResults):

        hyp_ref_uq_pair = [(val_result.hypothesis[i], val_result.reference[i], val_result.uncertainty[i].item()) for i in range(len(val_result.hypothesis))]

        hyp_ref_uq_pair.sort(key=lambda x: abs(x[2]))
        
        for i in range(0, len(hyp_ref_uq_pair), int(interval * len(hyp_ref_uq_pair))):
            interval_pairs = hyp_ref_uq_pair[:i + int(interval * len(hyp_ref_uq_pair))]
            hypothesis_in_interval = [pair[0] for pair in interval_pairs]
            reference_in_interval = [pair[1] for pair in interval_pairs]
            interval_bleu_scores = corpus_bleu(hypothesis_in_interval, [reference_in_interval]).score
            bleu_scores[idx].append(interval_bleu_scores)
        
        
    # Plot the data retained curve
    plt.figure()
    for i in range(len(bleu_scores)):
        plt.plot([i * interval for i in range(len(bleu_scores[i]))], bleu_scores[i], label=f"{methods[i]}")
    plt.legend()
    plt.xlabel("Data retained")
    plt.ylabel("BLEU Score")
    plt.title(f"Data Retained Curve for {run_name}")
    plt.savefig(save_path)
    plt.show()
    print("Data retained curve saved at: ", save_path)



def plot_uq_histogram_and_roc(validation_result_id: ValidationResult, validation_result_ood: ValidationResult, method: str,save_path:str,run_name:str) -> None:
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

    plt.figure()
    plt.hist(test_uq_values_id, bins=20, alpha=0.5, label='In-distribution')    
    plt.hist(test_uq_values_ood, bins=20, alpha=0.5, label='Out-of-distribution')
    plt.xlabel("UQ Value")
    plt.ylabel("Frequency")
    plt.title(f"UQ Histogram with {method} for {run_name}")
    plt.legend(loc='upper left')
    plt.savefig(save_path)
    plt.show()
    print("UQ histogram saved at: ", save_path)

    # Generate true labels (0 for in-distribution, 1 for out-of-distribution)
    true_labels = [0] * len(test_uq_values_id) + [1] * len(test_uq_values_ood)
    uq_values = test_uq_values_id + test_uq_values_ood

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(true_labels, uq_values)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve with {method} for {run_name}')
    plt.legend()
    plt.savefig(save_path.replace('.svg', '_roc.svg'))
    plt.show()
    print("ROC curve saved at: ", save_path.replace('.svg', '_roc.svg'))