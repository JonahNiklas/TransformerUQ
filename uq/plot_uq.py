import torch
from typing import List, Tuple
import matplotlib.pyplot as plt
from sacrebleu import sentence_bleu

def plot_data_retained_curve(hypothesis_uq_pairs: List[Tuple[str,str, torch.Tensor]],save_path: str) -> None:
    """
    Plot the data retained curve for the given hypothesis-UQ pairs.

    Args:
    hypothesis_uq_pairs: list of tuples containing the hypothesis and the uncertainty quantification value.
    """
    
    # Sort the hypothesis-UQ pairs by UQ value
    hypothesis_uq_pairs.sort(key=lambda x: x[2].item())
    
    interval = 0.1
    
    bleu_scores = []
    for i in range(0, len(hypothesis_uq_pairs), int(interval * len(hypothesis_uq_pairs))):
        interval_pairs = hypothesis_uq_pairs[i:i + int(interval * len(hypothesis_uq_pairs))]
        interval_bleu_scores = [sentence_bleu(pair[0], [pair[1]]) for pair in interval_pairs]
        bleu_scores.append(interval_bleu_scores)
        
        
    # Plot the data retained curve
    plt.figure()
    plt.plot(range(0, len(hypothesis_uq_pairs), int(interval * len(hypothesis_uq_pairs))), bleu_scores)
    plt.xlabel("Data retained")
    plt.ylabel("BLEU Score")
    plt.title("Data Retained Curve")
    plt.savefig(save_path)
    plt.show()



def plot_uq_histogram(hypothesis_uq_pairs: List[Tuple[str,str, torch.Tensor]], hypothesis_uq_pairs_ood: List[Tuple[str,str, torch.Tensor]],save_path:str) -> None:
    """
    Plot the histogram of the given UQ values.

    Args:
    uq_values: list of UQ values.
    """
    test_uq_values = [pair[2].item() for pair in hypothesis_uq_pairs]
    test_ood_uq_values = [pair[2].item() for pair in hypothesis_uq_pairs_ood]

    min_length = min(len(test_uq_values), len(test_ood_uq_values))
    test_uq_values = test_uq_values[:min_length]
    test_ood_uq_values = test_ood_uq_values[:min_length]

    assert len(test_uq_values) == len(test_ood_uq_values)

    plt.figure()
    plt.hist(test_uq_values, bins=20, alpha=0.5, label='In-distribution')    
    plt.hist(test_ood_uq_values, bins=20, alpha=0.5, label='Out-of-distribution')
    plt.xlabel("UQ Value")
    plt.ylabel("Frequency")
    plt.title("UQ Histogram")
    plt.legend(loc='upper right')
    plt.savefig(save_path)
    plt.show()
    
    