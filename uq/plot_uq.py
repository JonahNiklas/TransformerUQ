import torch
from typing import List, Tuple
import matplotlib.pyplot as plt
from sacrebleu import corpus_bleu

def plot_data_retained_curve(hyp_ref_uq_pairs: List[List[Tuple[str,str, float]]],methods: list[str],save_path: str) -> None:
    # Sort the hypothesis-UQ pairs by UQ value
    bleu_scores = [[] for _ in range(len(hyp_ref_uq_pairs))]
    interval = 0.05
    for idx,hyp_ref_uq_pair in enumerate(hyp_ref_uq_pairs):
        hyp_ref_uq_pair.sort(key=lambda x: x[2])
        
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
    plt.xlabel("Data retained")
    plt.ylabel("BLEU Score")
    plt.title("Data Retained Curve")
    plt.savefig(save_path)
    plt.show()
    print("Data retained curve saved at: ", save_path)



def plot_uq_histogram(hyp_ref_uq_pair: List[Tuple[str,str, float]], hyp_ref_uq_pair_ood: List[Tuple[str,str, float]],method: str,save_path:str) -> None:
    """
    Plot the histogram of the given UQ values.

    Args:
    uq_values: list of UQ values.
    """
    test_uq_values = [pair[2] for pair in hyp_ref_uq_pair]
    test_ood_uq_values = [pair[2] for pair in hyp_ref_uq_pair_ood]

    min_length = min(len(test_uq_values), len(test_ood_uq_values))
    test_uq_values = test_uq_values[:min_length]
    test_ood_uq_values = test_ood_uq_values[:min_length]

    assert len(test_uq_values) == len(test_ood_uq_values)

    plt.figure()
    plt.hist(test_uq_values, bins=20, alpha=0.5, label='In-distribution')    
    plt.hist(test_ood_uq_values, bins=20, alpha=0.5, label='Out-of-distribution')
    plt.xlabel("UQ Value")
    plt.ylabel("Frequency")
    plt.title("UQ Histogram with "+method)
    plt.legend(loc='upper right')
    plt.savefig(save_path)
    plt.show()
    print("UQ histogram saved at: ", save_path)
    
    