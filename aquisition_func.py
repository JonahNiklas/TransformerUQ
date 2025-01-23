# aquisition functions for quantifying uncertainty in among the generated text

import numpy as np
import sacrebleu

def _length_penalty(output, alpha):

    return ((5 + len(output)) / 6) ** alpha

def beam_score(output, probability,alpha=0.6):
    """
    Given an output and its probability, calculate the beam score
    """
    return np.log(probability) / _length_penalty(output,alpha)
    

def sequence_probability(outputs,output_probabilities,alpha=0.6):
    """
    Given a list of outputs and their probabilities, calculate the probability of the sequence
    """
    return np.log(np.sum(output_probabilities))/_length_penalty(outputs,alpha)


def BLEU_variance(outputs):
    """
    Given a list of outputs, approximate the variance of the BLEU scores between them
    """
    n = len(outputs)
    bleu_distances = []
    for i in range(n):
        for j in range(i + 1, n):
            bleu_score = sacrebleu.corpus_bleu(outputs[i], [outputs[j]]).score
            bleu_distances.append((1 - bleu_score)**2)
    return np.sum(bleu_distances)

def BLEU_mean_output(outputs):
    """
    Given a list of outputs, find the output with the least BLEU distance to the rest
    """
    n = len(outputs)
    min_bleu_distance = float('inf')
    min_index = -1
    for i in range(n):
        bleu_distance_sum = 0
        for j in range(n):
            if i != j:
                bleu_distance_sum += sacrebleu.corpus_bleu(outputs[i], [outputs[j]]).score
                bleu_distance_sum += sacrebleu.corpus_bleu(outputs[j], [outputs[i]]).score
            if bleu_distance_sum > min_bleu_distance:
                break
        if bleu_distance_sum < min_bleu_distance:
            min_bleu_distance = bleu_distance_sum
            min_index = i
    return outputs[min_index]