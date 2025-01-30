# aquisition functions for quantifying uncertainty in among the generated text

import numpy as np
import sacrebleu
import torch
from typing import Union, List, cast
from hyperparameters import hyperparameters

def _length_penalty(output: List[str], alpha: float) -> torch.Tensor:
    lengths = torch.tensor([len(out) for out in output]).to(hyperparameters.device)
    return torch.tensor(((5 + lengths) / 6) ** alpha)

class AcquisitionFunction:
    def __init__(self, multiple_inference: bool = False, num_inferences: int = 5, alpha: float = 0.6) -> None:
        self.alpha = alpha
        self.multiple_inference = multiple_inference
        self.num_inferences = num_inferences

    def __call__(self, output: Union[List[str], List[List[str]]], logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method")

class BeamScore(AcquisitionFunction):
    def __call__(self, output: Union[List[str], List[List[str]]], logits: torch.Tensor) -> torch.Tensor:
        # logits dim: (batch_size, max_len)
        assert isinstance(output[0], str), "Output should be a list of strings"
        output = cast(List[str], output)

        log_prob = torch.sum(torch.log_softmax(logits, dim=1), dim=1) # (batch_size)
        return log_prob / _length_penalty(output, self.alpha)

# class SequenceProbability(AcquisitionFunction):
#     def __init__(self, multiple_inference: bool = True, num_inferences: int = 5, alpha: float = 0.6) -> None:
#         super().__init__(multiple_inference, num_inferences, alpha)

#     def __call__(self, output: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
#         # logits dim: (batch_size, num_inferences, max_len, vocab_size)
#         # output dim: (batch_size, num_inferences, max_len)
#         assert isinstance(output[0], str), "Output should be a list of strings"

#         logits = logits[:, :, :, ouptut.idx]

#         probabilities = torch.softmax(logits, dim=2) # (batch_size, num_inferences, max_len)
#         probability = torch.prod(probabilities, dim=2) # (batch_size, num_inferences)
#         probability_sum = torch.sum(probability, dim=1) # (batch_size)
#         return torch.log(probability_sum) / _length_penalty(output, self.alpha)

class BLEUVariance(AcquisitionFunction):
    def __init__(self, multiple_inference: bool = True, num_inferences: int = 5, alpha: float = 0.6) -> None:
        super().__init__(multiple_inference, num_inferences, alpha)

    def __call__(self, output: Union[List[str], List[List[str]]], probability: torch.Tensor) -> torch.Tensor:
        assert isinstance(output[0], list), "Output should be a list of lists"
        batch = len(output)
        bleu_distances = torch.zeros(batch)
        for b in range(batch):
            for i in range(self.num_inferences):
                for j in range(i + 1, self.num_inferences):
                    bleu_dist = sacrebleu.sentence_bleu(output[b][i], [output[b][j]]).score
                    bleu_distances[b] += (1 - bleu_dist / 100) ** 2
        return bleu_distances
        
def BLEU_mean_output_batch(outputs: List[List[str]]) -> List[str]:
    """
    Given a batch of outputs, find the output with the least BLEU distance to the rest for each batch element
    """
    batch_size = len(outputs)
    mean_outputs = []
    for b in range(batch_size):
        batch_outputs = outputs[b]
        n = len(batch_outputs)
        min_bleu_distance = float('inf')
        min_index = -1
        for i in range(n):
            bleu_distance_sum = float(0)
            for j in range(n):
                if i != j:
                    bleu_distance_sum += sacrebleu.corpus_bleu(batch_outputs[i], [batch_outputs[j]]).score
                    bleu_distance_sum += sacrebleu.corpus_bleu(batch_outputs[j], [batch_outputs[i]]).score
                if bleu_distance_sum > min_bleu_distance:
                    break
            if bleu_distance_sum < min_bleu_distance:
                min_bleu_distance = bleu_distance_sum
                min_index = i
        mean_outputs.append(batch_outputs[min_index])
    return mean_outputs