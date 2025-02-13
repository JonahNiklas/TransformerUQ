# aquisition functions for quantifying uncertainty in among the generated text

import numpy as np
import sacrebleu
import torch
from typing import Union, List, cast
from hyperparameters import hyperparameters
from sentence_transformers import SentenceTransformer

def _length_penalty(output: torch.Tensor, alpha: float) -> torch.Tensor:
    lengths = torch.sum(output != 0, dim=1)
    return torch.tensor(((5 + lengths) / 6) ** alpha)

class AcquisitionFunction:
    def __init__(self, multiple_inference: bool = False, num_inferences: int = 5, alpha: float = 0.6) -> None:
        self.alpha = alpha
        self.multiple_inference = multiple_inference
        self.num_inferences = num_inferences

    def __call__(self, hypothesis: List[List[str]], tgt_tokens: torch.Tensor, token_softmax_probs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method")

class BeamScore(AcquisitionFunction):
    def __call__(self, hypothesis: List[List[str]], tgt_tokens: torch.Tensor, token_softmax_probs: torch.Tensor) -> torch.Tensor:
        # token_softmax_probs dim: (batch_size, max_len)
        #only use first inference
        token_softmax_probs = token_softmax_probs[:, 0, :]
        tgt_tokens = tgt_tokens[:, 0, :]
        log_prob = torch.log(token_softmax_probs)
        seq_prob = torch.sum(log_prob, dim=1)
        return seq_prob / _length_penalty(tgt_tokens, self.alpha)

# class SequenceProbability(AcquisitionFunction):
#     def __init__(self, multiple_inference: bool = True, num_inferences: int = 5, alpha: float = 0.6) -> None:
#         super().__init__(multiple_inference, num_inferences, alpha)

#     def __call__(self, output: torch.Tensor, token_softmax_probs: torch.Tensor) -> torch.Tensor:
#         # token_softmax_probs dim: (batch_size, num_inferences, max_len, vocab_size)
#         # output dim: (batch_size, num_inferences, max_len)
#         assert isinstance(output[0], str), "Output should be a list of strings"

#         token_softmax_probs = token_softmax_probs[:, :, :, ouptut.idx]

#         probabilities = torch.softmax(token_softmax_probs, dim=2) # (batch_size, num_inferences, max_len)
#         probability = torch.prod(probabilities, dim=2) # (batch_size, num_inferences)
#         probability_sum = torch.sum(probability, dim=1) # (batch_size)
#         return torch.log(probability_sum) / _length_penalty(output, self.alpha)

class VR_mpnet_dot(AcquisitionFunction):
    def __init__(self, multiple_inference: bool = True, num_inferences: int = hyperparameters.uq.num_inferences, alpha: float = 0.6) -> None:
        super().__init__(multiple_inference, num_inferences, alpha)
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def __call__(self, hypothesis: List[List[str]], tgt_tokens: torch.Tensor, token_softmax_probs: torch.Tensor) -> torch.Tensor:

        batch = len(hypothesis)
        distances = torch.zeros(batch).to(hyperparameters.device)
        for b in range(batch):
            embeddings = self.model.encode(hypothesis[b], convert_to_tensor=True, normalize_embeddings=True).to(hyperparameters.device)
            for i in range(self.num_inferences):
                for j in range(i + 1, self.num_inferences):
                    dot_product = torch.dot(embeddings[i], embeddings[j]).item()
                    distances[b] += 1-dot_product

        return distances/self.num_inferences


class VR_mpnet_base_cosine(AcquisitionFunction):
    def __init__(self, multiple_inference: bool = True, num_inferences: int = hyperparameters.uq.num_inferences, alpha: float = 0.6) -> None:
        super().__init__(multiple_inference, num_inferences, alpha)
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def __call__(self, hypothesis: List[List[str]], tgt_tokens: torch.Tensor, token_softmax_probs: torch.Tensor) -> torch.Tensor:

        batch = len(hypothesis)
        distances = torch.zeros(batch).to(hyperparameters.device)
        for b in range(batch):
            embeddings = self.model.encode(hypothesis[b], convert_to_tensor=True, normalize_embeddings=True).to(hyperparameters.device)
            for i in range(self.num_inferences):
                for j in range(i + 1, self.num_inferences):
                    cosine_similarity = torch.nn.functional.cosine_similarity(
                        embeddings[i].unsqueeze(0),
                        embeddings[j].unsqueeze(0)
                    ).item()
                    distances[b] += 1-cosine_similarity

        return distances/self.num_inferences

class VR_mpnet_base_matrix_norm(AcquisitionFunction):
    def __init__(self, multiple_inference: bool = True, num_inferences: int = hyperparameters.uq.num_inferences, alpha: float = 0.6) -> None:
        super().__init__(multiple_inference, num_inferences, alpha)
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def __call__(self, hypothesis: List[List[str]], tgt_tokens: torch.Tensor, token_softmax_probs: torch.Tensor) -> torch.Tensor:

        batch = len(hypothesis)
        distances = torch.zeros(batch).to(hyperparameters.device)
        for b in range(batch):
            embeddings = self.model.encode(hypothesis[b], convert_to_tensor=True, normalize_embeddings=True).to(hyperparameters.device)
            for i in range(self.num_inferences):
                for j in range(i + 1, self.num_inferences):
                    matrix_norm = torch.norm(embeddings[i] - embeddings[j])
                    distances[b] += matrix_norm

        return distances/self.num_inferences

class BLEUVariance(AcquisitionFunction):
    def __init__(self, multiple_inference: bool = True, num_inferences: int = hyperparameters.uq.num_inferences, alpha: float = 0.6) -> None:
        super().__init__(multiple_inference, num_inferences, alpha)

    def __call__(self, hypothesis: List[List[str]], tgt_tokens: torch.Tensor, token_softmax_probs: torch.Tensor) -> torch.Tensor:
        batch = len(hypothesis)
        bleu_distances = torch.zeros(batch).to(hyperparameters.device)
        for b in range(batch):
            for i in range(self.num_inferences):
                for j in range(self.num_inferences):
                    if i != j:
                        continue
                    bleu_dist = sacrebleu.sentence_bleu(hypothesis[b][i], [hypothesis[b][j]]).score
                    bleu_distances[b] += (1 - bleu_dist / 100) ** 2
        return bleu_distances/self.num_inferences
        
def BLEU_mean_output_batch(outputs: List[List[str]]) -> List[str]:
    """
    Given a batch of outputs, find the output with the least BLEU distance to the rest for each batch element
    """
    batch_size = len(outputs)
    mean_outputs = []
    for b in range(batch_size):
        batch_outputs = outputs[b]
        if len(batch_outputs) == 0:
            continue
        n = len(batch_outputs)
        min_bleu_distance = float('inf')
        min_index = -1
        for i in range(n):
            bleu_distance_sum = float(0)
            for j in range(n):
                if i != j:
                    bleu_distance_sum += 1-sacrebleu.corpus_bleu([batch_outputs[i]], [[batch_outputs[j]]]).score/100
                    bleu_distance_sum += 1- sacrebleu.corpus_bleu([batch_outputs[j]], [[batch_outputs[i]]]).score/100
                if bleu_distance_sum > min_bleu_distance:
                    break
            if bleu_distance_sum < min_bleu_distance:
                min_bleu_distance = bleu_distance_sum
                min_index = i
        mean_outputs.append(batch_outputs[min_index])
    return mean_outputs