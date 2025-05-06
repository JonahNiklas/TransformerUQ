import numpy as np
import sacrebleu
import torch
from typing import Union, List, cast
from gpt2project.search_methods_gpt import AutoregressiveInferenceResultsGPT
from hyperparameters import hyperparameters
from sentence_transformers import SentenceTransformer
import logging
from abc import abstractmethod

logger = logging.getLogger(__name__)


def _length_penalty(output: torch.Tensor, alpha: float) -> torch.Tensor:
    lengths = torch.sum(output != 0, dim=1)
    penalty: torch.Tensor = ((5 + lengths) / 6) ** alpha
    return penalty


class AcquisitionFunctionGPT:
    def __init__(
        self,
        multiple_inference: bool = False,
        num_inferences: int = 5,
        alpha: float = 0.6,
    ) -> None:
        self.alpha = alpha
        self.multiple_inference = multiple_inference
        self.num_inferences = num_inferences
        self.name = None

    @abstractmethod
    def __call__(
        self,
        hypothesis: List[List[str]],
        inference_results: List[AutoregressiveInferenceResultsGPT],
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method")


class BeamScore(AcquisitionFunctionGPT):
    def __call__(
        self,
        hypothesis: List[List[str]],
        inference_results: List[AutoregressiveInferenceResultsGPT],
    ) -> torch.Tensor:
        # token_softmax_probs dim: (batch_size, max_len)
        # only use first inference
        token_softmax_probs = inference_results[
            0
        ].get_softmax_probs_for_selected_token()
        generated_tokens = inference_results[0].token_ids
        assert (
            token_softmax_probs >= 0
        ).all(), "Softmax probabilities should be positive"
        log_prob = torch.log(token_softmax_probs)
        seq_prob = torch.sum(log_prob, dim=1)
        return seq_prob / _length_penalty(generated_tokens, self.alpha)


class mpnet_cosine(AcquisitionFunctionGPT):
    def __init__(
        self,
        multiple_inference: bool = True,
        num_inferences: int = hyperparameters.uq.num_inferences,
        alpha: float = 0.6,
    ) -> None:
        super().__init__(multiple_inference, num_inferences, alpha)
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def __call__(
        self,
        hypothesis: List[List[str]],
        inference_results: List[AutoregressiveInferenceResultsGPT],
    ) -> torch.Tensor:

        batch = len(hypothesis)
        distances = torch.zeros(batch).to(hyperparameters.device)
        for b in range(batch):
            if len(hypothesis[b]) < self.num_inferences:
                continue
            embeddings = self.model.encode(
                hypothesis[b],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).to(hyperparameters.device)
            for i in range(self.num_inferences):
                for j in range(i + 1, self.num_inferences):
                    cosine_similarity = torch.nn.functional.cosine_similarity(
                        embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)
                    ).item()
                    distances[b] += 1 - cosine_similarity

        return distances / self.num_inferences


class roberta_cosine(AcquisitionFunctionGPT):
    def __init__(
        self,
        multiple_inference: bool = True,
        num_inferences: int = hyperparameters.uq.num_inferences,
        alpha: float = 0.6,
    ) -> None:
        super().__init__(multiple_inference, num_inferences, alpha)
        self.model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

    def __call__(
        self,
        hypothesis: List[List[str]],
        inference_results: List[AutoregressiveInferenceResultsGPT],
    ) -> torch.Tensor:

        batch = len(hypothesis)
        distances = torch.zeros(batch).to(hyperparameters.device)
        for b in range(batch):
            if len(hypothesis[b]) < self.num_inferences:
                continue
            embeddings = self.model.encode(
                hypothesis[b],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).to(hyperparameters.device)
            for i in range(self.num_inferences):
                for j in range(i + 1, self.num_inferences):
                    cosine_similarity = torch.nn.functional.cosine_similarity(
                        embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)
                    ).item()
                    distances[b] += 1 - cosine_similarity

        return distances / self.num_inferences


class mpnet_norm(AcquisitionFunctionGPT):
    def __init__(
        self,
        multiple_inference: bool = True,
        num_inferences: int = hyperparameters.uq.num_inferences,
        alpha: float = 0.6,
    ) -> None:
        super().__init__(multiple_inference, num_inferences, alpha)
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def __call__(
        self,
        hypothesis: List[List[str]],
        inference_results: List[AutoregressiveInferenceResultsGPT],
    ) -> torch.Tensor:

        batch = len(hypothesis)
        distances = torch.zeros(batch).to(hyperparameters.device)
        for b in range(batch):
            if len(hypothesis[b]) < self.num_inferences:
                continue
            embeddings = self.model.encode(
                hypothesis[b],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).to(hyperparameters.device)
            for i in range(self.num_inferences):
                for j in range(i + 1, self.num_inferences):
                    matrix_norm = torch.norm(embeddings[i] - embeddings[j])
                    distances[b] += matrix_norm

        return distances / self.num_inferences


class BLEUVar(AcquisitionFunctionGPT):
    def __init__(
        self,
        multiple_inference: bool = True,
        num_inferences: int = hyperparameters.uq.num_inferences,
        alpha: float = 0.6,
    ) -> None:
        super().__init__(multiple_inference, num_inferences, alpha)

    def __call__(
        self,
        hypothesis: List[List[str]],
        inference_results: List[AutoregressiveInferenceResultsGPT],
    ) -> torch.Tensor:
        batch = len(hypothesis)
        bleu_distances = torch.zeros(batch).to(hyperparameters.device)
        for b in range(batch):
            if len(hypothesis[b]) < self.num_inferences:
                continue
            for i in range(self.num_inferences):
                for j in range(self.num_inferences):
                    if i == j:
                        continue
                    bleu_dist = sacrebleu.sentence_bleu(
                        hypothesis[b][i], [hypothesis[b][j]]
                    ).score
                    bleu_distances[b] += (1 - bleu_dist / 100) ** 2
        if (bleu_distances < 0.001).any():
            logger.warning(f"Very low BLEU distances detected: {bleu_distances}")
        return bleu_distances / self.num_inferences


class BALD(AcquisitionFunctionGPT):
    def __init__(
        self,
        multiple_inference: bool = True,
        num_inferences: int = hyperparameters.uq.num_inferences,
        alpha: float = 0.6,
    ) -> None:
        super().__init__(multiple_inference, num_inferences, alpha)

    def __call__(
        self,
        hypothesis: List[List[str]],
        inference_results: List[AutoregressiveInferenceResultsGPT],
    ) -> torch.Tensor:
        """
        Calculate the Bayesian Active Learning by Disagreement (BALD) score for a batch of sequences.
        The BALD score is calculated as the difference between the entropy of the mean softmax probabilities
        and the mean entropy of the softmax probabilities across multiple inferences.

        Args:
            hypothesis: unused
            tgt_tokens: Target tokens used for length penalty
            token_softmax_probs: Softmax probabilities over the whole vocabulary for each token in the sequence

        Returns:
            BALD score for each sequence in the batch
        """
        # token_softmax_probs dim: (batch_size, num_inferences, max_len)
        token_softmax_probs = torch.stack(
            [ir.softmax_probs for ir in inference_results],
            dim=1,
        )
        token_ids = torch.stack([ir.token_ids for ir in inference_results], dim=1)

        # Calculate the mean of the softmax probabilities
        mean_probs = torch.mean(
            token_softmax_probs, dim=1
        )  # (batch_size, max_len, vocab_size)

        # Calculate the entropy of the mean probabilities
        entropy_mean_probs = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-10), dim=-1
        )  # (batch_size, max_len)

        # Calculate the mean entropy of the softmax probabilities
        entropy_probs = -torch.sum(
            token_softmax_probs * torch.log(token_softmax_probs + 1e-10), dim=-1
        )  # (batch_size, num_inferences, max_len)
        mean_entropy_probs = torch.mean(entropy_probs, dim=1)  # (batch_size, max_len)

        # Calculate the BALD score
        bald_score = entropy_mean_probs - mean_entropy_probs  # (batch_size, max_len)

        # Sum the BALD score over the sequence length and normalize by length penalty
        bald_score_sum = torch.sum(bald_score, dim=1)  # (batch_size)

        # Calculate the mean length penalty across all inferences
        length_penalty = torch.mean(
            _length_penalty(token_ids, self.alpha), dim=1
        )  # (batch_size)

        return bald_score_sum / length_penalty
    

class ProbabilityVariance(AcquisitionFunctionGPT):
    """
    Used by Hellaswag. This is class is just for types and aq name.
    """
    def __call__(
        self,
        hypothesis: List[List[str]],
        inference_results: List[AutoregressiveInferenceResultsGPT],
    ) -> torch.Tensor:
        raise NotImplementedError("This is just for types and aq name.")
