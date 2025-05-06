from __future__ import annotations

from typing import List

import numpy as np
from sacrebleu import corpus_bleu

from gpt2project.data_processing.abstract_evaluation_dataset import (
    AbstractEvaluationDataset,
    DatasetExample,
    DatasetExampleWithConcepts,
)
from gpt2project.utils.list_type_assertion import assert_list_element_type


class AbstractEval:
    def __call__(
        self,
        output_text: List[str],
        dataset_examples: List[DatasetExample | DatasetExampleWithConcepts],
    ) -> float:
        raise NotImplementedError("Evaluation function not implemented.")


class MultipleTargetEval(AbstractEval):
    def __call__(
        self,
        output_text: List[str],
        dataset_examples: List[DatasetExample | DatasetExampleWithConcepts],
    ) -> float:
        raise NotImplementedError("Evaluation function not implemented.")


class KeywordEval(AbstractEval):
    def __call__(
        self,
        output_text: List[str],
        dataset_examples: List[DatasetExample | DatasetExampleWithConcepts],
    ) -> float:
        raise NotImplementedError("Evaluation function not implemented.")


class MultipleChoiceEval(AbstractEval):
    def __call__(
        self,
        output_text: List[str],
        dataset_examples: List[DatasetExample | DatasetExampleWithConcepts],
    ) -> float:
        score = 0
        assert len(output_text) == len(dataset_examples)
        for i, example in enumerate(dataset_examples):
            if example.targets[0].lower() == output_text[i].lower():
                score += 1
        return score / len(dataset_examples)


class TargetUsageEval(MultipleTargetEval):
    # Evaluate the model based on the presence of the target in the output
    # score is 1 if any of the targets is present in the output
    def __call__(
        self,
        output_text: List[str],
        dataset_examples: List[DatasetExample | DatasetExampleWithConcepts],
    ) -> float:
        targets = [example.targets for example in dataset_examples]
        scores = [0.0] * len(output_text)
        for i in range(len(output_text)):
            assert isinstance(targets[i], list), f"targets[{i}] is not a list"
            for t in targets[i]:
                if t.lower() in output_text[i].lower():
                    scores[i] = 1.0
                    break
        return sum(scores) / len(scores)


class SingleTargetEval:
    def __call__(
        self,
        output_text: List[str],
        target: List[str],
    ) -> float:
        raise NotImplementedError("Evaluation function not implemented.")


class F1Eval(MultipleTargetEval):
    def __call__(
        self,
        output_text: List[str],
        dataset_examples: List[DatasetExample | DatasetExampleWithConcepts],
    ) -> float:
        """
        Evaluate the model based on the F1 score between the output and the target.

        Args:
            output_text (List[str]): List of generated output texts.
            dataset_examples (List[DatasetExample]): List of dataset examples.

        Returns:
            float: Average F1 score.
        """
        targets = [example.targets for example in dataset_examples]
        scores: List[float] = []
        for b in range(len(output_text)):
            output_text[b] = output_text[b].lower()
            target_b = targets[b][0].lower()
            tp = len(set(output_text[b].split()) & set(target_b.split()))
            fp = len(set(output_text[b].split()) - set(target_b.split()))
            fn = len(set(target_b.split()) - set(output_text[b].split()))
            if tp == 0:
                scores.append(0)
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                scores.append(2 * precision * recall / (precision + recall))
        return np.mean(scores).item()


class BLEU_eval(MultipleTargetEval):
    def __call__(
        self,
        output_text: List[str],
        dataset_examples: List[DatasetExample | DatasetExampleWithConcepts],
    ) -> float:
        # Calculate BLEU score
        targets_transposed = AbstractEvaluationDataset.get_all_targets_transposed(
            assert_list_element_type(dataset_examples, DatasetExampleWithConcepts)
        )
        bleu = corpus_bleu(output_text, targets_transposed)
        return bleu.score


class ConceptUsageEval(AbstractEval):

    def __call__(
        self,
        output_text: List[str],
        _dataset_examples: List[DatasetExample | DatasetExampleWithConcepts],
    ) -> float:
        """
        Evaluate the model based on the presence of the concepts in the output.

        Args:
            output_text (List[str]): List of generated output texts.
            dataset_examples (List[DatasetExample]): List of dataset examples.

        Returns:
            float: Average concept usage score.
        """
        dataset_examples: List[DatasetExampleWithConcepts] = assert_list_element_type(
            _dataset_examples, DatasetExampleWithConcepts
        )
        concepts = [example.concepts for example in dataset_examples]
        scores = []
        for b in range(len(output_text)):
            score = 0
            output_text[b] = output_text[b].lower()
            for c in concepts[b]:
                if c.lower() in output_text[b]:
                    score += 1
            scores.append(score / len(concepts[b]))
        return np.mean(scores).item()


if __name__ == "__main__":
    f1_eval = F1Eval()
    output_text = ["hello", "worrd", "world"]
    target = [["hello"], ["world"], ["world"]]
    dataset_examples = [DatasetExample(targets=target, prompt="") for target in target]
    score = f1_eval(output_text, dataset_examples)
    print(score)
