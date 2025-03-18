from typing import List

import numpy as np
from sacrebleu import corpus_bleu


class MultipleTargetEval:
    def __call__(self, output_text: List[str], targets: List[List[str]]) -> float:
        raise NotImplementedError("Evaluation function not implemented.")


class KeywordEval:
    def __call__(self, output_text: List[str], concepts: List[List[str]]) -> float:
        raise NotImplementedError("Evaluation function not implemented.")


class TargetUsageEval(MultipleTargetEval):
    # Evaluate the model based on the presence of the target in the output
    # score is 1 if any of the targets is present in the output
    def __call__(self, output_text: List[str], targets: List[List[str]]) -> float:
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
        target: List[List[str]],
    ) -> float:
        """
        Evaluate the model based on the F1 score between the output and the target.

        Args:
            output_text (List[str]): List of generated output texts.
            target (List[List[str]]): List of target texts. All inner lists have length 1.

        Returns:
            float: Average F1 score.
        """
        scores: List[float] = []
        for b in range(len(output_text)):
            output_text[b] = output_text[b].lower()
            target_b = target[b][0].lower()
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
        targets: List[List[str]],
    ) -> float:
        # Calculate BLEU score
        bleu = corpus_bleu(output_text, targets)
        return bleu.score


class ConceptUsageEval(KeywordEval):
    def __call__(
        self,
        output_text: List[str],
        concepts: List[List[str]],
    ) -> float:
        """
        Evaluate the model based on the presence of the concepts in the output.

        Args:
            output_text (List[str]): List of generated output texts.
            concepts (List[List[str]]): List of concepts.

        Returns:
            float: Average concept usage score.
        """
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
    score = f1_eval(output_text, target)
    print(score)
