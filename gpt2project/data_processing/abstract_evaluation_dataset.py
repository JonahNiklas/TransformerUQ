from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator, List
from pydantic import BaseModel
from torch.utils.data import Dataset


class DatasetExample(BaseModel):
    prompt: str
    targets: List[str]


class DatasetExampleWithConcepts(DatasetExample):
    concepts: List[str]


class DatasetExampleMultipleChoice(DatasetExample):
    choice_options: List[str]


class AbstractEvaluationDataset(Dataset, ABC):

    @property
    @abstractmethod
    def only_first_word(self) -> bool:
        """Whether to only predict the first word."""
        pass

    @property
    @abstractmethod
    def break_on_newline(self) -> bool:
        """Whether to break prompts on newlines."""
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Maximum number of tokens to predict."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> DatasetExample:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator[DatasetExample]:
        for i in range(len(self)):
            yield self[i]

    def get_all_examples(self) -> List[DatasetExample]:
        return [self[i] for i in range(len(self))]

    @staticmethod
    def get_all_targets_transposed(examples: List[DatasetExampleWithConcepts]) -> List[List[str]]:
        """
        Returns all targets in a transposed manner to be directly used by sacrebleu.corpus_bleu.
        The first dimension corresponds to one set of targets for the entire dataset.
        The second dimension corresponds to the targets for each example in the dataset.
        For cases with variable numbers of targets per example, None is inserted.

        For more information, see: https://github.com/mjpost/sacrebleu?tab=readme-ov-file#variable-number-of-references
        """
        targets = [example.targets for example in examples]
        num_targets = max(len(target) for target in targets)
        transposed_targets: List[List[str]] = [
            [""] * len(targets) for _ in range(num_targets)
        ]
        for i, target in enumerate(targets):
            for j, target_j in enumerate(target):
                transposed_targets[j][i] = target_j

        assert all(
            len(targets) == len(transposed_targets[i]) for i in range(num_targets)
        )

        return transposed_targets
