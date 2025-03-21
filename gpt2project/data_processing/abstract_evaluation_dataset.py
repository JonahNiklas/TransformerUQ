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
