from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterator, List
from torch.utils.data import Dataset

@dataclass
class DatasetExample():
    prompt: str
    targets: List[str]

@dataclass
class DatasetExampleWithConcepts(DatasetExample):
    concepts: List[str]

class AbstractEvaluationDataset(Dataset):
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

