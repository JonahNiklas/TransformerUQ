from typing import Any, List

from datasets import load_dataset

from gpt2project.data_processing.abstract_evaluation_dataset import (
    AbstractEvaluationDataset,
    DatasetExample,
)


class LambadaDataset(AbstractEvaluationDataset):
    def __init__(self) -> None:
        self.dataset = _get_lambada_data()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DatasetExample:
        return self.dataset[idx]


def _get_lambada_data() -> List[DatasetExample]:
    dataset: Any = load_dataset("cimec/lambada", split="validation")
    subset = 1000
    dataset = dataset.select(range(subset))

    processed_dataset: List[DatasetExample] = [
        DatasetExample(
            " ".join(example["text"].split()[:-1]),
            example["text"].split()[-1],
        )
        for example in dataset
    ]
    return processed_dataset
