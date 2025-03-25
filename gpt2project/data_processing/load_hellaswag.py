from typing import Any, List
from typing_extensions import override
from datasets import load_dataset

from gpt2project.data_processing.abstract_evaluation_dataset import (
    AbstractEvaluationDataset,
    DatasetExample,
    DatasetExampleMultipleChoice,
    DatasetExampleWithConcepts,
)


class HellaSwag(AbstractEvaluationDataset):

    @property
    def only_first_word(self) -> bool:
        return False

    @property
    def break_on_newline(self) -> bool:
        return False

    @property
    def max_tokens(self) -> int:
        return -1

    def __init__(self) -> None:
        self.dataset = _get_hellaswag_data()

    def __len__(self) -> int:
        return len(self.dataset)

    @override
    def __getitem__(self, idx: int) -> DatasetExampleMultipleChoice:
        return self.dataset[idx]


def _get_hellaswag_data() -> List[DatasetExampleMultipleChoice]:
    # Load the HellaSwag dataset
    dataset: Any = load_dataset("Rowan/hellaswag", split="validation")

    processed_dataset: List[DatasetExampleMultipleChoice] = [
        DatasetExampleMultipleChoice(
            prompt=example["ctx"],
            targets=[example["label"]],
            choice_options=example["endings"],
        )
        for example in dataset
    ]

    return processed_dataset


if __name__ == "__main__":
    dataset = HellaSwag()
    print(len(dataset))
    print(dataset[0])
