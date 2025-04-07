from __future__ import annotations
from typing import List, Any
from torch.utils.data import Dataset
from datasets import load_dataset

from gpt2project.data_processing.abstract_evaluation_dataset import (
    AbstractEvaluationDataset,
    DatasetExample,
)


class Squad(AbstractEvaluationDataset):

    @property
    def only_first_word(self) -> bool:
        return False

    @property
    def break_on_newline(self) -> bool:
        return True

    @property
    def max_tokens(self) -> int:
        return 20

    def __init__(self) -> None:
        self.dataset = _get_squad_data()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DatasetExample:
        return self.dataset[idx]


def _extract_answers(answer_dict: Any) -> List[str]:
    text = answer_dict["text"]
    assert isinstance(text, list)
    return text


def _get_squad_data() -> List[DatasetExample]:
    data: Any = load_dataset(
        "rajpurkar/squad",
        split="validation",
    )
    subset = 1000
    data = data.select(range(subset))

    contexts = [str(context) for context in data["context"]]
    questions = [str(question) for question in data["question"]]
    answers = [_extract_answers(answer) for answer in data["answers"]]

    processed_dataset: List[DatasetExample] = [
        DatasetExample(
            prompt=_create_squad_prompt(context, question),
            targets=answer,
        )
        for context, question, answer in zip(contexts, questions, answers)
    ]

    return processed_dataset


def _create_squad_prompt(context: str, question: str) -> str:
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    return prompt


if __name__ == "__main__":
    dataset = Squad()
    print(len(dataset))
    print(dataset[0])
