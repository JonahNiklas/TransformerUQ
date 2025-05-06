from typing import Any, List, Tuple
from datasets import load_dataset

from gpt2project.data_processing.abstract_evaluation_dataset import (
    AbstractEvaluationDataset,
    DatasetExample,
)


class TriviaQA(AbstractEvaluationDataset):

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
        self.dataset = _get_triviaqa_data()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DatasetExample:
        return self.dataset[idx]


def _generate_prompt(question: str, context: str) -> str:
    template = f"""Context: {context}
Q: {question}
A:"""
    return template


def _get_triviaqa_data() -> List[DatasetExample]:
    # Load the TriviaQA dataset
    dataset: Any = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")
    subset = 1000
    max_hits = 3
    dataset = dataset.select(range(subset))

    processed_dataset: List[DatasetExample] = [
        DatasetExample(
            prompt=_generate_prompt(
                example["question"],
                " ".join(example["search_results"]["description"][:max_hits]),
            ),
            targets=example["answer"]["normalized_aliases"],
        )
        for example in dataset
    ]
    return processed_dataset


if __name__ == "__main__":
    dataset = TriviaQA()
    print(len(dataset))
    print(dataset[0])
