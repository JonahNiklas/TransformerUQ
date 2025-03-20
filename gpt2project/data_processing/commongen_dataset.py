from typing import Any, List
from datasets import load_dataset

from gpt2project.data_processing.abstract_evaluation_dataset import (
    AbstractEvaluationDataset,
    DatasetExampleWithConcepts,
)


def _get_prompt(words: List[str]) -> str:
    template = f"""Task: Generate a meaningful sentence using the provided words.

Example 1:
Words: field, look, stand.
Sentence: The player stood in the field looking at the batter.

Example 2:
Words: climb, building, side.
Sentence: I climbed the side of the building.

Now try:
Words: {", ".join(words)}.
Sentence:"""
    return template


class CommonGenDataset(AbstractEvaluationDataset):
    def __init__(self) -> None:
        self.dataset = _get_common_gen_data()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DatasetExampleWithConcepts:
        return self.dataset[idx]


def _get_common_gen_data() -> List[DatasetExampleWithConcepts]:
    # Load the CommonGen dataset
    dataset: Any = load_dataset("common_gen", split="validation")
    merged_dataset: List[DatasetExampleWithConcepts] = []

    for example in dataset:
        idx_n = int(example["concept_set_idx"])
        concepts = example["concepts"]
        assert isinstance(concepts, List)
        assert isinstance(concepts[0], str)
        target = example["target"]
        assert isinstance(target, str)
        prompt = _get_prompt(concepts)

        if idx_n > len(merged_dataset) - 1:
            merged_dataset.append(
                DatasetExampleWithConcepts(
                    prompt,
                    [target],
                    concepts,
                )
            )
            continue

        merged_dataset[idx_n].targets.append(target)

    print("Rows in merged dataset:", len(merged_dataset))
    return merged_dataset


if __name__ == "__main__":
    dataset = CommonGenDataset()
    print(len(dataset))
    print(dataset[0])
