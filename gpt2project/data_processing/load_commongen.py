from typing import Any, List, Tuple
from datasets import load_dataset
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


def generate_input_text(words: List[str]) -> str:
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


class CommonGenDataset(Dataset):
    def __init__(self, dataset: List[dict]):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        item = self.dataset[idx]
        input_text = item["input"]
        concepts = item["concepts"]
        target_text = item["target"]
        return input_text, concepts, target_text


def collate_fn(
    batch: List[Tuple[str, List[str], str]],
) -> Tuple[List[str], List[List[str]], List[str], torch.Tensor]:
    input_texts, concepts_list, target_texts = zip(*batch)
    tokenizer = tiktoken.get_encoding("gpt2")
    # Tokenize each input text
    encodings = [tokenizer.encode(text) for text in input_texts]
    return (
        list(input_texts),
        list(concepts_list),
        list(target_texts),
        torch.tensor(encodings),
    )


def get_common_gen_dataloader() -> (
    DataLoader[Tuple[List[str], List[List[str]], List[List[str]], torch.Tensor]]
):
    # Load the CommonGen dataset
    dataset = load_dataset("common_gen", split="validation")
    merged_dataset = []

    first_example = dataset[0]
    idx = first_example["concept_set_idx"]
    input = generate_input_text(first_example["concepts"])
    target = [first_example["target"]]
    concepts = first_example["concepts"]

    for example in dataset:
        idx_n = example["concept_set_idx"]
        if idx_n == idx:
            target.append(example["target"])
            continue

        merged_dataset.append(
            {
                "concept_set_idx": idx,
                "input": input,
                "target": target,
                "concepts": concepts,
            }
        )
        idx = idx_n
        input = generate_input_text(example["concepts"])
        target = [example["target"]]
        concepts = example["concepts"]

    print("Rows in merged dataset:", len(merged_dataset))

    # Create a dataset object
    commongen_dataset = CommonGenDataset(merged_dataset)

    dataloader = DataLoader(
        commongen_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    return dataloader


if __name__ == "__main__":
    dataloader = get_common_gen_dataloader()
    for batch in dataloader:
        input_texts, concepts_list, target_texts, padded_encodings = batch
        print(input_texts)
        print(concepts_list)
        print(target_texts)
        print(padded_encodings)
        break
