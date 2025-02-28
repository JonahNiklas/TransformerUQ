from typing import Any, List, Tuple
from datasets import load_dataset
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


def generate_input_text(words: List[str]) -> str:
    template = f"""Task: Generate a meaningful sentence using the provided words.

Example 1:
Words: dog, run, park.
Sentence: The dog ran happily in the park.

Example 2:
Words: chef, cook, kitchen.
Sentence: The chef cooked a delicious meal in the kitchen.

Now try:
Words: {", ".join(words)}.
Sentence: """
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


def collate_fn(batch: Any) -> Tuple[List[str], List[str], List[str], torch.Tensor]:
    input_texts, concepts_list, target_texts = zip(*batch)
    tokenizer = tiktoken.get_encoding("gpt2")
    # Tokenize each input text
    encodings = [
        torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in input_texts
    ]
    # Pad sequences with a padding value (e.g., 0)
    padded_encodings = pad_sequence(encodings, batch_first=True, padding_value=0)
    return (
        list(input_texts),
        list(concepts_list),
        list(target_texts),
        padded_encodings,
    )


def get_common_gen_dataloader(batch_size: int, shuffle: bool) -> DataLoader:
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

    # Create a DataLoader object with batching
    dataloader = DataLoader(
        commongen_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader
