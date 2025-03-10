from typing import Any, List, Tuple
from datasets import load_dataset
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


class LAMBADADataset(Dataset):
    def __init__(self, dataset: List[dict]):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        item = self.dataset[idx]
        input_text = item["input"]
        target = item["target"]
        return input_text, target


def collate_fn(batch: Any) -> Tuple[List[str], List[str], List[str], torch.Tensor]:
    input_texts, targets = zip(*batch)
    tokenizer = tiktoken.get_encoding("gpt2")
    # Tokenize each input text
    encodings = [
        torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in input_texts
    ]
    # Pad sequences with a padding value (e.g., 0)
    padded_encodings = pad_sequence(encodings, batch_first=True, padding_value=0)
    return (
        list(input_texts),
        list(targets),
        padded_encodings,
    )


def get_lambada_dataloader(
    batch_size: int, shuffle: bool
) -> DataLoader[Tuple[List[str], List[str], List[str], torch.Tensor]]:
    # Load the LAMBADA dataset
    dataset = load_dataset("cimec/lambada", split="validation")
    subset = 1000
    dataset = dataset.select(range(subset))

    processed_dataset = []

    for example in dataset:
        processed_dataset.append(
            {
                "input": " ".join(example["text"].split()[:-1]),
                "target": example["text"].split()[-1],
            }
        )

    print("Rows in processed dataset:", len(processed_dataset))

    # Create a dataset object
    lambada_dataset = LAMBADADataset(processed_dataset)

    # Create a DataLoader object with batching
    dataloader = DataLoader(
        lambada_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader


if __name__ == "__main__":
    dataloader = get_lambada_dataloader(batch_size=8, shuffle=False)
    for batch in dataloader:
        input_texts, targets, padded_encodings = batch
        print(input_texts)
        print(targets)
        print(padded_encodings)
        break
