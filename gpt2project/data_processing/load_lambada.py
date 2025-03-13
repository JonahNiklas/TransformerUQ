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


def collate_fn(
    batch: List[Tuple[str, str]],
) -> Tuple[List[str], List[str], torch.Tensor]:
    input_texts, targets = zip(*batch)
    tokenizer = tiktoken.get_encoding("gpt2")
    # Tokenize each input text
    encodings = [tokenizer.encode(text) for text in input_texts]
    return (
        list(input_texts),
        list(targets),
        torch.tensor(encodings),
    )


def get_lambada_dataloader(
    shuffle: bool,
) -> DataLoader[Tuple[List[str], List[str], List[str], torch.Tensor]]:
    # Load the LAMBADA dataset
    dataset = load_dataset("cimec/lambada", split="validation")
    subset = 1000
    dataset = dataset.select(range(subset))

    processed_dataset = [
        {
            "input": " ".join(example["text"].split()[:-1]),
            "target": example["text"].split()[-1],
        }
        for example in dataset
    ]

    # Create a dataset object
    lambada_dataset = LAMBADADataset(processed_dataset)

    dataloader = DataLoader(
        lambada_dataset, batch_size=1, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader


if __name__ == "__main__":
    dataloader = get_lambada_dataloader(shuffle=False)
    for batch in dataloader:
        input_texts, targets, padded_encodings = batch
        print(input_texts)
        print(targets)
        print(padded_encodings)
        break
