from typing import Any, List, Tuple
from datasets import load_dataset
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


def generate_input_text(question: str, context: str) -> str:
    template = f"""Context: {context}
Q: {question}
A:"""
    return template


class TriviaQADataset(Dataset):
    def __init__(self, dataset: List[dict]):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        item = self.dataset[idx]
        input_text = item["input"]
        question = item["question"]
        targets = item["answer"]
        return input_text, question, targets


def collate_fn(
    batch: List[Tuple[str, str, List[str]]],
) -> Tuple[List[str], List[str], List[str], torch.Tensor]:
    input_texts, questions, answers = zip(*batch)
    tokenizer = tiktoken.get_encoding("gpt2")
    # Tokenize each input text
    encodings = [tokenizer.encode(text) for text in input_texts]
    return (
        list(input_texts),
        list(questions),
        list(answers),
        torch.tensor(encodings),
    )


def get_triviaqa_dataloader(
    shuffle: bool = False,
) -> DataLoader[Tuple[List[str], List[str], List[str], torch.Tensor]]:
    # Load the TriviaQA dataset
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")
    subset = 1000
    max_hits = 3
    dataset = dataset.select(range(subset))

    processed_dataset = [
        {
            "input": generate_input_text(
                example["question"],
                " ".join(example["search_results"]["description"][:max_hits]),
            ),
            "question": example["question"],
            "answer": example["answer"]["normalized_aliases"],
        }
        for example in dataset
    ]

    print("Rows in processed dataset:", len(processed_dataset))

    # Create a dataset object
    triviaqa_dataset = TriviaQADataset(processed_dataset)

    dataloader = DataLoader(
        triviaqa_dataset, batch_size=1, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader


if __name__ == "__main__":
    dataloader = get_triviaqa_dataloader(shuffle=False)
    for batch in dataloader:
        input_texts, questions, answers, padded_encodings = batch
        print(input_texts)
        print(questions)
        print(answers)
        print(padded_encodings)
        break
