from typing import Any, List, Tuple
from datasets import load_dataset
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


def generate_input_text(question: str, context: str) -> str:
    template = f"""Task: Answer the following question.

Question: {question}
Context: {context}
Answer:"""
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


def collate_fn(batch: Any) -> Tuple[List[str], List[str], List[str], torch.Tensor]:
    input_texts, questions, answers = zip(*batch)
    tokenizer = tiktoken.get_encoding("gpt2")
    # Tokenize each input text
    encodings = [
        torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in input_texts
    ]
    # Pad sequences with a padding value (e.g., 0)
    padded_encodings = pad_sequence(encodings, batch_first=True, padding_value=0)
    return (
        list(input_texts),
        list(questions),
        list(answers),
        padded_encodings,
    )


def get_triviaqa_dataloader(
    batch_size: int, shuffle: bool
) -> DataLoader[Tuple[List[str], List[str], List[str], torch.Tensor]]:
    # Load the TriviaQA dataset
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")
    subset = 1000
    max_hits = 3
    dataset = dataset.select(range(subset))

    processed_dataset = []

    for example in dataset:
        question = example["question"]
        answers = example["answer"]["normalized_aliases"]
        input_text = generate_input_text(
            question, " ".join(example["search_results"]["description"][:max_hits])
        )
        processed_dataset.append(
            {
                "input": input_text,
                "question": question,
                "answer": answers,
            }
        )

    print("Rows in processed dataset:", len(processed_dataset))

    # Create a dataset object
    triviaqa_dataset = TriviaQADataset(processed_dataset)

    # Create a DataLoader object with batching
    dataloader = DataLoader(
        triviaqa_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader


if __name__ == "__main__":
    dataloader = get_triviaqa_dataloader(batch_size=8, shuffle=False)
    for batch in dataloader:
        input_texts, questions, answers, padded_encodings = batch
        print(input_texts)
        print(questions)
        print(answers)
        print(padded_encodings)
        break
