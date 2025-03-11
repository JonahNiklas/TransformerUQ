from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


def extract_answers(answer_dict: dict) -> List[str]:
    text = answer_dict["text"]
    assert isinstance(text, list)
    return text


def get_squad_data() -> Tuple[List[str], List[str], List[List[str]]]:
    data = load_dataset("christti/squad-augmented-v2", split="validation")

    # Extract the relevant columns and apply extract_answers
    contexts = data["context"]
    questions = data["question"]
    answers = [extract_answers(answer) for answer in data["answers"]]

    return list(contexts), list(questions), answers


class SquadDataset(Dataset):
    def __init__(self) -> None:
        contexts, questions, answers = get_squad_data()
        self.contexts = contexts
        self.questions = questions
        self.answers = answers

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> Tuple[str, str, List[str]]:
        return self.contexts[idx], self.questions[idx], self.answers[idx]


def collate_fn(
    batch: List[Tuple[str, str, List[str]]],
) -> Tuple[List[str], List[str], List[List[str]]]:
    contexts, questions, answers = zip(*batch)
    return list(contexts), list(questions), list(answers)


def get_squad_dataloader(
    batch_size: int, shuffle: bool = True
) -> DataLoader[Tuple[List[str], List[str], List[List[str]]]]:
    dataset = SquadDataset()
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader


def create_squad_prompt(context: str, question: str) -> str:
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    return prompt


def create_squad_prompt_batched(contexts: List[str], questions: List[str]) -> List[str]:
    return [
        create_squad_prompt(contexts[i], questions[i]) for i in range(len(contexts))
    ]
