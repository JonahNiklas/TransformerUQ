from typing import List, Tuple
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# deprecated
# def extract_answers(answer_str: str) -> List[str]:
#     match = re.search(r"array\((\[.*?\]),\s*dtype", answer_str, re.DOTALL)
#     if match:
#         text_array = ast.literal_eval(match.group(1))
#         assert isinstance(text_array, list)
#         return text_array

#     return []


def extract_answers(answer_dict: dict) -> List[str]:
    text = answer_dict["text"]
    assert isinstance(text, list)
    return text


def get_squad_dataframe(force_new_clean: bool = False) -> pd.DataFrame:

    cleaned_file_path = "local/gpt-data/squad/validation_cleaned.csv"
    if not os.path.exists(cleaned_file_path) or force_new_clean:
        data = load_dataset("christti/squad-augmented-v2", split="validation")

        # Convert the dataset to a pandas DataFrame
        df = pd.DataFrame(data)

        # Select the relevant columns
        cleaned_data = df[["context", "question", "answers"]]

        cleaned_data.loc[:, "answers"] = cleaned_data["answers"].apply(extract_answers)

        # Save the cleaned data to a new CSV file
        os.makedirs("local/gpt-data/squad", exist_ok=True)
        cleaned_data.to_csv(cleaned_file_path, index=False)

    cleaned_data = pd.read_csv(cleaned_file_path)
    cleaned_data["answers"] = cleaned_data["answers"].apply(eval)
    return cleaned_data


class SquadDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, str, List[str]]:
        item = self.data.iloc[idx]
        context = item["context"]
        question = item["question"]
        answers = item["answers"]
        return context, question, answers


def collate_fn(
    batch: List[Tuple[str, str, List[str]]]
) -> Tuple[List[str], List[str], List[List[str]]]:
    contexts, questions, answers = zip(*batch)
    return list(contexts), list(questions), list(answers)


def get_squad_dataloader(
    batch_size: int, shuffle: bool = True, force_new_clean: bool = False
) -> DataLoader:
    dataframe = get_squad_dataframe(force_new_clean)
    dataset = SquadDataset(dataframe)
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


class SquadEval:
    def __call__(self, output_text: List[str], targets: List[List[str]]) -> float:
        raise NotImplementedError("Evaluation function not implemented.")


class TargetUsageEval(SquadEval):
    # Evaluate the model based on the presence of the target in the output
    # score is 1 if any of the targets is present in the output
    def __call__(self, output_text: List[str], targets: List[List[str]]) -> float:
        scores = [0.0] * len(output_text)
        for i in range(len(output_text)):
            for t in targets[i]:
                if t in output_text[i]:
                    scores[i] = 1.0
                    break
        return sum(scores) / len(scores)


if __name__ == "__main__":
    get_squad_dataframe(force_new_clean=True)
