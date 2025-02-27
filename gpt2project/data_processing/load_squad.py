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
    text = answer_dict['text']
    assert isinstance(text, list)
    return text

def get_squad_dataframe(force_new_clean:bool = False) -> pd.DataFrame:

    cleaned_file_path = 'local/gpt-data/squad/validation_cleaned.csv'
    if os.path.exists(cleaned_file_path) and not force_new_clean:
        cleaned_data = pd.read_csv(cleaned_file_path)
        return cleaned_data
    
    data = load_dataset("christti/squad-augmented-v2",split='validation')

    # Convert the dataset to a pandas DataFrame
    df = pd.DataFrame(data)

    # Select the relevant columns
    cleaned_data = df[['context', 'question', 'answers']]

    cleaned_data.loc[:, 'answers'] = cleaned_data['answers'].apply(extract_answers)

    # Save the cleaned data to a new CSV file
    os.makedirs('local/gpt-data/squad', exist_ok=True)
    cleaned_data.to_csv(cleaned_file_path, index=False)
    return cleaned_data

class SquadDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, str, List[str]]:
        item = self.data.iloc[idx]
        context = item['context']
        question = item['question']
        answers = item['answers']
        return context, question, answers

def get_squad_dataloader(batch_size: int, shuffle: bool = True) -> DataLoader:
    dataframe = get_squad_dataframe()
    dataset = SquadDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def create_squad_prompt(context: str, question: str) -> str:
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    return prompt

def create_squad_prompt_batched(contexts: List[str], questions: List[str]) -> List[str]:
    return [create_squad_prompt(contexts[i], questions[i]) for i in range(len(contexts))]

if __name__ == "__main__":
    get_squad_dataframe(force_new_clean=True)