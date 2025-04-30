from __future__ import annotations
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import pickle
import time
from typing import List
import uuid

import fsspec
import pandas as pd
from gpt2project.data_processing.abstract_evaluation_dataset import (
    AbstractEvaluationDataset,
    DatasetExample,
    DatasetExampleWithConcepts,
)
from google import genai
from google.genai import types
from gpt2project.data_processing.commongen_dataset import CommonGen
from dotenv import load_dotenv

from gpt2project.data_processing.lambada_dataset import Lambada
from gpt2project.data_processing.squad_dataset import Squad
from gpt2project.data_processing.triviaqa_dataset import TriviaQA
from google.genai.types import CreateBatchJobConfig

load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class TranslatedDataset(AbstractEvaluationDataset):

    def __init__(self, dataset: AbstractEvaluationDataset) -> None:
        super().__init__()
        self.original_dataset = dataset

        cache_path = Path("local/translated_benchmarks")
        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_path / f"{dataset.__class__.__name__}.pkl"
        if self.cache_path.exists():
            logger.warning(
                f"Loading translated {dataset.__class__.__name__} dataset from cache: {self.cache_path}"
            )
            self.translated_dataset: (
                List[DatasetExample] | List[DatasetExampleWithConcepts]
            ) = pickle.load(self.cache_path.open("rb"))
            return

        self.translator = DatasetTranslator()
        logger.info(
            f"Translating dataset: {dataset.__class__.__name__} using {self.translator.model_name}"
        )

        estimated_cost = self.translator.estimate_cost(dataset)
        logger.info(f"Estimated cost to translate dataset: ${estimated_cost:.2f}")

        confirmation = input("Do you want to continue? (y/n)")
        if confirmation != "y":
            raise ValueError("User did not confirm translation")

        self.translated_dataset = self.translator.translate(dataset)
        pickle.dump(self.translated_dataset, self.cache_path.open("wb"))

    @property
    def only_first_word(self) -> bool:
        return self.original_dataset.only_first_word

    @property
    def break_on_newline(self) -> bool:
        return self.original_dataset.break_on_newline

    @property
    def max_tokens(self) -> int:
        return self.original_dataset.max_tokens

    def get_original_item(self, idx: int) -> DatasetExample:
        return self.original_dataset[idx]

    def __getitem__(self, idx: int) -> DatasetExample:
        assert self.translated_dataset is not None
        return self.translated_dataset[idx]

    def __len__(self) -> int:
        return len(self.translated_dataset)

    def __repr__(self) -> str:
        return f"Translated{self.original_dataset.__class__.__name__}"


class DatasetTranslator:

    def __init__(self, model_name: str = "gemini-2.0-flash-001") -> None:
        self.project_id = "ntnu-ai"
        self.model_name = model_name
        # self.client = genai.Client(
        #     api_key=os.getenv("GEMINI_API_KEY"), vertexai=True
        # )
        self.client = genai.Client(
            vertexai=True, project=self.project_id, location="us-central1"
        )

    def translate(
        self, dataset: AbstractEvaluationDataset
    ) -> List[DatasetExample] | List[DatasetExampleWithConcepts]:
        translated_prompts = self._batch_translate(
            [example.prompt for example in dataset]
        )
        translated_dataset_examples = [
            DatasetExample(prompt=translated_prompt, targets=example.targets)
            for translated_prompt, example in zip(translated_prompts, dataset)
        ]

        return translated_dataset_examples

    def _batch_translate(self, gemini_prompts: List[str]) -> List[str]:
        jsonl_lines = [
            json.dumps(
                {
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": prompt}],
                            },
                        ],
                        "system_instruction": {
                            "parts": [
                                {
                                    "text": "I am translating different benchmarks for use in AI experiments from English to German. I want you to respond with everything by translating the text given from English to German. Do not respond with anything else than the translation. The text given often contains another task, just response with that task translated word for word. Always translate everything in full and every single word!"
                                }
                            ],
                        },
                        "generationConfig": {"temperature": 0.0},
                    },
                }
            )
            for prompt in gemini_prompts
        ]
        bucket_name = "gpt2project-gemini-batch-jobs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gcs_bucket = f"gs://{self.project_id}/{bucket_name}"
        gcs_input_uri = f"{gcs_bucket}/batch_job_input_{timestamp}.jsonl"
        fs = fsspec.filesystem("gcs")
        with fs.open(gcs_input_uri, "w") as f:
            for line in jsonl_lines:
                f.write(line + "\n")

        gcs_output_uri = f"{gcs_bucket}/batch_job_output_{timestamp}.jsonl"
        gcs_batch_job = self.client.batches.create(
            model=self.model_name,
            src=gcs_input_uri,
            config=CreateBatchJobConfig(dest=gcs_output_uri),
        )

        while (
            gcs_batch_job.state == "JOB_STATE_RUNNING"
            or gcs_batch_job.state == "JOB_STATE_PENDING"
        ):
            time.sleep(10)
            assert gcs_batch_job.name is not None
            gcs_batch_job = self.client.batches.get(name=gcs_batch_job.name)
            logger.info(
                f"Job state: {gcs_batch_job.state}. Pinging again in 10 seconds"
            )

        if gcs_batch_job.state == "JOB_STATE_SUCCEEDED":
            logger.info("GCS batch job succeeded")
            fs = fsspec.filesystem("gcs")
            assert gcs_batch_job.dest is not None
            file_paths = fs.glob(f"{gcs_batch_job.dest.gcs_uri}/*/predictions.jsonl")
            df_raw = pd.read_json(f"gs://{file_paths[0]}", lines=True)
            responses_list = df_raw["response"].tolist()

            unordered_prompts = pd.json_normalize(
                df_raw["request"].tolist(), record_path=["contents", ["parts"]]
            )["text"].tolist()

            def extract_text(responses_list_item: dict, i: int) -> str:
                content = responses_list_item["candidates"][0]["content"]
                if "parts" not in content:
                    logger.error(f"Found no generated text for prompt #{i}")
                    return ""
                generated_text = content["parts"][0]["text"]
                assert isinstance(generated_text, str)
                return generated_text

            unordered_translated_prompts = [
                extract_text(responses_list[i], i) for i in range(len(responses_list))
            ]

            # Order the translated prompts according to the original prompts (gemini_prompts)
            indexes = [
                unordered_prompts.index(original_prompt)
                for original_prompt in gemini_prompts
            ]
            ordered_translated_prompts = [
                unordered_translated_prompts[i] for i in indexes
            ]
            return ordered_translated_prompts
        else:
            logger.error(f"GCS batch job failed with state {gcs_batch_job.state}")
            raise ValueError("GCS batch job failed")

    def estimate_cost(self, dataset: AbstractEvaluationDataset) -> float:
        if self.model_name != "gemini-2.0-flash-001":
            raise ValueError("Only Gemini 2.0 Flash is supported for cost estimation")

        input_cost_per_million_tokens = 0.1
        output_cost_per_million_tokens = 0.4

        num_input_tokens = (
            self.client.models.count_tokens(
                model=self.model_name, contents=[example.prompt for example in dataset]  # type: ignore
            ).total_tokens
            or 0
        )
        num_output_tokens = (
            num_input_tokens  # assume same number of input and output tokens
        )

        total_cost = (num_input_tokens / 1e6) * input_cost_per_million_tokens + (
            num_output_tokens / 1e6
        ) * output_cost_per_million_tokens
        return total_cost


if __name__ == "__main__":
    commongen = CommonGen()
    translated_commongen = TranslatedDataset(commongen)
    print(translated_commongen)

    lambada = Lambada()
    translated_lambada = TranslatedDataset(lambada)
    print(translated_lambada)

    squad = Squad()
    translated_squad = TranslatedDataset(squad)
    print(translated_squad)

    triviaqa = TriviaQA()
    translated_triviaqa = TranslatedDataset(triviaqa)
    print(translated_triviaqa)
