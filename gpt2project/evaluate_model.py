from __future__ import annotations
from typing import List
import tiktoken
from gpt2project.data_processing.abstract_evaluation_dataset import (
    AbstractEvaluationDataset,
    DatasetExample,
)
from gpt2project.data_processing.commongen_dataset import CommonGenDataset
from gpt2project.data_processing.lambada_dataset import LambadaDataset
from gpt2project.gpt2_generate import (
    generate_for_entire_dataset,
)
from gpt2project.search_methods_gpt import (
    GPT_search_method,
    greedy_search_gpt,
)
from gpt2project.utils.benchmark_eval_funcs import (
    AbstractEval,
    BLEU_eval,
    ConceptUsageEval,
    F1Eval,
)
from hyperparameters import hyperparameters
from gpt2project.gpt2model import GPT
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def main() -> None:
    # ##### CONFIGURATIONS #####
    dataset: AbstractEvaluationDataset = (
        LambadaDataset()
    )  # or LambadaDataset() or TriviaQADataset() or SquadDataset()
    eval_function: AbstractEval = F1Eval()
    search_method: GPT_search_method = greedy_search_gpt
    model = GPT.from_pretrained("gpt2").to(hyperparameters.device)
    # ###########################
    logger.info(f"Evaluating {model.__class__.__name__} on {dataset.__class__.__name__} using {eval_function.__class__.__name__}")

    tokenizer = tiktoken.get_encoding("gpt2")

    generated_texts: List[str] = generate_for_entire_dataset(
        model,
        dataset,
        tokenizer,
        search_method=search_method,
    )

    _print_some_generated_texts(generated_texts, dataset.get_all_examples())

    score = eval_function(generated_texts, dataset.get_all_examples())

    logger.info("#" * 30)
    logger.info(f"Average score: {score:.3f}")
    logger.info("#" * 30)
    # Commongen average score 26.02: 3.009
    # Commongen average score 03.03: 7.994
    # Commongen average score 04.03: 8.412


def _print_some_generated_texts(generated_texts: List[str], dataset_examples: List[DatasetExample], num_examples: int = 25) -> None:
    for i in range(num_examples):
        logger.info(f"EXAMPLE {i+1}:")
        print(f"{dataset_examples[i].prompt}\033[94m{generated_texts[i]}\033[0m")
        print(f"Targets: {dataset_examples[i].targets}")
        logger.info("#" * 30)

if __name__ == "__main__":
    main()
