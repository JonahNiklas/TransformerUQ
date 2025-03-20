from __future__ import annotations
from typing import List
import tiktoken
from gpt2project.data_processing.abstract_evaluation_dataset import (
    AbstractEvaluationDataset,
)
from gpt2project.data_processing.commongen_dataset import CommonGenDataset
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
)
from hyperparameters import hyperparameters
from gpt2project.gpt2model import GPT
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def main() -> None:
    # ##### CONFIGURATIONS #####
    dataset: AbstractEvaluationDataset = (
        CommonGenDataset()
    )  # or LambadaDataset() or TriviaQADataset() or SquadDataset()
    eval_function: AbstractEval = BLEU_eval()
    max_tokens: int = 20
    only_first_word: bool = False
    break_on_newline: bool = True
    search_method: GPT_search_method = greedy_search_gpt
    # ###########################

    model = GPT.from_pretrained("gpt2").to(hyperparameters.device)
    tokenizer = tiktoken.get_encoding("gpt2")

    generated_texts: List[str] = generate_for_entire_dataset(
        model,
        dataset,
        tokenizer,
        search_method=search_method,
        break_on_newline=break_on_newline,
        only_first_word=only_first_word,
        max_tokens=max_tokens,
    )

    score = eval_function(generated_texts, dataset.get_all_examples())

    logger.info("#" * 30)
    logger.info("Average score: ", score)
    logger.info("#" * 30)
    # Commongen average score 26.02: 3.009
    # Commongen average score 03.03: 7.994
    # Commongen average score 04.03: 8.412


if __name__ == "__main__":
    main()
