from typing import Any, List, Tuple
import numpy as np
import tiktoken
import torch
import tiktoken
from tqdm import tqdm
from gpt2project.data_processing.load_commongen import get_common_gen_dataloader
from gpt2project.gpt2_generate import (
    generate_autoregressivly_gpt2,
)
from gpt2project.search_methods_gpt import (
    GPT_search_method,
    greedy_search_gpt,
    topk_sampling_gpt,
)
from sacrebleu import corpus_bleu
from gpt2project.utils.decode import decode_token_id_batch
from hyperparameters import hyperparameters
from gpt2project.gpt2model import GPT

import logging

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


class CommongenEval:
    def __call__(
        self,
        output_text: List[str],
        concepts: List[List[str]],
        targets: List[List[str]],
    ) -> float:
        raise NotImplementedError("Evaluation function not implemented.")


class BLEU_eval(CommongenEval):
    def __call__(
        self,
        output_text: List[str],
        concepts: List[List[str]],
        targets: List[List[str]],
    ) -> float:
        # Calculate BLEU score
        bleu = corpus_bleu(output_text, targets)
        return bleu.score


class ConceptUsageEval(CommongenEval):
    def __call__(
        self,
        output_text: List[str],
        concepts: List[List[str]],
        targets: List[List[str]],
    ) -> float:
        scores = []
        for b in range(len(output_text)):
            score = 0
            output_text[b] = output_text[b].lower()
            for c in concepts[b]:
                if c.lower() in output_text[b]:
                    score += 1
            scores.append(score / len(concepts[b]))
        return np.mean(scores).item()


def evaluate_model_batch(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    encoding_tensors: torch.Tensor,
    concepts: List[List[str]],
    targets_texts: List[List[str]],
    eval_function_commongen: CommongenEval,
) -> float:
    # Use the padded encoding tensor directly to generate responses
    outputs = generate_autoregressivly_gpt2(
        model,
        tokenizer,
        encoding_tensors,
        search_method=greedy_search_gpt,
        break_on_newline=True,
        max_tokens=20,
    )
    token_ids = outputs.token_ids

    output_texts = decode_token_id_batch(token_ids, tokenizer)
    score = eval_function_commongen(output_texts, concepts, targets_texts)
    if score == 0:
        logger.debug(f"Output texts: {output_texts}")
        logger.debug(f"Concepts: {concepts}")
        logger.debug(f"Targets texts: {targets_texts}")
    return score


if __name__ == "__main__":
    # Load the GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = tiktoken.get_encoding(model_name)
    model = GPT.from_pretrained(model_name).to(hyperparameters.device)

    dataloader = get_common_gen_dataloader(batch_size=1, shuffle=False)
    n_batch_to_validate = -1

    # Example of iterating through the DataLoader
    outputs = []

    pbar = tqdm(
        enumerate(dataloader),
        desc="Running commongen validation (avg: N/A)",
        total=len(dataloader),
    )
    for i, batch in pbar:
        if i == n_batch_to_validate:
            break
        input_texts, concepts, target_texts, encoding_tensors = batch
        output = evaluate_model_batch(
            model=model,
            tokenizer=tokenizer,
            encoding_tensors=encoding_tensors,
            concepts=concepts,
            targets_texts=target_texts,
            eval_function_commongen=BLEU_eval(),
        )
        outputs.append(output)

        # Update tqdm description with current average
        pbar.set_description(
            f"Running commongen validation (current: {output:.4f}, avg: {np.mean(outputs):.4f})"
        )

    print("Average score: ", np.mean(outputs))
    # Average score:  3.9192467438661587
    # Average score 26.02: 3.009
    # Average score 03.03: 7.994
    # Average score 04.03: 8.412

    # # Example input words
    # words = ["tree", "car", "crash"]
    # input_text = generate_input_text(words)

    # # Generate a sentence using the input words
    # output_sentence = evaluate_model_single_example(input_text)
    # print(output_sentence)
