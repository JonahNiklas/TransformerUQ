from typing import Any, List, Tuple
import numpy as np
import tiktoken
import torch
import tiktoken
from tqdm import tqdm
from gpt2project.data_processing.load_commongen import get_common_gen_dataloader
from gpt2project.gpt2_generate import (
    generate_autoregressivly_gpt2,
    generate_autoregressivly_gpt2_with_uq,
    generate_karpathy,
    karpathy,
)
from gpt2project.search_methods_gpt import (
    GPT_search_method,
    greedy_search_gpt,
    topk_sampling_gpt,
)
from sacrebleu import corpus_bleu
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
        max_tokens=20,
    )
    token_ids = outputs.token_ids

    output_texts = _clean_and_decode_output_tokens(token_ids, tokenizer)
    score = eval_function_commongen(output_texts, concepts, targets_texts)
    if score == 0:
        logger.debug(f"Output texts: {output_texts}")
        logger.debug(f"Concepts: {concepts}")
        logger.debug(f"Targets texts: {targets_texts}")
    return score


def _clean_and_decode_output_tokens(
    token_ids: torch.Tensor,
    tokenizer: tiktoken.Encoding,
) -> List[str]:
    remove_prefix_tokens = [
        tokenizer.encode("\n")[0],
        tokenizer.encode("~")[0],
        tokenizer.encode("~~")[0],
        tokenizer.encode(" ")[0],
    ]
    new_line_token = tokenizer.encode("\n")[0]
    non_breaking_space_token = tokenizer.encode("\xa0")[0]
    
    output_texts = []
    # clean the output tokens and decode them
    for b in range(len(token_ids)):  # iterate over batch
        ids = token_ids[b][len(encoding_tensors[b]) :]
        ids = ids[ids != non_breaking_space_token]
        while ids[0] in remove_prefix_tokens:
            ids = ids[1:]
        if new_line_token in ids:
            ids = ids[: ids.tolist().index(new_line_token)]
        output_texts.append(tokenizer.decode(ids.tolist()))
    return output_texts


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
            eval_function_commongen=ConceptUsageEval(),
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

    # # Example input words
    # words = ["tree", "car", "crash"]
    # input_text = generate_input_text(words)

    # # Generate a sentence using the input words
    # output_sentence = evaluate_model_single_example(input_text)
    # print(output_sentence)
