from __future__ import annotations

from dataclasses import dataclass
from typing import List

import tiktoken

from gpt2project.bayesformer_gpt import BayesformerGPT
from gpt2project.data_processing.commongen_dataset import AbstractEvaluationDataset
from gpt2project.gpt2model import GPT
from gpt2project.search_methods_gpt import GPT_search_method
from gpt2project.uq.gpt_aq_funcs import AcquisitionFunctionGPT
from gpt2project.utils.benchmark_eval_funcs import AbstractEval, ConceptUsageEval

@dataclass
class EvaluationRunConfig:
    model: GPT | BayesformerGPT
    tokenizer: tiktoken.Encoding
    run_name: str
    enable_mcdo: bool
    search_method: GPT_search_method
    eval_function: AbstractEval | ConceptUsageEval
    n_batches_to_validate: int
    benchmark_name: str
    dataset: AbstractEvaluationDataset
    stepsize: int
    aq_funcs: List[AcquisitionFunctionGPT]