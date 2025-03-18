from __future__ import annotations
from ast import Call
from regex import B
from sympy import plot
import tiktoken
from gpt2project.search_methods_gpt import greedy_search_gpt
from gpt2project.uq.gpt2_squad_uq import get_squad_run
from gpt2project.uq.gpt_commongen_uq import get_commongen_run
from gpt2project.uq.gpt_lambada_uq import get_lambada_run
from gpt2project.uq.gpt_trivia_uq import get_triviaqa_run
from gpt2project.utils.benchmark_eval_funcs import (
    BLEU_eval,
    ConceptUsageEval,
    F1Eval,
    KeywordEval,
    MultipleTargetEval,
    TargetUsageEval,
)
from hyperparameters import hyperparameters
from gpt2project.utils.checkpoint import get_model_from_wandb_checkpoint
from utils.general_plotter import plot_ret_curve
import logging
from typing import Callable, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_evaluation(
    benchmark_name: str,
    get_run_func: Callable,
    eval_functions: List[MultipleTargetEval | KeywordEval],
    n_to_validate: int,
    model_BayesGPT: object,
    model_GPT: object,
    tokenizer: object,
    run_name: str,
    enable_mcdo: bool,
    search_method: Callable,
) -> None:
    logger.info(benchmark_name)
    for model in [model_BayesGPT, model_GPT]:
        for eval_function in eval_functions:
            get_run_func(
                model,
                tokenizer,
                run_name,
                enable_mcdo,
                search_method,
                eval_function=eval_function,
                n_batch_to_validate=n_to_validate,
            )
            plot_ret_curve(
                plot_data_paths=[
                    f"local/gpt-results/{benchmark_name.lower()}/GPT/mcdo{enable_mcdo}/{search_method.__name__}/plot_data_run1_{eval_function.__class__.__name__}_n{n_to_validate}_step25.pt",
                    f"local/gpt-results/{benchmark_name.lower()}/BayesGPT/mcdo{enable_mcdo}/{search_method.__name__}/plot_data_run1_{eval_function.__class__.__name__}_n{n_to_validate}_step25.pt",
                ],
                title=benchmark_name,
                save_filepath=f"local/gpt-results/{benchmark_name.lower()}/GPT/mcdo{enable_mcdo}/{search_method.__name__}/{benchmark_name.lower()}_combined_retcurve_{run_name}_{eval_function.__class__.__name__}_n{n_to_validate}_step25.svg",
            )


if __name__ == "__main__":
    # Load GPT and BayesGPT
    tokenizer = tiktoken.get_encoding("gpt2")
    model_BayesGPT = get_model_from_wandb_checkpoint(
        wandb_artifact_path="sondresorbye-magson/GPT2Project/model-checkpoint-76291:v2",
        checkpoint_name="model_bayesformer_76291.pt",
        remove_orig_prefix=True,
    )
    model_BayesGPT.to(hyperparameters.device)
    model_BayesGPT.eval()

    model_GPT = get_model_from_wandb_checkpoint(
        wandb_artifact_path="sondresorbye-magson/GPT2Project/model-checkpoint-76291:v1",
        checkpoint_name="model_transformer_76291.pt",
        remove_orig_prefix=True,
    )
    model_GPT.to(hyperparameters.device)
    model_GPT.eval()

    run_name = "sbatch_r1"
    enable_mcdo = True
    search_method = greedy_search_gpt

    tasks: List[
        Tuple[str, Callable, List[int], List[MultipleTargetEval | KeywordEval]]
    ] = [
        ("LAMBADA", get_lambada_run, [-1], [F1Eval()]),
        ("TriviaQA", get_triviaqa_run, [-1], [TargetUsageEval()]),
        ("CommonGen", get_commongen_run, [-1], [BLEU_eval(), ConceptUsageEval()]),
        ("SQuAD", get_squad_run, [1000], [TargetUsageEval(), BLEU_eval()]),
    ]

    for task_name, get_run_func, n_to_validate_list, eval_functions in tasks:
        for n_to_validate in n_to_validate_list:
            run_evaluation(
                task_name,
                get_run_func,
                eval_functions,
                n_to_validate,
                model_BayesGPT,
                model_GPT,
                tokenizer,
                run_name,
                enable_mcdo,
                search_method,
            )
