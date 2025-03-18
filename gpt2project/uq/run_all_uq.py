from __future__ import annotations
import tiktoken
from gpt2project.gpt2model import GPT
from gpt2project.search_methods_gpt import GPT_search_method, greedy_search_gpt
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
from torch import nn
from hyperparameters import hyperparameters
from gpt2project.utils.checkpoint import get_model_from_wandb_checkpoint
from utils.general_plotter import (
    get_gpt_cache_folder,
    get_gpt_cache_path,
    plot_ret_curve,
)
import logging
from typing import Callable, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_evaluation(
    benchmark_name: str,
    get_run_func: Callable,
    eval_functions: List[MultipleTargetEval | KeywordEval],
    n_to_validate: int,
    model_GPT: nn.Module,
    gpt_model_name: str,
    model_BayesGPT: nn.Module,
    bayesgpt_model_name: str,
    tokenizer: tiktoken.Encoding,
    run_name: str,
    enable_mcdo: bool,
    search_method: GPT_search_method,
    stepsize: int,
) -> None:
    logger.info(benchmark_name)
    for eval_function in eval_functions:
        for model in [model_GPT, model_BayesGPT]:
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
                get_gpt_cache_path(
                    benchmark_name=benchmark_name,
                    model_name=gpt_model_name,
                    enable_mcdo=enable_mcdo,
                    search_method=search_method.__name__,
                    run_name=run_name,
                    eval_function_name=eval_function.__class__.__name__,
                    n_batch_to_validate=n_to_validate,
                    stepsize=stepsize,
                ),
                get_gpt_cache_path(
                    benchmark_name=benchmark_name,
                    model_name=bayesgpt_model_name,
                    enable_mcdo=enable_mcdo,
                    search_method=search_method.__name__,
                    run_name=run_name,
                    eval_function_name=eval_function.__class__.__name__,
                    n_batch_to_validate=n_to_validate,
                    stepsize=stepsize,
                ),
            ],
            title=benchmark_name,
            save_filepath=get_gpt_cache_folder(
                benchmark_name,
                gpt_model_name,
                enable_mcdo,
                search_method.__name__,
            )
            + f"/{benchmark_name.lower()}_combined_retcurve_{run_name}_{eval_function.__class__.__name__}_n{n_to_validate}_step{stepsize}.svg",
        )


if __name__ == "__main__":
    # Load GPT and BayesGPT
    tokenizer = tiktoken.get_encoding("gpt2")
    model_BayesGPT = get_model_from_wandb_checkpoint(
        wandb_artifact_path="sondresorbye-magson/GPT2Project/model-checkpoint-76291:v2",
        checkpoint_name="model_bayesformer_76291.pt",
    )
    model_BayesGPT.to(hyperparameters.device)
    model_BayesGPT.eval()

    model_GPT = get_model_from_wandb_checkpoint(
        wandb_artifact_path="sondresorbye-magson/GPT2Project/model-checkpoint-76291:v1",
        checkpoint_name="model_transformer_76291.pt",
    )
    model_GPT.to(hyperparameters.device)
    model_GPT.eval()

    run_name = "sbatch_r1"
    gpt_model_name = "GPT"
    bayesgpt_model_name = "BayesGPT"
    enable_mcdo = True
    search_method = greedy_search_gpt
    step_size = 25

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
                model_GPT,
                gpt_model_name,
                model_BayesGPT,
                bayesgpt_model_name,
                tokenizer,
                run_name,
                enable_mcdo,
                search_method,
                step_size,
            )
