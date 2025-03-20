from __future__ import annotations

import logging
from typing import List, Tuple

import tiktoken

from gpt2project.data_processing.abstract_evaluation_dataset import (
    AbstractEvaluationDataset,
)
from gpt2project.data_processing.commongen_dataset import CommonGenDataset
from gpt2project.data_processing.lambada_dataset import LambadaDataset
from gpt2project.data_processing.squad_dataset import SquadDataset
from gpt2project.data_processing.triviaqa_dataset import TriviaQADataset
from gpt2project.gpt2_generate import generate_with_uq_for_entire_dataset
from gpt2project.search_methods_gpt import greedy_search_gpt
from gpt2project.uq.calc_plot_data import calc_retention_curve
from gpt2project.uq.evaluation_run_config import EvaluationRunConfig
from gpt2project.uq.gpt_aq_funcs import BALD, BeamScore, BLEUVar
from gpt2project.utils.benchmark_eval_funcs import (
    AbstractEval,
    BLEU_eval,
    ConceptUsageEval,
    F1Eval,
    TargetUsageEval,
)
from gpt2project.utils.checkpoint import get_model_from_wandb_checkpoint
from hyperparameters import hyperparameters
from utils.general_plotter import (
    _get_gpt_plot_data_folder,
    get_gpt_plot_data_path,
    plot_ret_curve,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
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
    enable_mcdo = True
    search_method = greedy_search_gpt
    step_size = 25

    aq_funcs = [BeamScore(), BALD(), BLEUVar()]

    tasks: List[
        Tuple[
            str,
            AbstractEvaluationDataset,
            List[int],
            List[AbstractEval | ConceptUsageEval],
        ]
    ] = [
        ("LAMBADA", LambadaDataset(), [-1], [F1Eval()]),
        # ("TriviaQA", TriviaQADataset(), [-1], [TargetUsageEval()]),
        # (
        #     "CommonGen",
        #     CommonGenDataset(),
        #     [-1],
        #     [BLEU_eval(), ConceptUsageEval()],
        # ),
        # (
        #     "SQuAD",
        #     SquadDataset(),
        #     [1000],
        #     [TargetUsageEval(), BLEU_eval()],
        # ),
        # ("HellaSwag", get_hellaswag_run, [-1], [MultipleChoiceEval()]),
    ]

    for task_name, dataset, n_to_validate_list, eval_functions in tasks:
        for n_to_validate in n_to_validate_list:
            for eval_function in eval_functions:
                gpt_run_config, bayesgpt_run_config = (
                    EvaluationRunConfig(
                        model=model,
                        tokenizer=tokenizer,
                        run_name=run_name,
                        enable_mcdo=enable_mcdo,
                        search_method=search_method,
                        eval_function=eval_function,
                        n_batches_to_validate=n_to_validate,
                        benchmark_name=task_name,
                        dataset=dataset,
                        stepsize=step_size,
                        aq_funcs=aq_funcs,
                    )
                    for model in [model_GPT, model_BayesGPT]
                )
                logger.info(f"Running {task_name}, config: {gpt_run_config}")
                _evaluate_model_on_benchmark(gpt_run_config)
                logger.info(f"Running {task_name}, config: {bayesgpt_run_config}")
                _evaluate_model_on_benchmark(bayesgpt_run_config)

                plot_ret_curve(
                    plot_data_paths=[
                        get_gpt_plot_data_path(gpt_run_config),
                        get_gpt_plot_data_path(bayesgpt_run_config),
                    ],
                    title=task_name,
                    save_filepath=_get_gpt_plot_data_folder(gpt_run_config)
                    + f"/{task_name.lower()}_combined_retcurve_{run_name}_{eval_function.__class__.__name__}_n{n_to_validate}_step{step_size}.svg",
                )


def _evaluate_model_on_benchmark(evaluation_run_config: EvaluationRunConfig) -> None:
    all_outputs, all_uqs = generate_with_uq_for_entire_dataset(evaluation_run_config)
    calc_retention_curve(all_outputs, all_uqs, evaluation_run_config)


if __name__ == "__main__":
    main()
