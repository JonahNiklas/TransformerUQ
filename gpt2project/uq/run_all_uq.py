from __future__ import annotations

import logging
from typing import List, Tuple

import tiktoken
from tqdm import tqdm

from gpt2project.data_processing.abstract_evaluation_dataset import (
    AbstractEvaluationDataset,
)
from gpt2project.data_processing.commongen_dataset import CommonGen
from gpt2project.data_processing.lambada_dataset import Lambada
from gpt2project.data_processing.load_hellaswag import HellaSwag
from gpt2project.data_processing.squad_dataset import Squad
from gpt2project.data_processing.triviaqa_dataset import TriviaQA
from gpt2project.gpt2_generate import generate_with_uq_for_entire_dataset
from gpt2project.gpt_generate_hellaswag import eval_with_uq_for_entire_hellaswag_dataset
from gpt2project.search_methods_gpt import greedy_search_gpt
from gpt2project.uq.calc_plot_data import calc_retention_curve
from gpt2project.uq.evaluation_run_config import EvaluationRunConfig
from gpt2project.uq.gpt_aq_funcs import (
    BALD,
    AcquisitionFunctionGPT,
    BeamScore,
    BLEUVar,
    mpnet_cosine,
)
from gpt2project.utils.benchmark_eval_funcs import (
    AbstractEval,
    BLEU_eval,
    ConceptUsageEval,
    F1Eval,
    MultipleChoiceEval,
    TargetUsageEval,
)
from gpt2project.utils.checkpoint import get_model_from_wandb_checkpoint
from hyperparameters import hyperparameters
from utils.general_plotter import (
    PlotData,
    _get_gpt_plot_data_folder,
    plot_ret_curve,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    run_name = "full_run_24032025"
    enable_mcdo = True
    search_method = greedy_search_gpt
    step_size = 25

    # wandb.init(project="GPT2Project", name=run_name, job_type="inference")

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

    tasks: List[
        Tuple[
            AbstractEvaluationDataset,
            List[AbstractEval | ConceptUsageEval],
            List[AcquisitionFunctionGPT],
        ]
    ] = [
        (Lambada(), [F1Eval()], [BeamScore(), BALD()]),
        (
            TriviaQA(),
            [TargetUsageEval()],
            [BeamScore(), mpnet_cosine(), BLEUVar()],
        ),
        (
            CommonGen(),
            [BLEU_eval(), ConceptUsageEval()],
            [BeamScore(), mpnet_cosine(), BLEUVar()],
        ),
        (
            Squad(),
            [TargetUsageEval(), BLEU_eval()],
            [BeamScore(), mpnet_cosine(), BLEUVar()],
        ),
        (
            HellaSwag(),
            [MultipleChoiceEval()],
            [BeamScore()],
        ),  # BeamScore here is just a arbitrary placeholder since UQ is hardcoded into the Hellaswag evaluation code
    ]

    run_configs = [
        EvaluationRunConfig(
            model=model,
            tokenizer=tokenizer,
            run_name=run_name,
            enable_mcdo=enable_mcdo,
            search_method=search_method,
            eval_function=eval_function,
            dataset=dataset,
            stepsize=step_size,
            aq_funcs=aq_funcs,
        )
        for dataset, eval_functions, aq_funcs in tasks
        for model in [model_GPT, model_BayesGPT]
        for eval_function in eval_functions
    ]

    for run_config in tqdm(run_configs, desc="Total progress"):
        logger.info(
            f"Running {run_config.model.__class__.__name__} on {run_config.dataset.__class__.__name__} with {run_config.eval_function.__class__.__name__}"
        )
        plot_data = _evaluate_model_on_benchmark(run_config)

        plot_ret_curve(
            plot_data,
            title=run_config.dataset.__class__.__name__,
            save_filepath=_get_gpt_plot_data_folder(run_config) + f"/{run_name}.svg",
        )


def _evaluate_model_on_benchmark(
    evaluation_run_config: EvaluationRunConfig,
) -> PlotData:
    all_outputs, all_uqs = (
        eval_with_uq_for_entire_hellaswag_dataset(evaluation_run_config)
        if isinstance(evaluation_run_config.dataset, HellaSwag)
        else generate_with_uq_for_entire_dataset(evaluation_run_config)
    )
    plot_data = calc_retention_curve(
        all_outputs, all_uqs, evaluation_run_config=evaluation_run_config
    )
    return plot_data


if __name__ == "__main__":
    main()
