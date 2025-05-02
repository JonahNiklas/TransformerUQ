from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import tiktoken
import torch
from tqdm import tqdm

from gpt2project.data_processing.abstract_evaluation_dataset import (
    AbstractEvaluationDataset,
)
from gpt2project.data_processing.commongen_dataset import CommonGen
from gpt2project.data_processing.lambada_dataset import Lambada
from gpt2project.data_processing.load_hellaswag import HellaSwag
from gpt2project.data_processing.squad_dataset import Squad
from gpt2project.data_processing.translated_dataset import TranslatedDataset
from gpt2project.data_processing.triviaqa_dataset import TriviaQA
from gpt2project.gpt2_generate import generate_with_uq_for_entire_dataset
from gpt2project.gpt_generate_hellaswag import eval_with_uq_for_entire_hellaswag_dataset
from gpt2project.search_methods_gpt import greedy_search_gpt, topk_sampling_gpt
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
from uq.plot_uq import plot_combined_roc_curve, plot_uq_histogram
from uq.validate_uq import ValidationResult
from utils.general_plotter import (
    PlotData,
    _get_gpt_plot_data_folder,
    get_gpt_plot_data_path,
    plot_ret_curve,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    run_name = "gpt_ood_run_30042025"
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
        (
            Lambada(),
            [F1Eval()],
            [BeamScore(), BALD()],
        ),
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
        plot_data, all_outputs, all_uqs = _evaluate_model_on_benchmark(run_config)
        plot_ret_curve(
            plot_data,
            title=repr(run_config.dataset),
            save_filepath=_get_gpt_plot_data_folder(run_config) + f"/{run_name}.svg",
        )

        ood_run_config = EvaluationRunConfig(
            model=run_config.model,
            tokenizer=run_config.tokenizer,
            run_name=run_config.run_name,
            enable_mcdo=run_config.enable_mcdo,
            search_method=run_config.search_method,
            eval_function=run_config.eval_function,
            dataset=TranslatedDataset(run_config.dataset),
            stepsize=run_config.stepsize,
            aq_funcs=run_config.aq_funcs,
        )
        ood_plot_data, ood_all_outputs, ood_all_uqs = _evaluate_model_on_benchmark(
            ood_run_config
        )
        plot_ret_curve(
            ood_plot_data,
            title=repr(ood_run_config.dataset),
            save_filepath=_get_gpt_plot_data_folder(ood_run_config)
            + f"/{run_name}.svg",
        )

        _plot_uq_histogram(all_uqs, ood_all_uqs, run_config.aq_funcs, run_config)


def _evaluate_model_on_benchmark(
    evaluation_run_config: EvaluationRunConfig,
) -> Tuple[PlotData, List[List[str]], torch.Tensor]:
    all_outputs, all_uqs = (
        eval_with_uq_for_entire_hellaswag_dataset(evaluation_run_config)
        if isinstance(evaluation_run_config.dataset, HellaSwag)
        else generate_with_uq_for_entire_dataset(evaluation_run_config)
    )
    plot_data = calc_retention_curve(
        all_outputs, all_uqs, evaluation_run_config=evaluation_run_config
    )
    return plot_data, all_outputs, all_uqs


def _plot_uq_histogram(
    all_uqs: torch.Tensor,
    ood_all_uqs: torch.Tensor,
    aq_funcs: List[AcquisitionFunctionGPT],
    run_config: EvaluationRunConfig,
) -> None:
    assert all_uqs.shape == (len(run_config.dataset), len(aq_funcs))

    validation_result_id_list: List[ValidationResult] = []
    validation_result_ood_list: List[ValidationResult] = []
    for i, aq_func in enumerate(aq_funcs):
        validation_result_id = ValidationResult(
            hypothesis=[], reference=[], uncertainty=all_uqs[:, i]
        )
        validation_result_ood = ValidationResult(
            hypothesis=[], reference=[], uncertainty=ood_all_uqs[:, i]
        )
        plot_uq_histogram(
            validation_result_id,
            validation_result_ood,
            aq_func.__class__.__name__,
            get_gpt_plot_data_path(run_config, file_extension=f"_{aq_func.__class__.__name__}_histogram.svg"),
            run_config.run_name,
        )
        validation_result_id_list.append(validation_result_id)
        validation_result_ood_list.append(validation_result_ood)

    plot_combined_roc_curve(
        validation_result_id_list,
        validation_result_ood_list,
        [aq_func.__class__.__name__ for aq_func in aq_funcs],
        get_gpt_plot_data_path(run_config, file_extension="roc.svg"),
    )


if __name__ == "__main__":
    main()
