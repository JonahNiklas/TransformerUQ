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
    TargetUsageEval,
)
from hyperparameters import hyperparameters
from gpt2project.utils.checkpoint import get_model_from_wandb_checkpoint
from utils.general_plotter import plot_ret_curve


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

    run_name = "run1"

    enable_mcdo = True
    search_method = greedy_search_gpt

    # Lambada
    get_lambada_run(
        model_BayesGPT,
        tokenizer,
        run_name,
        enable_mcdo,
        search_method,
    )

    get_lambada_run(
        model_GPT,
        tokenizer,
        run_name,
        enable_mcdo,
        search_method,
    )

    plot_ret_curve(
        plot_data_paths=[
            f"local/gpt-results/lambada/GPT/mcdo{enable_mcdo}/{search_method.__name__}/plot_data_run1_F1Eval_n-1_step25.pt",
            f"local/gpt-results/lambada/BayesGPT/mcdo{enable_mcdo}/{search_method.__name__}/plot_data_run1_F1Eval_n-1_step25.pt",
        ],
        title="LAMBADA",
        save_filepath=f"local/gpt-results/lambada/GPT/mcdo{enable_mcdo}/{search_method.__name__}/lambada_combined_retcurve_F1Eval_n-1_step25.svg",
    )

    # TriviaQA
    get_triviaqa_run(
        model_BayesGPT,
        tokenizer,
        run_name,
        enable_mcdo,
        search_method,
    )

    get_triviaqa_run(
        model_GPT,
        tokenizer,
        run_name,
        enable_mcdo,
        search_method,
    )

    plot_ret_curve(
        plot_data_paths=[
            f"local/gpt-results/triviaqa/GPT/mcdo{enable_mcdo}/{search_method.__name__}/plot_data_run1_TargetUsageEval_n-1_step25.pt",
            f"local/gpt-results/triviaqa/BayesGPT/mcdo{enable_mcdo}/{search_method.__name__}/plot_data_run1_TargetUsageEval_n-1_step25.pt",
        ],
        title="TriviaQA",
        save_filepath=f"local/gpt-results/triviaqa/GPT/mcdo{enable_mcdo}/{search_method.__name__}/triviaqa_combined_retcurve_TargetUsageEval_n-1_step25.svg",
    )

    # CommonGen with BLEU
    get_commongen_run(
        model_BayesGPT,
        tokenizer,
        run_name,
        enable_mcdo,
        search_method,
        eval_function=BLEU_eval(),
    )

    get_commongen_run(
        model_GPT,
        tokenizer,
        run_name,
        enable_mcdo,
        search_method,
        eval_function=BLEU_eval(),
    )

    plot_ret_curve(
        plot_data_paths=[
            f"local/gpt-results/commongen/GPT/mcdo{enable_mcdo}/{search_method.__name__}/plot_data_run1_BLEU_eval_n-1_step25.pt",
            f"local/gpt-results/commongen/BayesGPT/mcdo{enable_mcdo}/{search_method.__name__}/plot_data_run1_BLEU_eval_n-1_step25.pt",
        ],
        title="CommonGen",
        save_filepath=f"local/gpt-results/commongen/GPT/mcdo{enable_mcdo}/{search_method.__name__}/commongen_combined_retcurve_BLEU_eval_n-1_step25.svg",
    )

    # CommonGen with ConceptUsageEval
    get_commongen_run(
        model_BayesGPT,
        tokenizer,
        run_name,
        enable_mcdo,
        search_method,
        eval_function=ConceptUsageEval(),
    )

    get_commongen_run(
        model_GPT,
        tokenizer,
        run_name,
        enable_mcdo,
        search_method,
        eval_function=ConceptUsageEval(),
    )

    plot_ret_curve(
        plot_data_paths=[
            f"local/gpt-results/commongen/GPT/mcdo{enable_mcdo}/{search_method.__name__}/plot_data_run1_ConceptUsageEval_n-1_step25.pt",
            f"local/gpt-results/commongen/BayesGPT/mcdo{enable_mcdo}/{search_method.__name__}/plot_data_run1_ConceptUsageEval_n-1_step25.pt",
        ],
        title="CommonGen",
        save_filepath=f"local/gpt-results/commongen/GPT/mcdo{enable_mcdo}/{search_method.__name__}/commongen_combined_retcurve_ConceptUsageEval_n-1_step25.svg",
    )

    # SQuAD with TargetUsageEval
    get_squad_run(
        model_BayesGPT,
        tokenizer,
        run_name,
        enable_mcdo,
        search_method,
        eval_function=TargetUsageEval(),
        n_batch_to_validate=1000,
    )

    get_squad_run(
        model_GPT,
        tokenizer,
        run_name,
        enable_mcdo,
        search_method,
        eval_function=TargetUsageEval(),
        n_batch_to_validate=1000,
    )

    plot_ret_curve(
        plot_data_paths=[
            f"local/gpt-results/squad/GPT/mcdo{enable_mcdo}/{search_method.__name__}/plot_data_run1_TargetUsageEval_n1000_step25.pt",
            f"local/gpt-results/squad/BayesGPT/mcdo{enable_mcdo}/{search_method.__name__}/plot_data_run1_TargetUsageEval_n1000_step25.pt",
        ],
        title="SQuAD",
        save_filepath=f"local/gpt-results/squad/GPT/mcdo{enable_mcdo}/{search_method.__name__}/squad_combined_retcurve_TargetUsageEval_n1000_step25.svg",
    )

    # SQuAD with BLEU
    get_squad_run(
        model_BayesGPT,
        tokenizer,
        run_name,
        enable_mcdo,
        search_method,
        eval_function=BLEU_eval(),
        n_batch_to_validate=1000,
    )

    get_squad_run(
        model_GPT,
        tokenizer,
        run_name,
        enable_mcdo,
        search_method,
        eval_function=BLEU_eval(),
        n_batch_to_validate=1000,
    )

    plot_ret_curve(
        plot_data_paths=[
            f"local/gpt-results/squad/GPT/mcdo{enable_mcdo}/{search_method.__name__}/plot_data_run1_BLEU_eval_n1000_step25.pt",
            f"local/gpt-results/squad/BayesGPT/mcdo{enable_mcdo}/{search_method.__name__}/plot_data_run1_BLEU_eval_n1000_step25.pt",
        ],
        title="SQuAD",
        save_filepath=f"local/gpt-results/squad/GPT/mcdo{enable_mcdo}/{search_method.__name__}/squad_combined_retcurve_BLEU_eval_n1000_step25.svg",
    )
