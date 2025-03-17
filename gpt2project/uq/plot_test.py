from utils.general_plotter import plot_ret_curve

if __name__ == "__main__":
    plot_ret_curve(
        plot_data_paths=[
            "local/gpt-results/commongen/plot_data_own_gpt-76291steps_ConceptUsageEval_b1_n-1_step25.pt",
            "local/gpt-results/commongen/plot_data_BayesGPT-76291steps_ConceptUsageEval_b1_n-1_step25.pt",
        ],
        title="bayesGPT vs GPT",
        save_filepath="local/gpt-results/commongen/commongen_ret_curve_comparison_ConceptUsageEval_b1_n-1_step25.svg",
    )
    plot_ret_curve(
        plot_data_paths=[
            "local/gpt-results/commongen/plot_data_own_gpt-76291steps_BLEU_eval_b1_n-1_step25.pt",
            "local/gpt-results/commongen/plot_data_BayesGPT-76291steps_BLEU_eval_b1_n-1_step25.pt",
        ],
        title="bayesGPT vs GPT",
        save_filepath="local/gpt-results/commongen/commongen_ret_curve_comparison_BLEU_eval_b1_n-1_step25.svg",
    )

    # plot_ret_curve(
    #     plot_data_paths=[
    #         "local/gpt-results/triviaqa/plot_data_gpt2-pretrained_ConceptUsageEval_b1_n-1_step25.pt",
    #     ],
    #     title="TriviaQA",
    #     save_filepath="local/gpt-results/triviaqa/triviaqa_ret_curve_gpt2-pretrained_ConceptUsageEval_b1_n-1_step25.svg",
    # )

    plot_ret_curve(
        plot_data_paths=[
            "local/gpt-results/squad/plot_data_bayesGPT-BBm_TargetUsageEval_b1_n1000_step25.pt",
            "local/gpt-results/squad/plot_data_GPT-BBm_TargetUsageEval_b1_n1000_step25.pt",
        ],
        title="SQuAD",
        save_filepath="local/gpt-results/squad/squad_combined_retcurve_TargetUsageEval_b1_n1000_step25.svg",
    )

    plot_ret_curve(
        plot_data_paths=[
            "local/gpt-results/lambada/plot_data_BayesGPT_F1Eval_b1_n-1_step25.pt",
            "local/gpt-results/lambada/plot_data_GPT_F1Eval_b1_n-1_step25.pt",
        ],
        title="LAMBADA",
        save_filepath="local/gpt-results/lambada/lambada_combined_retcurve_F1Eval_b1_n-1_step25.svg",
    )
