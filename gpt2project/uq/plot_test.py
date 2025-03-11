from gpt2project.uq.general_plotter import plot_ret_curve

if __name__ == "__main__":
    plot_ret_curve(
        plot_data_paths=[
            "local/gpt-results/commongen/plot_data_gpt2-pretrained_ConceptUsageEval_b1_n-1_step25.pt",
        ],
        title="Commongen",
        save_filepath="local/gpt-results/commongen/commongen_ret_curve_gpt2-pretrained_ConceptUsageEval_b1_n-1_step25.svg",
    )
