from utils.general_plotter import plot_ret_curve


if __name__ == "__main__":

    val_spec = [
        {
            "search_method": "greedy",
            "dropout": True,
        },
        {
            "search_method": "beam",
            "dropout": True,
        },
        {
            "search_method": "sample",
            "dropout": True,
        },
        {
            "search_method": "sample",
            "dropout": False,
        },
    ]
    trans_run_id = "xn8evvcd"
    trans_run_name = "Transformer"

    bayes_run_id = "7sy5cau3"  
    bayes_run_name = "Bayesformer"
    for spec in val_spec:
        search_method: str = str(spec["search_method"])
        dropout: bool = bool(spec["dropout"])

        plot_ret_curve(
            plot_data_paths=[
                f"local/results/{trans_run_id}/{search_method}/dropout{dropout}/{trans_run_name}_{search_method}_drop{dropout}_retcurve_ood_data.pt",
                f"local/results/{bayes_run_id}/{search_method}/dropout{dropout}/{bayes_run_name}_{search_method}_drop{dropout}_retcurve_ood_data.pt",
            ],
            title="Transformer vs Bayesformer",
            save_filepath= f"local/results/{trans_run_id}/{search_method}/dropout{dropout}/ret_curve_comparison_{search_method}_drop{dropout}.svg",
        )