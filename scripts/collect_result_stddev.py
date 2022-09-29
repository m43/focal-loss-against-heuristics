"""
Script that collects the results from CSV files from InferSent logs and generates the summaries we are interested in.

Run like: `python -m scripts.collect_results_infersent_csv`
"""
import argparse
import os.path
from functools import reduce

import pandas as pd

from scripts.collect_results_wandb import REPORT_METRICS
from src.utils.util import nice_print, HORSE, get_str_formatted_time

REPORT_METRIC_CSV_PREFIX = "report_metric."
STDDEV_CSV_PREFIX = "stddev."

if __name__ == '__main__':
    nice_print(HORSE)
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default="logs/S01.results",
                        help="Where to save the computed results")
    args = parser.parse_args()

    assert os.path.exists(args.results_dir)

    config = pd.read_csv(os.path.join(args.results_dir, "config.csv"), sep=";")
    config = config.set_index("run_path")
    assert config.index.is_unique

    report_metric_csvs = [fname for fname in os.listdir(args.results_dir) if fname.startswith(REPORT_METRIC_CSV_PREFIX)]
    for fname in report_metric_csvs:
        df = pd.read_csv(os.path.join(args.results_dir, fname), sep=";")

        df["dataset"] = df.run_path.apply(lambda run_path: config.loc[run_path]["Dataset"])
        df["gamma"] = df.run_path.apply(lambda run_path: config.loc[run_path]["Gamma"])
        df["seed"] = df.run_path.apply(lambda run_path: config.loc[run_path]["Seed"])
        df["n_hans"] = df.run_path.apply(lambda run_path: config.loc[run_path]["HANS Examples in Train"])

        stddev_dataframes = []
        for rm in REPORT_METRICS:
            stddev_dataframes.append(
                df[["dataset", "gamma", "n_hans", rm.pretty_name]].groupby(["dataset", "n_hans","gamma"], as_index=False).agg(
                    {rm.pretty_name: ['mean', 'std', 'count']})
            )

        stddev_df = reduce(lambda x, y: pd.merge(x, y, on=['dataset', "n_hans", "gamma"]), stddev_dataframes)

        stddev_df.to_csv(os.path.join(args.results_dir, f"{STDDEV_CSV_PREFIX}{fname}"))

    with open(os.path.join(args.results_dir, f"merged_stddev_{get_str_formatted_time()}.csv"), "w") as f:
        for fname in report_metric_csvs:
            stddev_fname = f"{STDDEV_CSV_PREFIX}{fname}"
            f.write(stddev_fname)
            f.write("\n")
            with open(os.path.join(args.results_dir, stddev_fname)) as f_in:
                f.write(f_in.read())
            f.write("\n\n\n")
    print("Done.")
