"""
Script that collects the results from CSV files from InferSent logs and generates the summaries we are interested in.

Run like: `python -m scripts.collect_results_csv`
"""
import argparse
import os.path
from collections import namedtuple

import numpy as np
import pandas as pd

from scripts.collect_results_wandb import ConfigKey, process_results, TABLES
from src.utils.util import nice_print, HORSE, ensure_dir, get_str_formatted_time

INFERSENT_DATASET_LABEL_MAPPING = {
    "mnlimatched_train": "mnli_train",
    "mnlimatched_valid": "mnli_validation_mismatched",
    "mnlimatched_test": "mnli_validation_matched",

    "snli_train": "snli_train",
    "snli_valid": "snli_validation",
    "snli_test": "snli_test",

    "hans_lexical_overlap": "hans_validation",
    "hans_subsequence": "hans_validation",
    "hans_constituent": "hans_validation",
}
CONFIG_KEYS_TO_REPORT = [
    ConfigKey('dataset', "Gamma"),
    ConfigKey('gamma', "Gamma"),
    ConfigKey('seed', "Gamma"),
]
RunCSVInfo = namedtuple("RunCSVInfo", ["dataset", "gamma", "seed", "path"])
RUN_CSV_INFOS = [
    RunCSVInfo("mnli", 0.0, 72, "/home/user72/Desktop/results.csv"),
]

if __name__ == '__main__':
    nice_print(HORSE)
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default="logs/S01.infersent.results",
                        help="Where to save the computed results")
    args = parser.parse_args()

    runs = {}
    for run_csv_info in RUN_CSV_INFOS:
        df = pd.read_csv(run_csv_info.path, sep=";")
        df.rename(columns={"epoch": "step", "dataset_label": "datapoint_dataset"}, inplace=True)
        df["ce_loss"] = -np.log(df.datapoint_true_prob)
        df.loc[df.datapoint_dataset == "hans_subsequence", 'datapoint_idx'] += 10000
        df.loc[df.datapoint_dataset == "hans_constituent", 'datapoint_idx'] += 20000
        df["datapoint_dataset"] = df.datapoint_dataset.apply(lambda ds_label: INFERSENT_DATASET_LABEL_MAPPING[ds_label])

        n = len(df)
        hardness_dataframes = []
        for table_name, table_info in TABLES.items():
            hardness = table_info["hardness"]
            if hardness is None:
                continue
            hardness["datapoint_dataset"] = table_name
            hardness_dataframes += [hardness]
        assert len(hardness_dataframes) != 0
        hardness_df = pd.concat(hardness_dataframes)

        df = pd.merge(
            left=df,
            right=hardness_df,
            how="left",
            left_on=["datapoint_dataset", "datapoint_idx"],
            right_on=["datapoint_dataset", "idx"],
            validate="many_to_one"
        )
        df.drop('idx', axis=1, inplace=True)

        # If merged correctly, adding hardness should not change number of rows
        assert n == len(df)
        # Also, we expect there to always be some hardness annotations
        assert df.hardness.sum() > 0

        runs[run_csv_info.path] = {
            "config": {
                "dataset": run_csv_info.dataset,
                "gamma": run_csv_info.gamma,
                "seed": run_csv_info.seed,
            },
            "dataframe": df,
        }

    ensure_dir(args.results_dir)
    results = process_results(runs, CONFIG_KEYS_TO_REPORT)

    for k, v in results.items():
        with open(os.path.join(args.results_dir, k), "w") as f:
            f.write(v)
    with open(os.path.join(args.results_dir, f"merged_{get_str_formatted_time()}.csv"), "w") as f:
        for k, v in results.items():
            f.write(k)
            f.write("\n")
            f.write(v)
            f.write("\n\n")
    print("Done.")