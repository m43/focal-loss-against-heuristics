"""
Script that collects the results from WANDB and generates the summaries (like tables and plots) we are interested in.

Run like: `python -m scripts.collect_results_wandb`
"""
import argparse
import os.path
import os.path
import pickle
from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from tqdm import tqdm

from src.constants import INTEGER_TO_DATASET, DATASET_TO_INTEGER
from src.utils.util import nice_print, HORSE, ensure_dir, get_str_formatted_time

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


def load_all_wandb_tables_with_given_artifact_string(
        wandb_run,
        artifact_filter_str,
        csv_file_path_in_artifact,
        cache_path="/tmp/wandb",
):
    artifacts = [a for a in wandb_run.logged_artifacts() if artifact_filter_str in a.name]
    df_all = pd.DataFrame({})
    for artifact in artifacts:
        artifact_dir = artifact.download(os.path.join(cache_path, f"{artifact.name}"))
        csv_path = os.path.join(artifact_dir, csv_file_path_in_artifact)

        df = pd.read_csv(csv_path)
        df["wandb_artifact_name"] = artifact.name
        source_url = f"https://wandb.ai/{artifact.entity}/{artifact.project}/artifacts/{artifact.type}/{artifact.name.replace(':', '/')}/files"
        df["source"] = source_url

        df = df.reset_index(drop=True)
        df_all = pd.concat([df_all, df])

    df_all = df_all.reset_index(drop=True)
    return df_all


def load_run_dataframes_from_wandb(wandb_run, tables):
    print(f"Loading all dataframes for wandb_run: {wandb_run}")

    dataframes = {}
    for table_name, table_info in tqdm(tables.items()):
        dataframes[table_name] = load_all_wandb_tables_with_given_artifact_string(
            wandb_run=wandb_run,
            artifact_filter_str=table_info["wandb_artifact_name"],
            csv_file_path_in_artifact='df.csv',
        )

        # Add hardness column
        if table_info["hardness"] is not None and not dataframes[table_name].empty:
            n = len(dataframes[table_name])
            dataframes[table_name] = pd.merge(
                left=dataframes[table_name],
                right=table_info["hardness"],
                left_on="datapoint_idx",
                right_on="idx",
                validate="many_to_one"
            )
            dataframes[table_name].drop('idx', axis=1, inplace=True)

            # If merged correctly, adding hardness should not change number of rows.
            assert n == len(dataframes[table_name])
        else:
            dataframes[table_name]["hardness"] = None

    return dataframes


def load_runs_from_wandb(run_paths, tables):
    runs = {}

    wapi = wandb.Api()
    for run_path in tqdm(run_paths):
        wandb_run = wapi.run(run_path)

        run = {
            "config": wandb_run.config,
            "dataframes": load_run_dataframes_from_wandb(wandb_run, tables),
        }

        runs[run_path] = run

    return runs


ConfigKey = namedtuple("ConfigKey", ["config_key", "pretty_name"])
CONFIG_KEYS_TO_REPORT: List[ConfigKey] = [
    ConfigKey('focal_loss_gamma', "Gamma"),
    ConfigKey('run_config/dataset', "Dataset"),
    ConfigKey('batch_size', "Batch Size"),
    ConfigKey('adam_epsilon', "AdamW Epsilon"),
    ConfigKey('warmup_ratio', "Warmup Ratio"),
    ConfigKey('weight_decay', "Weight Decay"),
    ConfigKey('learning_rate', "Learning Rate"),
    ConfigKey('run_config/seed', "Seed"),
    ConfigKey('run_config/precision', "Precision"),
    ConfigKey('run_config/num_hans_train_examples', "HANS Examples in Train")
]

ReportMetric = namedtuple("ReportMetric", ["pretty_name", "dataset", "aggregate_function", "key", "hardness"])
EarlyStoppingMetric = namedtuple("CheckpointMetric",
                                 ["pretty_name", "dataset", "aggregate_function", "key", "idxmin_or_idxmax"])
EARLY_STOPPING_METRICS = [
    EarlyStoppingMetric("Last Step", "hans_validation", "max", "step", "idxmax"),
    EarlyStoppingMetric("MNLI.V.M.acc", "mnli_validation_matched", "mean", "datapoint_true_pred", "idxmax"),
    EarlyStoppingMetric("MNLI.V.M.loss", "mnli_validation_matched", "mean", "datapoint_loss", "idxmin"),
    EarlyStoppingMetric("MNLI.V.MM.acc", "mnli_validation_mismatched", "mean", "datapoint_true_pred", "idxmax"),
    EarlyStoppingMetric("MNLI.V.MM.loss", "mnli_validation_mismatched", "mean", "datapoint_loss", "idxmin"),
    EarlyStoppingMetric("SNLI.V.acc", "snli_validation", "mean", "datapoint_true_pred", "idxmax"),
    EarlyStoppingMetric("SNLI.V.loss", "snli_validation", "mean", "datapoint_loss", "idxmin"),
]
REPORT_METRICS = [
    ReportMetric("HANS.valid.acc", "hans_validation", "mean", "datapoint_true_pred", None),
    ReportMetric("MNLI.train.M.acc", "mnli_train", "mean", "datapoint_true_pred", None),
    ReportMetric("MNLI.valid.M.acc", "mnli_validation_matched", "mean", "datapoint_true_pred", None),
    ReportMetric("MNLI.valid.M.EASY.acc", "mnli_validation_matched", "mean", "datapoint_true_pred", "easy"),
    ReportMetric("MNLI.valid.M.HARD.acc", "mnli_validation_matched", "mean", "datapoint_true_pred", "hard"),
    ReportMetric("MNLI.valid.MM.acc", "mnli_validation_mismatched", "mean", "datapoint_true_pred", None),
    ReportMetric("MNLI.valid.MM.EASY.acc", "mnli_validation_mismatched", "mean", "datapoint_true_pred", "easy"),
    ReportMetric("MNLI.valid.MM.HARD.acc", "mnli_validation_mismatched", "mean", "datapoint_true_pred", "hard"),
    ReportMetric("SNLI.train.acc", "snli_train", "mean", "datapoint_true_pred", None),
    ReportMetric("SNLI.valid.acc", "snli_validation", "mean", "datapoint_true_pred", None),
    ReportMetric("SNLI.test.acc", "snli_test", "mean", "datapoint_true_pred", None),
    ReportMetric("SNLI.test.EASY.acc", "snli_test", "mean", "datapoint_true_pred", "easy"),
    ReportMetric("SNLI.test.HARD.acc", "snli_test", "mean", "datapoint_true_pred", "hard"),
]


def process_results(
        runs,
        config_keys_to_report: List[ConfigKey] = CONFIG_KEYS_TO_REPORT,
        early_stopping_metrics: List[EarlyStoppingMetric] = EARLY_STOPPING_METRICS,
        report_metrics: List[ReportMetric] = REPORT_METRICS,
):
    assert len(runs) > 0
    assert len(config_keys_to_report) > 0
    assert len(early_stopping_metrics) > 0
    for esm in early_stopping_metrics:
        assert esm.dataset in DATASET_TO_INTEGER
    assert len(set([esm.pretty_name for esm in early_stopping_metrics])) == len(early_stopping_metrics)

    results = {}

    config_csv_str = ""
    config_csv_str += "run_path;"
    config_csv_str += ";".join([ck.pretty_name for ck in config_keys_to_report])
    config_csv_str += "\n"
    for run_id, run in runs.items():
        config_csv_str += f"{run_id};"
        config_csv_str += ";".join([str(run["config"][ck.config_key]) for ck in config_keys_to_report])
        config_csv_str += "\n"

    results["config.csv"] = config_csv_str
    print(results["config.csv"], "\n")

    for esm in tqdm(early_stopping_metrics):
        report_metrics_csv_str = ""
        report_metrics_csv_str += "run_path;early_stopping_step;"
        report_metrics_csv_str += ";".join([str(rm.pretty_name) for rm in report_metrics])
        report_metrics_csv_str += "\n"
        for run_id, run in runs.items():
            df = run["dataframe"]
            esm_df = df[df.datapoint_dataset == esm.dataset].copy()
            if esm_df.empty:
                continue

            esm_df["__step"] = esm_df["step"]  # To resolve edge case when esm.key=="step"
            esm_df = esm_df[["__step", esm.key]]
            esm_step = esm_df.groupby("__step").agg(esm.aggregate_function)[esm.key].apply(esm.idxmin_or_idxmax)

            report_metrics_csv_str += f"{run_id};{esm_step}"
            for rm in report_metrics:
                if rm.dataset == "hans_validation":
                    # TODO heuristics
                    pass
                rm_df = df[(df.datapoint_dataset == rm.dataset) & (df.step == esm_step)]
                if rm_df.empty:
                    value = -1
                else:
                    if rm.hardness == "easy":
                        rm_df = rm_df[rm_df.hardness == 0]
                        assert not rm_df.empty
                    if rm.hardness == "hard":
                        rm_df = rm_df[rm_df.hardness == 1]
                        assert not rm_df.empty
                    value = rm_df[rm.key].agg(rm.aggregate_function)
                report_metrics_csv_str += f";{value}"
            report_metrics_csv_str += "\n"

            # TODO possibly add plots here if `esm_step` is needed

        results[f"report_metric.earlystopping_on_{esm.pretty_name}.csv"] = report_metrics_csv_str
        print(results[f"report_metric.earlystopping_on_{esm.pretty_name}.csv"], "\n")

    return results


MNLI_VALIDATION_MATCHED_HARDNESS = pd.read_csv("data/mnli_validation_matched_hardness.csv")
MNLI_VALIDATION_MISMATCHED_HARDNESS = pd.read_csv("data/mnli_validation_mismatched_hardness.csv")
SNLI_TEST_HARDNESS = pd.read_csv("data/snli_test_hardness.csv")

TABLES = {
    "mnli_train": {
        "hardness": None,
        "wandb_artifact_name": "Train-mnli_train_epoch_end_df"
    },
    "mnli_validation_matched": {
        "hardness": MNLI_VALIDATION_MATCHED_HARDNESS,
        "wandb_artifact_name": "Valid-mnli_validation_matched_epoch_end_df"
    },
    "mnli_validation_mismatched": {
        "hardness": MNLI_VALIDATION_MISMATCHED_HARDNESS,
        "wandb_artifact_name": "Valid-mnli_validation_mismatched_epoch_end_df"
    },
    "hans_validation": {
        "hardness": None,
        "wandb_artifact_name": "Valid-hans_validation_epoch_end_df"
    },
    "snli_train": {
        "hardness": None,
        "wandb_artifact_name": "Train-snli_train_epoch_end_df"
    },
    "snli_validation": {
        "hardness": None,
        "wandb_artifact_name": "Valid-snli_validation_epoch_end_df"
    },
    "snli_test": {
        "hardness": SNLI_TEST_HARDNESS,
        "wandb_artifact_name": "Valid-snli_test_epoch_end_df"
    },
}

RUN_PATHS = [
    "epfl-optml/nli/S1.01.A_e-03_model-bert_dataset-mnli_gamma-0.0_seed-72_09.25_01.53.02",
    # "epfl-optml/nli/S1.01.B_e-04__model-bert_dataset-mnli_gamma-0.0_seed-24_09.25_01.53.02",
    # "epfl-optml/nli/S1.01.C_e-10__model-bert_dataset-mnli_gamma-0.0_seed-24_09.25_01.53.02",
    # "epfl-optml/nli/S1.01.C_e-10__model-bert_dataset-mnli_gamma-0.0_seed-72_09.25_01.53.02",
    # "epfl-optml/nli/S1.01.D_e-3_p-32__model-bert_dataset-mnli_gamma-0.0_seed-24_09.25_01.53.02",
    # "epfl-optml/nli/S1.01.E_eps-8_model-bert_dataset-mnli_gamma-0.0_seed-24_09.25_01.53.02",
    # "epfl-optml/nli/S1.01.F_mahabadi_eps8-bs8-wmp0-wd0-p32__model-bert_dataset-mnli_gamma-0.0_seed-24_09.25_01.53.02",
    # "epfl-optml/nli/S1.01.G_clark_lr-5e-5__model-bert_dataset-mnli_gamma-0.0_seed-24_09.25_01.53.02",
    # "epfl-optml/nli/S1.02_model-bert_dataset-mnli_gamma-0.5_seed-72_09.25_01.53.02",
    # "epfl-optml/nli/S1.03_model-bert_dataset-mnli_gamma-1.0_seed-72_09.25_01.53.02",
    # "epfl-optml/nli/S1.04_model-bert_dataset-mnli_gamma-2.0_seed-72_09.25_01.53.04",
    # "epfl-optml/nli/S1.05_model-bert_dataset-mnli_gamma-5.0_seed-72_09.25_01.53.12",
    # "epfl-optml/nli/S1.06_model-bert_dataset-mnli_gamma-10.0_seed-72_09.25_01.53.12",
    # "epfl-optml/nli/S1.07_model-bert_dataset-snli_gamma-0.0_seed-72_09.25_01.53.12",
    # "epfl-optml/nli/S1.08_model-bert_dataset-snli_gamma-0.5_seed-72_09.25_01.53.32",
    # "epfl-optml/nli/S1.09_model-bert_dataset-snli_gamma-1.0_seed-72_09.25_02.57.08",
    # "epfl-optml/nli/S1.10_model-bert_dataset-snli_gamma-2.0_seed-72_09.25_02.57.08",
    # "epfl-optml/nli/S1.11_model-bert_dataset-snli_gamma-5.0_seed-72_09.25_03.02.37",
    # "epfl-optml/nli/S1.12_model-bert_dataset-snli_gamma-10.0_seed-72_09.25_03.16.14",
]

if __name__ == '__main__':
    nice_print(HORSE)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cached_runs_pickle_path', type=str, default="logs/cached_runs.pkl",
                        help="If not None and if pickle file exists,"
                             "load the runs from the pickle file instead of querying wandb again.")
    parser.add_argument('--results_dir', type=str, default="logs/S01.results",
                        help="Where to save the computed results")
    args = parser.parse_args()

    if args.cached_runs_pickle_path is not None and os.path.exists(args.cached_runs_pickle_path):
        with open(args.cached_runs_pickle_path, "rb") as f:
            runs = pickle.load(f)
        print(f"Loaded runs from cache pickle with path: `{args.cached_runs_pickle_path}`")
    else:
        print(f"Collecting run data from WANDB public API.")
        runs = load_runs_from_wandb(RUN_PATHS, TABLES)
        print(f"Runs collected.")
        if args.cached_runs_pickle_path is not None:
            with open(args.cached_runs_pickle_path, "wb") as f:
                pickle.dump(runs, f)
            print(f"Saved runs to cache pickle with path: `{args.cached_runs_pickle_path}`")

    for run_path, run in runs.items():
        df = pd.concat(run["dataframes"])
        df["ce_loss"] = df["datapoint_true_prob"].apply(lambda x: -np.log(x))
        df["datapoint_dataset"] = df["datapoint_dataset"].apply(lambda x: INTEGER_TO_DATASET[x])

    ensure_dir(args.results_dir)
    results = process_results(runs)

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

'''
def preprocess_table(df, hardness=None, sort_by_loss=False):
    max_step = df.step.max()
    df = df[df.step == max_step]
    if hardness is not None:
        assert len(df) == len(hardness)
    if sort_by_loss:
        df = df.sort_values("datapoint_loss")
    return df


# Plot -1: What are the accuracies at all?
df_heatmap = df.rename(columns={"datapoint_true_pred": "accuracy"}).pivot_table(values="accuracy", index="gamma")
sns.heatmap(df_heatmap, linewidths=.5, annot=True, square=True, fmt=".4f", vmin=0.6, vmax=1).set(title="Accuracy (wrt gamma)")
plt.show()

df_heatmap = df.pivot_table(values="datapoint_true_pred", index="gamma", columns="hardness")
sns.heatmap(df_heatmap, linewidths=.5, annot=True, square=True, fmt=".4f", vmin=0.6, vmax=1).set(title='Accuracy (wrt gamma and hardness)')
plt.show()

# Plot A
fig = plt.figure(dpi=100)
g = sns.displot(df, x="datapoint_loss", hue="hardness", bins=100, log_scale=(True, False), palette="deep", kind="hist", row="gamma", height=3, aspect=3)
line_position = [-0.5**gamma*np.log(0.5) for gamma in [0.0, 10.0]]
for ax, pos in zip(g.axes.flat, line_position):
    ax.axvline(x=pos, color='r', linestyle=':')
# plt.tight_layout()
plt.show()

fig = plt.figure(dpi=100)
g = sns.displot(df, x="ce_loss", hue="hardness", bins=100, log_scale=(True, False), palette="deep", kind="hist", row="gamma", height=3, aspect=3)
line_position = [-0.5**gamma*np.log(0.5) for gamma in [0.0, 0.0]]
for ax, pos in zip(g.axes.flat, line_position):
    ax.axvline(x=pos, color='r', linestyle=':')
# plt.tight_layout()
plt.show()

"""
Observations:
1. There are two modes in the distribution, one with 2-3 orders of magnitude larger loss.
2. There are many more hard examples around the larger-loss mode.
3. For gamma=10, there is a noticeable shift to the right between the hard and non-hard distributions.
4. Larger gamma reduces the effect of the two modes, moving the hard examples to the left, i.e. reducing their loss. This promises better peformance.
5. It does not seem to distinguish hardness of examples as the harder examples have higher loss
6. The red line is in the middle for gamma=10.0 means that a lot of the probabilities cluster around 50% (also visible on the probs histogram, plot B). But what does this practically mean for the performance? How is it related (if at all) with the peformance going down?
7. ??? But what does it mean that the gamma=10.0 is clustered around 50%? Why does it even happen?
"""

# Plot B
sns.displot(df.rename(columns={"datapoint_true_prob": "probability of correct"}), x="probability of correct", hue="hardness", col="gamma", bins=100, log_scale=(False, True), palette="deep", height=3, aspect=3)
# plt.suptitle("Histogram of probability of correct prediction (wrt. gamma and hardness)")
plt.show()

# sns.displot(df, x="datapoint_true_prob", hue="gamma", col="gamma", bins=40, log_scale=(False, True), palette="deep", row="hardness", height=3, aspect=3)
# plt.show()
"""
Observations:
1. TODO
"""



# Plot C
fig = plt.figure(dpi=100)
sns.displot(df, x="datapoint_loss", hue="hardness", log_scale=(True, False), linewidth=3, palette="deep", kind="ecdf", row="gamma", height=3, aspect=3)
plt.show()

fig = plt.figure(dpi=100)
sns.displot(df, x="datapoint_loss", hue="hardness", log_scale=(False, False), linewidth=3, palette="deep", kind="ecdf", row="gamma", height=3, aspect=3)
plt.show()

fig = plt.figure(dpi=100)
sns.displot(df, x="ce_loss", hue="hardness", log_scale=(False, False), linewidth=3, palette="deep", kind="ecdf", row="gamma", height=3, aspect=3)
plt.show()

"""
Observations:
1. Harder has larger loss.
2. There is no notion of a 90-degree hinge.
3. Larger gamma (of value 10) makes the high loss tails disappear for hard examples.
"""
'''
