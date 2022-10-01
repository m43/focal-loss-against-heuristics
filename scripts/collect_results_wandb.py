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
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.constants import INTEGER_TO_DATASET, DATASET_TO_INTEGER, INTEGER_TO_HEURISTIC
from src.utils.util import nice_print, HORSE, ensure_dir, get_str_formatted_time

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set_style("ticks")

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)


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

ReportMetric = namedtuple("ReportMetric",
                          ["pretty_name", "dataset", "aggregate_function", "key", "hardness", "hans_label",
                           "hans_heuristic"])
EarlyStoppingMetric = namedtuple("CheckpointMetric",
                                 ["pretty_name", "dataset", "aggregate_function", "key", "idxmin_or_idxmax"])
EARLY_STOPPING_METRICS = [
    EarlyStoppingMetric("Last Step", "hans_validation", "max", "step", "idxmax"),
    # EarlyStoppingMetric("MNLI.V.M.acc", "mnli_validation_matched", "mean", "datapoint_true_pred", "idxmax"),
    # EarlyStoppingMetric("MNLI.V.M.loss", "mnli_validation_matched", "mean", "datapoint_loss", "idxmin"),
    # EarlyStoppingMetric("MNLI.V.MM.acc", "mnli_validation_mismatched", "mean", "datapoint_true_pred", "idxmax"),
    EarlyStoppingMetric("MNLI.V.MM.loss", "mnli_validation_mismatched", "mean", "datapoint_loss", "idxmin"),
    # EarlyStoppingMetric("SNLI.V.acc", "snli_validation", "mean", "datapoint_true_pred", "idxmax"),
    # EarlyStoppingMetric("SNLI.V.loss", "snli_validation", "mean", "datapoint_loss", "idxmin"),
]
REPORT_METRICS = [
    ReportMetric("MNLI.train.M.acc", "mnli_train", "mean", "datapoint_true_pred", None, None, None),
    ReportMetric("HANS.valid.acc", "hans_validation", "mean", "datapoint_true_pred", None, None, None),
    # ReportMetric("HANS.valid.E.LO.acc", "hans_validation", "mean", "datapoint_true_pred", None, 0, "lexical_overlap"),
    # ReportMetric("HANS.valid.E.S.acc", "hans_validation", "mean", "datapoint_true_pred", None, 0, "subsequence"),
    # ReportMetric("HANS.valid.E.C.acc", "hans_validation", "mean", "datapoint_true_pred", None, 0, "constituent"),
    # ReportMetric("HANS.valid.NE.LO.acc", "hans_validation", "mean", "datapoint_true_pred", None, 1, "lexical_overlap"),
    # ReportMetric("HANS.valid.NE.S.acc", "hans_validation", "mean", "datapoint_true_pred", None, 1, "subsequence"),
    # ReportMetric("HANS.valid.NE.C.acc", "hans_validation", "mean", "datapoint_true_pred", None, 1, "constituent"),
    ReportMetric("MNLI.valid.M.acc", "mnli_validation_matched", "mean", "datapoint_true_pred", None, None, None),
    # ReportMetric("MNLI.valid.M.EASY.acc", "mnli_validation_matched", "mean", "datapoint_true_pred", "easy", None, None),
    ReportMetric("MNLI.valid.M.HARD.acc", "mnli_validation_matched", "mean", "datapoint_true_pred", "hard", None, None),
    # ReportMetric("MNLI.valid.MM.acc", "mnli_validation_mismatched", "mean", "datapoint_true_pred", None, None, None),
    # ReportMetric("MNLI.valid.MM.EASY.acc", "mnli_validation_mismatched", "mean", "datapoint_true_pred", "easy", None,
    #              None),
    # ReportMetric("MNLI.valid.MM.HARD.acc", "mnli_validation_mismatched", "mean", "datapoint_true_pred", "hard", None,
    #              None),
    # ReportMetric("SNLI.train.acc", "snli_train", "mean", "datapoint_true_pred", None, None, None),
    # ReportMetric("SNLI.valid.acc", "snli_validation", "mean", "datapoint_true_pred", None, None, None),
    # ReportMetric("SNLI.test.acc", "snli_test", "mean", "datapoint_true_pred", None, None, None),
    # ReportMetric("SNLI.test.EASY.acc", "snli_test", "mean", "datapoint_true_pred", "easy", None, None),
    # ReportMetric("SNLI.test.HARD.acc", "snli_test", "mean", "datapoint_true_pred", "hard", None, None),
]


def process_results(
        runs,
        config_keys_to_report: List[ConfigKey] = CONFIG_KEYS_TO_REPORT,
        early_stopping_metrics: List[EarlyStoppingMetric] = EARLY_STOPPING_METRICS,
        report_metrics: List[ReportMetric] = REPORT_METRICS,
        plots_dir_path: str = None,
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

            # TODO train does not have the same step as validation so the df will be empty
            # _steps = mnli_train_df.step.unique()
            # esm_step_train = _steps[_steps >= esm_step].min()

            report_metrics_csv_str += f"{run_id};{esm_step}"
            for rm in report_metrics:
                rm_df = df[(df.step == esm_step) & (df.datapoint_dataset == rm.dataset)]

                if not rm_df.empty and rm.dataset == "hans_validation" and rm.hans_heuristic is not None:
                    assert rm.hans_label in [0, 1]
                    rm_df = rm_df[(rm_df.datapoint_label == rm.hans_label)]
                    if rm.hans_heuristic == "lexical_overlap":
                        rm_df = rm_df[(rm_df.datapoint_idx < 10000)]
                    elif rm.hans_heuristic == "subsequence":
                        rm_df = rm_df[(rm_df.datapoint_idx >= 10000) & (rm_df.datapoint_idx < 20000)]
                    elif rm.hans_heuristic == "constituent":
                        rm_df = rm_df[(rm_df.datapoint_idx >= 20000)]
                    else:
                        raise ValueError()
                    assert len(rm_df) == 5000

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

            if plots_dir_path is not None:
                gamma = run['config']['focal_loss_gamma']

                mnli_train_df = df[(df.datapoint_dataset == "mnli_train")]
                _steps = mnli_train_df.step.unique()
                esm_step_train = _steps[_steps >= esm_step].min()
                mnli_train_df = mnli_train_df[mnli_train_df.step == esm_step_train].copy()

                mnli_matched_df = df[(df.step == esm_step) & (df.datapoint_dataset == "mnli_validation_matched")].copy()

                hans_train_df = df[(df.step == esm_step_train) & (df.datapoint_dataset == "hans_train")].copy()
                hans_validation_df = df[(df.step == esm_step) & (df.datapoint_dataset == "hans_validation")].copy()
                for hdf in [hans_train_df, hans_validation_df]:
                    hdf["label"] = hdf.datapoint_label.apply(lambda l: "entailment" if l == 0 else "non-entailment")
                    hdf["heuristic"] = hdf.datapoint_heuristic.apply(lambda h: INTEGER_TO_HEURISTIC[h])

                for _df in [mnli_train_df, mnli_matched_df, hans_train_df, hans_validation_df]:
                    hardness_to_str = {None: "NA", 0: "false", 1: "true"}
                    _df.hardness = _df.hardness.apply(lambda x: hardness_to_str[x])

                ##################
                # ~~~ Plot A ~~~ #
                ##################
                # Test
                xlim = (1e-5, 1e2)
                facet_grid = sns.displot(
                    mnli_matched_df,
                    x="datapoint_loss",
                    hue="hardness",
                    bins=100,
                    binrange=[np.log10(x) for x in xlim],
                    log_scale=(True, False),
                    palette="deep",
                    kind="hist",
                    # row="gamma",
                    height=2.0,
                    aspect=3.5,
                )
                facet_grid.set_axis_labels(f"loss, $\gamma={gamma}$", "count")
                v_line_position = -0.5 ** gamma * np.log(0.5)
                facet_grid.axes.flat[0].axvline(x=v_line_position, color='r', linestyle=':')
                v_line_position = -0.5 ** gamma * np.log(1 / 3)
                facet_grid.axes.flat[0].axvline(x=v_line_position, color='b', linestyle=':')
                plt.gca().set_xlim(xlim)

                plot_id = f"{esm.pretty_name} -- Plot A -- MNLI.valid.M -- v1 Loss-- {gamma:2.1f} -- {run_id:<70}.pdf"
                plot_id = plot_id.replace("epfl-optml/nli/", "")
                plt.savefig(os.path.join(plots_dir_path, plot_id))
                plt.close()

                # Test, CE loss scale
                xlim = (1e-5, 1e2)
                facet_grid = sns.displot(
                    mnli_matched_df,
                    x="ce_loss",
                    hue="hardness",
                    bins=120,
                    binrange=[np.log10(x) for x in xlim],
                    log_scale=(True, False),
                    palette="deep",
                    kind="hist",
                    # row="gamma",
                    height=2.0,
                    aspect=3.5,
                )
                facet_grid.set_axis_labels(f"normalized loss, $\gamma={gamma}$", "proportion")
                v_line_position = -0.5 ** 0.0 * np.log(0.5)
                facet_grid.axes.flat[0].axvline(x=v_line_position, color='r', linestyle=':')
                v_line_position = -0.5 ** 0.0 * np.log(1 / 3)
                facet_grid.axes.flat[0].axvline(x=v_line_position, color='b', linestyle=':')
                plt.gca().set_xlim(xlim)

                plot_id = f"{esm.pretty_name} -- Plot A -- MNLI.valid.M -- v2 CE_LOSS -- {gamma:2.1f} -- {run_id:<70}.pdf"
                plot_id = plot_id.replace("epfl-optml/nli/", "")
                plt.savefig(os.path.join(plots_dir_path, plot_id))
                plt.close()

                # Train
                xlim = (1e-5, 1e2)
                facet_grid = sns.displot(
                    mnli_train_df,
                    x="datapoint_loss",
                    bins=100,
                    binrange=[np.log10(x) for x in xlim],
                    log_scale=(True, False),
                    palette="deep",
                    kind="hist",
                    height=2.0,
                    aspect=3.5,
                )
                facet_grid.set_axis_labels(f"loss, $\gamma={gamma}$", "count")
                v_line_position = -0.5 ** gamma * np.log(0.5)
                facet_grid.axes.flat[0].axvline(x=v_line_position, color='r', linestyle=':')
                v_line_position = -0.5 ** gamma * np.log(1 / 3)
                facet_grid.axes.flat[0].axvline(x=v_line_position, color='b', linestyle=':')
                plt.gca().set_xlim(xlim)

                plot_id = f"{esm.pretty_name} -- Plot A -- MNLI.train -- v1 Loss-- {gamma:2.1f} -- {run_id:<70}.pdf"
                plot_id = plot_id.replace("epfl-optml/nli/", "")
                plt.savefig(os.path.join(plots_dir_path, plot_id))
                plt.close()

                # Train, HANS all
                xlim = (1e-5, 1e2)
                facet_grid = sns.displot(
                    hans_train_df,
                    x="datapoint_loss",
                    bins=50,
                    hue="label",
                    binrange=[np.log10(x) for x in xlim],
                    log_scale=(True, False),
                    palette="deep",
                    kind="hist",
                    height=2.0,
                    aspect=3.5,
                )
                facet_grid.set_axis_labels(f"loss, $\gamma={gamma}$", "count")
                v_line_position = -0.5 ** gamma * np.log(0.5)
                facet_grid.axes.flat[0].axvline(x=v_line_position, color='r', linestyle=':')
                v_line_position = -0.5 ** gamma * np.log(1 / 3)
                facet_grid.axes.flat[0].axvline(x=v_line_position, color='b', linestyle=':')
                plt.gca().set_xlim(xlim)

                plot_id = f"{esm.pretty_name} -- Plot A -- HANS.train -- v1 Loss for all -- {gamma:2.1f} -- {run_id:<70}.pdf"
                plot_id = plot_id.replace("epfl-optml/nli/", "")
                plt.savefig(os.path.join(plots_dir_path, plot_id))
                plt.close()

                # Test, HANS all
                xlim = (1e-5, 1e2)
                facet_grid = sns.displot(
                    hans_validation_df,
                    x="datapoint_loss",
                    bins=100,
                    hue="label",
                    binrange=[np.log10(x) for x in xlim],
                    log_scale=(True, False),
                    palette="deep",
                    kind="hist",
                    height=2.0,
                    aspect=3.5,
                )
                facet_grid.set_axis_labels(f"loss, $\gamma={gamma}$", "count")
                v_line_position = -0.5 ** gamma * np.log(0.5)
                facet_grid.axes.flat[0].axvline(x=v_line_position, color='r', linestyle=':')
                v_line_position = -0.5 ** gamma * np.log(1 / 3)
                facet_grid.axes.flat[0].axvline(x=v_line_position, color='b', linestyle=':')
                plt.gca().set_xlim(xlim)

                plot_id = f"{esm.pretty_name} -- Plot A -- HANS.validation -- v1 Loss for all -- {gamma:2.1f} -- {run_id:<70}.pdf"
                plot_id = plot_id.replace("epfl-optml/nli/", "")
                plt.savefig(os.path.join(plots_dir_path, plot_id))
                plt.close()

                # Test, HANS per heuristic
                xlim = (1e-5, 1e2)
                facet_grid = sns.displot(
                    hans_validation_df,
                    x="datapoint_loss",
                    bins=100,
                    col="heuristic",
                    hue="label",
                    binrange=[np.log10(x) for x in xlim],
                    log_scale=(True, False),
                    palette="deep",
                    kind="hist",
                    height=2.0,
                    aspect=2.0,
                )
                facet_grid.set_axis_labels(f"loss, $\gamma={gamma}$", "count")
                v_line_position = -0.5 ** gamma * np.log(0.5)
                for ax in facet_grid.axes.flat:
                    ax.axvline(x=v_line_position, color='r', linestyle=':')
                v_line_position = -0.5 ** gamma * np.log(1 / 3)
                for ax in facet_grid.axes.flat:
                    ax.axvline(x=v_line_position, color='b', linestyle=':')
                plt.gca().set_xlim(xlim)

                plot_id = f"{esm.pretty_name} -- Plot A -- HANS.validation -- v2 Loss per heuristic -- {gamma:2.1f} -- {run_id:<70}.pdf"
                plot_id = plot_id.replace("epfl-optml/nli/", "")
                plt.savefig(os.path.join(plots_dir_path, plot_id))
                plt.close()

                ##################
                # ~~~ Plot B ~~~ #
                ##################
                # Test v1
                xlim = (-0.01, 1.01)
                facet_grid = sns.displot(
                    mnli_matched_df,
                    x="datapoint_true_prob",
                    hue="hardness",
                    # col="gamma",
                    bins=100,
                    binrange=xlim,
                    log_scale=(False, True),
                    palette="deep",
                    height=2.0,
                    aspect=3.5,
                )
                facet_grid.set_axis_labels(f"probability of correct classification, $\gamma={gamma}$", "count")
                # v_line_position = 0.5
                # facet_grid.axes.flat[0].axvline(x=v_line_position, color='r', linestyle=':')
                # v_line_position = 1 / 3
                # facet_grid.axes.flat[0].axvline(x=v_line_position, color='b', linestyle=':')
                plt.gca().set_xlim(xlim)

                plot_id = f"{esm.pretty_name} -- Plot B -- MNLI.valid.M -- v1 -- {gamma:2.1f} -- {run_id:<70}.pdf"
                plot_id = plot_id.replace("epfl-optml/nli/", "")
                plt.savefig(os.path.join(plots_dir_path, plot_id))
                plt.close()

                # Train, HANS all
                xlim = (-0.01, 1.01)
                facet_grid = sns.displot(
                    hans_train_df,
                    x="datapoint_true_prob",
                    hue="label",
                    bins=50,
                    binrange=xlim,
                    log_scale=(False, True),
                    palette="deep",
                    height=2.0,
                    aspect=3.5,
                )
                facet_grid.set_axis_labels(f"probability of correct classification, $\gamma={gamma}$", "count")
                # v_line_position = 0.5
                # facet_grid.axes.flat[0].axvline(x=v_line_position, color='r', linestyle=':')
                # v_line_position = 1 / 3
                # facet_grid.axes.flat[0].axvline(x=v_line_position, color='b', linestyle=':')
                plt.gca().set_xlim(xlim)

                plot_id = f"{esm.pretty_name} -- Plot B -- HANS.train -- v1 all -- {gamma:2.1f} -- {run_id:<70}.pdf"
                plot_id = plot_id.replace("epfl-optml/nli/", "")
                plt.savefig(os.path.join(plots_dir_path, plot_id))
                plt.close()

                # Test, HANS all
                xlim = (-0.01, 1.01)
                facet_grid = sns.displot(
                    hans_validation_df,
                    x="datapoint_true_prob",
                    hue="label",
                    bins=100,
                    binrange=xlim,
                    log_scale=(False, True),
                    palette="deep",
                    height=2.0,
                    aspect=3.5,
                )
                facet_grid.set_axis_labels(f"probability of correct classification, $\gamma={gamma}$", "count")
                # v_line_position = 0.5
                # facet_grid.axes.flat[0].axvline(x=v_line_position, color='r', linestyle=':')
                # v_line_position = 1 / 3
                # facet_grid.axes.flat[0].axvline(x=v_line_position, color='b', linestyle=':')
                plt.gca().set_xlim(xlim)

                plot_id = f"{esm.pretty_name} -- Plot B -- HANS.validation -- v1 all -- {gamma:2.1f} -- {run_id:<70}.pdf"
                plot_id = plot_id.replace("epfl-optml/nli/", "")
                plt.savefig(os.path.join(plots_dir_path, plot_id))
                plt.close()

                # Test, HANS per heuristic
                xlim = (-0.01, 1.01)
                facet_grid = sns.displot(
                    hans_validation_df,
                    x="datapoint_true_prob",
                    bins=100,
                    col="heuristic",
                    hue="label",
                    binrange=xlim,
                    log_scale=(False, True),
                    palette="deep",
                    height=2.0,
                    aspect=2.0,
                )
                facet_grid.set_axis_labels(f"probability of correct classification, $\gamma={gamma}$", "count")
                # v_line_position = 0.5
                # for ax in facet_grid.axes.flat:
                #     ax.axvline(x=v_line_position, color='r', linestyle=':')
                # v_line_position = 1 / 3
                # for ax in facet_grid.axes.flat:
                #     ax.axvline(x=v_line_position, color='b', linestyle=':')
                plt.gca().set_xlim(xlim)

                plot_id = f"{esm.pretty_name} -- Plot B -- HANS.validation -- v2 per heuristic -- {gamma:2.1f} -- {run_id:<70}.pdf"
                plot_id = plot_id.replace("epfl-optml/nli/", "")
                plt.savefig(os.path.join(plots_dir_path, plot_id))
                plt.close()

                ##################
                # ~~~ Plot C ~~~ #
                ##################
                # Test v1
                xlim = (5e-6, 5e1)
                facet_grid = sns.displot(
                    mnli_matched_df,
                    x="datapoint_loss",
                    hue="hardness",
                    log_scale=(True, False),
                    linewidth=3,
                    palette="deep",
                    kind="ecdf",
                    # row="gamma",
                    height=2.0,
                    aspect=3.5,
                )
                facet_grid.set_axis_labels(f"loss, $\gamma={gamma}$", "proportion")
                plt.gca().set_xlim(xlim)

                plot_id = f"{esm.pretty_name} -- Plot C -- MNLI.valid.M -- v1 ECDF Log-Lin -- {gamma:2.1f} -- {run_id:<70}.pdf"
                plot_id = plot_id.replace("epfl-optml/nli/", "")
                plt.savefig(os.path.join(plots_dir_path, plot_id))
                plt.close()

                # Test v2
                xlim = (-0.15, 12)
                facet_grid = sns.displot(
                    mnli_matched_df,
                    x="datapoint_loss",
                    hue="hardness",
                    log_scale=(False, False),
                    linewidth=3,
                    palette="deep",
                    kind="ecdf",
                    # row="gamma",
                    height=2.0,
                    aspect=3.5,
                )
                facet_grid.set_axis_labels(f"loss, $\gamma={gamma}$", "proportion")
                plt.gca().set_xlim(xlim)

                plot_id = f"{esm.pretty_name} -- Plot C -- MNLI.valid.M -- v2 ECDF Lin-Lin -- {gamma:2.1f} -- {run_id:<70}.pdf"
                plot_id = plot_id.replace("epfl-optml/nli/", "")
                plt.savefig(os.path.join(plots_dir_path, plot_id))
                plt.close()

                # Test v3
                xlim = (-0.15, 12)
                facet_grid = sns.displot(
                    mnli_matched_df,
                    x="ce_loss",
                    hue="hardness",
                    log_scale=(False, False),
                    linewidth=3,
                    palette="deep",
                    kind="ecdf",
                    # row="gamma",
                    height=2.0,
                    aspect=3.5,
                )
                facet_grid.set_axis_labels(f"normalized loss, $\gamma={gamma}$", "proportion")
                plt.gca().set_xlim(xlim)

                plot_id = f"{esm.pretty_name} -- Plot C -- MNLI.valid.M -- v3 CE-LOSS ECDF Lin-Lin -- {gamma:2.1f} -- {run_id:<70}.pdf"
                plot_id = plot_id.replace("epfl-optml/nli/", "")
                plt.savefig(os.path.join(plots_dir_path, plot_id))
                plt.close()

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
    ## S01
    # "epfl-optml/nli/S1.01.A_e-03_model-bert_dataset-mnli_gamma-0.0_seed-72_09.25_01.53.02",
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
    ## S02
    # "epfl-optml/nli/S2.02_model-bert_dataset-mnli_gamma-0.0_seed-36_09.26_06.48.33",
    # "epfl-optml/nli/S2.03_model-bert_dataset-mnli_gamma-0.0_seed-180_09.26_03.37.35",
    # "epfl-optml/nli/S2.04_model-bert_dataset-mnli_gamma-0.0_seed-360_09.26_05.57.53",
    # "epfl-optml/nli/S2.05_model-bert_dataset-mnli_gamma-0.0_seed-54_09.26_07.10.18",
    # "epfl-optml/nli/S2.06_model-bert_dataset-mnli_gamma-0.5_seed-72_09.26_07.54.36",
    # "epfl-optml/nli/S2.07_model-bert_dataset-mnli_gamma-0.5_seed-36_09.26_06.48.33",
    # "epfl-optml/nli/S2.08_model-bert_dataset-mnli_gamma-0.5_seed-180_09.26_03.37.35",
    # "epfl-optml/nli/S2.09_model-bert_dataset-mnli_gamma-0.5_seed-360_09.26_05.57.53",
    # "epfl-optml/nli/S2.10_model-bert_dataset-mnli_gamma-0.5_seed-54_09.26_07.10.26",
    # "epfl-optml/nli/S2.11_model-bert_dataset-mnli_gamma-1.0_seed-72_09.26_07.54.53",
    # "epfl-optml/nli/S2.12_model-bert_dataset-mnli_gamma-1.0_seed-36_09.26_06.49.02",
    # "epfl-optml/nli/S2.13_model-bert_dataset-mnli_gamma-1.0_seed-180_09.26_03.37.35",
    # "epfl-optml/nli/S2.14_model-bert_dataset-mnli_gamma-1.0_seed-360_09.26_05.58.26",
    # "epfl-optml/nli/S2.15_model-bert_dataset-mnli_gamma-1.0_seed-54_09.26_07.10.46",
    # "epfl-optml/nli/S2.16_model-bert_dataset-mnli_gamma-2.0_seed-72_09.26_08.30.39",
    # "epfl-optml/nli/S2.17_model-bert_dataset-mnli_gamma-2.0_seed-36_09.26_06.49.02",
    # "epfl-optml/nli/S2.18_model-bert_dataset-mnli_gamma-2.0_seed-180_09.26_03.38.48",
    # "epfl-optml/nli/S2.19_model-bert_dataset-mnli_gamma-2.0_seed-360_09.26_05.58.26",
    # "epfl-optml/nli/S2.20_model-bert_dataset-mnli_gamma-2.0_seed-54_09.26_07.12.50",
    # "epfl-optml/nli/S2.21_model-bert_dataset-mnli_gamma-5.0_seed-72_09.26_08.36.02",
    # "epfl-optml/nli/S2.22_model-bert_dataset-mnli_gamma-5.0_seed-36_09.26_06.51.39",
    # "epfl-optml/nli/S2.23_model-bert_dataset-mnli_gamma-5.0_seed-180_09.26_03.40.47",
    # "epfl-optml/nli/S2.24_model-bert_dataset-mnli_gamma-5.0_seed-360_09.26_06.23.57",
    # "epfl-optml/nli/S2.25_model-bert_dataset-mnli_gamma-5.0_seed-54_09.26_07.13.40",
    # "epfl-optml/nli/S2.26_model-bert_dataset-mnli_gamma-10.0_seed-72_09.26_08.36.02",
    # "epfl-optml/nli/S2.27_model-bert_dataset-mnli_gamma-10.0_seed-36_09.26_06.59.50",
    # "epfl-optml/nli/S2.28_model-bert_dataset-mnli_gamma-10.0_seed-180_09.26_03.41.34",
    # "epfl-optml/nli/S2.29_model-bert_dataset-mnli_gamma-10.0_seed-360_09.26_06.24.42",
    # "epfl-optml/nli/S2.30_model-bert_dataset-mnli_gamma-10.0_seed-54_09.26_07.13.54",
    # "epfl-optml/nli/S2.31_model-bert_dataset-snli_gamma-0.0_seed-72_09.26_09.06.20",
    # "epfl-optml/nli/S2.32_model-bert_dataset-snli_gamma-0.0_seed-36_09.26_06.59.50",
    # "epfl-optml/nli/S2.33_model-bert_dataset-snli_gamma-0.0_seed-180_09.26_03.46.05",
    # "epfl-optml/nli/S2.34_model-bert_dataset-snli_gamma-0.0_seed-360_09.26_06.24.42",
    # "epfl-optml/nli/S2.35_model-bert_dataset-snli_gamma-0.0_seed-54_09.26_07.15.18",
    # "epfl-optml/nli/S2.36_model-bert_dataset-snli_gamma-0.5_seed-72_09.26_13.55.34",
    # "epfl-optml/nli/S2.37_model-bert_dataset-snli_gamma-0.5_seed-36_09.26_07.02.03",
    # "epfl-optml/nli/S2.38_model-bert_dataset-snli_gamma-0.5_seed-180_09.26_05.08.03",
    # "epfl-optml/nli/S2.39_model-bert_dataset-snli_gamma-0.5_seed-360_09.26_06.29.29",
    # "epfl-optml/nli/S2.40_model-bert_dataset-snli_gamma-0.5_seed-54_09.26_07.16.13",
    # "epfl-optml/nli/S2.41_model-bert_dataset-snli_gamma-1.0_seed-72_09.26_14.00.01",
    # "epfl-optml/nli/S2.42_model-bert_dataset-snli_gamma-1.0_seed-36_09.26_07.02.03",
    # "epfl-optml/nli/S2.43_model-bert_dataset-snli_gamma-1.0_seed-180_09.26_05.08.03",
    # "epfl-optml/nli/S2.44_model-bert_dataset-snli_gamma-1.0_seed-360_09.26_06.48.11",
    # "epfl-optml/nli/S2.45_model-bert_dataset-snli_gamma-1.0_seed-54_09.26_07.17.57",
    # "epfl-optml/nli/S2.46_model-bert_dataset-snli_gamma-2.0_seed-72_09.26_14.07.33",
    # "epfl-optml/nli/S2.47_model-bert_dataset-snli_gamma-2.0_seed-36_09.26_07.02.56",
    # "epfl-optml/nli/S2.48_model-bert_dataset-snli_gamma-2.0_seed-180_09.26_05.12.32",
    # "epfl-optml/nli/S2.49_model-bert_dataset-snli_gamma-2.0_seed-360_09.26_06.48.11",
    # "epfl-optml/nli/S2.50_model-bert_dataset-snli_gamma-2.0_seed-54_09.26_07.20.21",
    # "epfl-optml/nli/S2.51_model-bert_dataset-snli_gamma-5.0_seed-72_09.26_14.19.16",
    # "epfl-optml/nli/S2.52_model-bert_dataset-snli_gamma-5.0_seed-36_09.26_07.03.58",
    # "epfl-optml/nli/S2.53_model-bert_dataset-snli_gamma-5.0_seed-180_09.26_05.12.32",
    # "epfl-optml/nli/S2.54_model-bert_dataset-snli_gamma-5.0_seed-360_09.26_06.48.24",
    # "epfl-optml/nli/S2.55_model-bert_dataset-snli_gamma-5.0_seed-54_09.26_07.31.04",
    # "epfl-optml/nli/S2.56_model-bert_dataset-snli_gamma-10.0_seed-72_09.26_14.19.18",
    # "epfl-optml/nli/S2.57_model-bert_dataset-snli_gamma-10.0_seed-36_09.26_07.10.00",
    # "epfl-optml/nli/S2.58_model-bert_dataset-snli_gamma-10.0_seed-180_09.26_05.41.10",
    # "epfl-optml/nli/S2.59_model-bert_dataset-snli_gamma-10.0_seed-360_09.26_06.48.24",
    # "epfl-optml/nli/S2.60_model-bert_dataset-snli_gamma-10.0_seed-54_09.26_07.41.02",
    ## S03
    "epfl-optml/nli/S3.01_model-bert_nhans-100_gamma-0.0_seed-72_09.28_13.56.33",
    # "epfl-optml/nli/S3.02_model-bert_nhans-100_gamma-0.0_seed-36_09.28_06.21.59",
    # "epfl-optml/nli/S3.03_model-bert_nhans-100_gamma-0.0_seed-180_09.28_01.33.07",
    # "epfl-optml/nli/S3.04_model-bert_nhans-100_gamma-0.0_seed-360_09.28_04.50.36",
    # "epfl-optml/nli/S3.05_model-bert_nhans-100_gamma-0.0_seed-54_09.28_07.36.20",
    "epfl-optml/nli/S3.06_model-bert_nhans-100_gamma-1.0_seed-72_09.28_13.56.15",
    # "epfl-optml/nli/S3.07_model-bert_nhans-100_gamma-1.0_seed-36_09.28_06.24.24",
    # "epfl-optml/nli/S3.08_model-bert_nhans-100_gamma-1.0_seed-180_09.28_01.33.07",
    # "epfl-optml/nli/S3.09_model-bert_nhans-100_gamma-1.0_seed-360_09.28_04.50.36",
    # "epfl-optml/nli/S3.10_model-bert_nhans-100_gamma-1.0_seed-54_09.28_07.49.34",
    "epfl-optml/nli/S3.11_model-bert_nhans-100_gamma-2.0_seed-72_09.28_13.58.43",
    # "epfl-optml/nli/S3.12_model-bert_nhans-100_gamma-2.0_seed-36_09.28_06.38.38",
    # "epfl-optml/nli/S3.13_model-bert_nhans-100_gamma-2.0_seed-180_09.28_01.57.16",
    # "epfl-optml/nli/S3.14_model-bert_nhans-100_gamma-2.0_seed-360_09.28_05.09.04",
    # "epfl-optml/nli/S3.15_model-bert_nhans-100_gamma-2.0_seed-54_09.28_07.59.55",
    "epfl-optml/nli/S3.16_model-bert_nhans-100_gamma-5.0_seed-72_09.28_14.00.15",
    # # 17
    # # 18
    # "epfl-optml/nli/S3.19_model-bert_nhans-100_gamma-5.0_seed-360_09.28_05.13.51",
    # "epfl-optml/nli/S3.20_model-bert_nhans-100_gamma-5.0_seed-54_09.28_08.01.23",
    "epfl-optml/nli/S3.21_model-bert_nhans-1000_gamma-0.0_seed-72_09.28_14.06.45",
    # "epfl-optml/nli/S3.22_model-bert_nhans-1000_gamma-0.0_seed-36_09.28_07.01.06",
    # "epfl-optml/nli/S3.23_model-bert_nhans-1000_gamma-0.0_seed-180_09.28_01.58.09",
    # "epfl-optml/nli/S3.24_model-bert_nhans-1000_gamma-0.0_seed-360_09.28_05.15.53",
    # "epfl-optml/nli/S3.25_model-bert_nhans-1000_gamma-0.0_seed-54_09.28_08.01.23",
    "epfl-optml/nli/S3.26_model-bert_nhans-1000_gamma-1.0_seed-72_09.28_14.06.46",
    # "epfl-optml/nli/S3.27_model-bert_nhans-1000_gamma-1.0_seed-36_09.28_07.01.42",
    # "epfl-optml/nli/S3.28_model-bert_nhans-1000_gamma-1.0_seed-180_09.28_01.58.14",
    # "epfl-optml/nli/S3.29_model-bert_nhans-1000_gamma-1.0_seed-360_09.28_05.35.24",
    # "epfl-optml/nli/S3.30_model-bert_nhans-1000_gamma-1.0_seed-54_09.28_08.03.53",
    "epfl-optml/nli/S3.31_model-bert_nhans-1000_gamma-2.0_seed-72_09.28_14.09.44",
    # "epfl-optml/nli/S3.32_model-bert_nhans-1000_gamma-2.0_seed-36_09.28_07.28.12",
    # "epfl-optml/nli/S3.33_model-bert_nhans-1000_gamma-2.0_seed-180_09.28_01.58.47",
    # "epfl-optml/nli/S3.34_model-bert_nhans-1000_gamma-2.0_seed-360_09.28_06.03.32",
    # "epfl-optml/nli/S3.35_model-bert_nhans-1000_gamma-2.0_seed-54_09.28_08.08.30",
    "epfl-optml/nli/S3.36_model-bert_nhans-1000_gamma-5.0_seed-72_09.28_14.09.44",
    # "epfl-optml/nli/S3.37_model-bert_nhans-1000_gamma-5.0_seed-36_09.28_07.30.50",
    # "epfl-optml/nli/S3.38_model-bert_nhans-1000_gamma-5.0_seed-180_09.28_03.56.56",
    # "epfl-optml/nli/S3.39_model-bert_nhans-1000_gamma-5.0_seed-360_09.28_06.12.37",
    # "epfl-optml/nli/S3.40_model-bert_nhans-1000_gamma-5.0_seed-54_09.28_13.55.24",
    # "epfl-optml/nli/S3.41_model-bert_nhans-5000_gamma-0.0_seed-72_09.28_14.11.15",
    # # 42-45
]

if __name__ == '__main__':
    nice_print(HORSE)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cached_runs_pickle_path', type=str, default="logs/cached_runs_s02_plots.pkl",
                        help="If not None and if pickle file exists,"
                             "load the runs from the pickle file instead of querying wandb again.")
    parser.add_argument('--results_dir', type=str, default="logs/S02.plots",
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
        del (run["dataframes"])

        df["ce_loss"] = df["datapoint_true_prob"].apply(lambda x: -np.log(x))
        df["datapoint_dataset"] = df["datapoint_dataset"].apply(lambda x: INTEGER_TO_DATASET[x])
        run["dataframe"] = df

    ensure_dir(args.results_dir)
    plots_dir = os.path.join(args.results_dir, f"plots")
    ensure_dir(plots_dir)
    results = process_results(runs, plots_dir_path=plots_dir)

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
******************
*** DEPRECATED ***
******************

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
