"""
Script to gather the necessary data from wandb.ai to create the tables reported in the paper.
"""

import pandas as pd
import wandb


def get_summary_from_run(run):
    """
    Get the numbers reported for the epoch with the best `Valid/mnli_acc_epoch` value.

    :param run: The wandb run object, `wandb.apis.public.Run`.
    :return: A pandas dataframe with the relevant numbers.
    """
    train_keys = [
        'epoch',
        "Train/mnli_acc_epoch",
        'Train/mnli_loss_epoch',
    ]
    train_lines = run.scan_history(keys=train_keys, page_size=100000000)
    train_df = pd.DataFrame.from_records(train_lines)

    val_keys = [
        '_step',
        'epoch',
        'Valid/mnli_acc_epoch',
        'Valid/mnli_loss_epoch',
        'Valid/hans_acc',
        'Valid/hans_loss',
        'Valid/Hans_acc/entailment_lexical_overlap',
        'Valid/Hans_acc/entailment_subsequence',
        'Valid/Hans_acc/entailment_constituent',
        'Valid/Hans_acc/non_entailment_lexical_overlap',
        'Valid/Hans_acc/non_entailment_subsequence',
        'Valid/Hans_acc/non_entailment_constituent',
    ]
    val_lines = run.scan_history(keys=val_keys, page_size=10000000000000)
    val_df = pd.DataFrame.from_records(val_lines)

    df = pd.merge(left=train_df, right=val_df)
    best_epoch = df.sort_values(by=["Valid/mnli_acc_epoch"], ascending=False).iloc[0]
    return best_epoch


run_filters = {
    "all": ["S7", "S8", "S9", "S10", "S11"],
    "exp1": ["S7"],
    "exp2": ["S7.03", "S8.01", "S9"],
    "exp3": ["S7.03", "S8.01", "S10"],
    "exp4": ["S7.03", "S8.01", "S11"],
}


def main(exp_id):
    assert exp_id in run_filters

    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("user72/bertfornli-exp1")
    summary_list = []
    config_list = []
    name_list = []
    for run in runs:
        # if run.name[:3] not in :
        if not any([run.name.startswith(f) for f in run_filters[exp_id]]):
            continue

        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        # summary_list.append(run.summary._json_dict)

        # Use custom summary
        summary_list.append(get_summary_from_run(run))

        # run.config is the input metrics.
        # We remove special values that start with _.
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        config_list.append(config)

        # run.name is the name of the run.
        name_list.append(run.name)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({'name': name_list})
    all_df = pd.concat([name_df, config_df, summary_df], axis=1)

    return all_df


if __name__ == '__main__':
    for exp_id in run_filters.keys():
        df = main(exp_id)
        df.to_csv(f"{exp_id}.csv")
