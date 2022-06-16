"""
Script that collects the necessary data from wandb.ai and creates a ridgeplot
with the loss distribution across sample type and training epochs.
"""

import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

WANDB_RUN_PATHS = [
    "user72/bertfornli-exp1/S7.03_gamma-0.0_adamw-1e-06_lr-2e-05_e-10_precision-32_06.11_11.18.07",
]


def load_all_tables(wandb_run, artifact_filter_str, csv_file_path_in_artifact, cache_path="/tmp/wandb"):
    artifacts = [a for a in wandb_run.logged_artifacts() if artifact_filter_str in a.name]

    df_all = pd.DataFrame({})
    for artifact in artifacts:
        artifact_dir = artifact.download(os.path.join(cache_path, f"{artifact.name}"))
        csv_path = os.path.join(artifact_dir, csv_file_path_in_artifact)

        df = pd.read_csv(csv_path)
        df = df.reset_index()
        df["artifact_name"] = artifact.name
        df_all = pd.concat([df_all, df])

    df_all = df_all.reset_index()
    return df_all


def ridgeline_plot(df, save_path, show_plot=True, dpi=200):
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(len(df.type_str.unique()), rot=-.25, light=.7)
    g = sns.FacetGrid(
        df, row="epoch", hue="type_str", col="type_str",
        sharex=True, sharey=True,
        aspect=2, height=3  # , palette=pal,
    )

    # Draw the densities in a few steps
    bw_adjust = 0.6
    g.map(sns.kdeplot, "loss", bw_adjust=bw_adjust,
          clip_on=False, fill=True, alpha=.8, linewidth=0.5)
    g.map(sns.kdeplot, "loss", bw_adjust=bw_adjust,
          clip_on=False, color="k", lw=1)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=1.5, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label, **kwargs):
        years = x.unique().tolist()  # TODO very hacky
        assert len(years) == 1
        ax = plt.gca()
        ax.text(0, .2, f"e:{years[0]}", fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "epoch")
    g.set_axis_labels("loss")  # TODO very hacky
    g.add_legend()
    plt.figtext(0.0, 0.5, 'Density', fontsize=18, ha='center', rotation="vertical")
    plt.figtext(0.5, 1.0, 'Loss distribution across sample type and training epochs', fontsize=24, ha='center')

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set(yticks=[])
    g.set(ylabel="")
    g.set_titles("")
    g.despine(bottom=True, left=True)

    g.savefig(save_path, dpi=dpi, format='png')

    g.tight_layout()

    if show_plot:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # Initialize wandb
    wapi = wandb.Api()
    wruns = [wapi.run(wrp) for wrp in WANDB_RUN_PATHS]

    for wrun in wruns:
        # wandb_logger = wandb.init(entity=wrun.entity, project=wrun.project, id=wrun.id, resume=True)

        # MNLI dataframes
        train_mnli_df = load_all_tables(
            wandb_run=wrun,
            artifact_filter_str="Train-mnli_epoch_end_df",
            csv_file_path_in_artifact='df.csv',
        )
        valid_mnli_df = load_all_tables(
            wandb_run=wrun,
            artifact_filter_str="Valid-mnli_epoch_end_df",
            csv_file_path_in_artifact='df.csv',
        )

        # HANS dataframes
        valid_hans_df = load_all_tables(
            wandb_run=wrun,
            artifact_filter_str="Valid-hans_epoch_end_df",
            csv_file_path_in_artifact='df.csv',
        )

        # ~~~ The Ridgeline plot ~~~ #
        ridgeline_plot(train_mnli_df[["epoch", "loss", "type_str"]], f"{wrun.name}__train__loss_ridgeline.png")
        ridgeline_plot(valid_mnli_df[["epoch", "loss", "type_str"]], f"{wrun.name}__valid__loss_ridgeline.png")
