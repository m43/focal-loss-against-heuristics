# Using Focal Loss to Fight Shallow Heuristics

[Frano Rajič](https://m43.github.io/), [Ivan Stresec](https://www.github.com/istresec), [Axel Marmet](https://github.com/axelmarmet), [Tim Poštuvan](https://github.com/timpostuvan)

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![MIT License](https://img.shields.io/github/license/m43/focal-loss-against-heuristics)](https://github.com/m43/focal-loss-against-heuristics/blob/main/LICENSE)

There is no such thing as a perfect dataset. In some datasets, deep neural networks discover underlying heuristics that allow them to take shortcuts in the learning process, resulting in poor generalization capability. Instead of using standard cross-entropy, we explore whether a modulated version of cross-entropy called focal loss can constrain the model so as not to use heuristics and improve generalization performance. Our experiments in natural language inference show that focal loss has a regularizing impact on the learning process, increasing accuracy on out-of-distribution data, but slightly decreasing performance on in-distribution data. Despite the improved out-of-distribution performance, we demonstrate the shortcomings of focal loss and its inferiority in comparison to the performance of methods such as unbiased focal loss and self-debiasing ensembles.

## Table of Contents

  - [Environment set-up](#environment-set-up)
  - [Reproducing results](#reproducing-results)
  - [Experiment logs](#experiment-logs)
  - [Project structure](#project-structure)
  - [License](#license)

## Environment set-up

This codebase has been tested with the packages and versions specified in `requirements.txt` and Python 3.9 on Manjaro Linux and Red Hat Enterprise Linux Server 7.7 (Maipo).

Start by cloning the repository:
```bash
git clone https://github.com/m43/optml-proj.git
```

With the repository cloned, we recommend creating a new [conda](https://docs.conda.io/en/latest/) virtual environment:
```bash
conda create -n optml python=3.9 -y
conda activate optml
```

Then, install [PyTorch](https://pytorch.org/) 1.11.0 and [torchvision](https://pytorch.org/vision/stable/index.html)
0.12.0. For example with CUDA 11 support:
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

Finally, install the required packages:
```bash
pip install -r requirements.txt
```

## Reproducing results

To reproduce any of the experiments, find the related run configuration in [`scripts/sbatch`](scripts/sbatch). The correspondence between the experimental results shown in the paper and the run configurations is described below. It can also be reverse-engineered from [`scripts/generate_sbatch.py`](scripts/generate_sbatch.py), which was used to generate the experimental run configurations.

Denote with `S2` all the runs that correspond to the 2nd set of experiments ([`scripts/sbatch/sbatch_02`](scripts/sbatch/sbatch_02)). Denote a particular run like `S3.01` -- the first run in the 3rd set of experiments ([`scripts/sbatch/sbatch_03/72.S3.01_B0100_gamma-0.0_seed-72.sh`](scripts/sbatch/sbatch_03/72.S3.01_B0100_gamma-0.0_seed-72.sh)). With this notation, the correspondence between runs and paper experiments is the following:
1. Impact of Focal Loss:
   - `S2`: BERT trained on MNLI and SNLI
   - `S4`: InferSent trained on MNLI and SNLI
2. Impact of Focal Loss when Adding HANS Samples to Training:
   - `S3`: BERT trained on MNLI with varying amounts of HANS samples added

To reproduce experiment `S2.21`, which corresponds to BERT trained on MNLI with a value of `gamma=5` and `seed=72`, run the python command from [`scripts/sbatch/sbatch_02/72.S2.21_BM_gamma-5.0_seed-72.sh`](scripts/sbatch/sbatch_02/72.S2.21_BM_gamma-5.0_seed-72.sh). A multi-GPU setup using DP is not optimized well and will not reproduce the exact same numbers as reported in the paper. Also, you might want to replace the `--wandb_entity epfl-optml` flag with your wandb entity (like your username, e.g., my username is [`user72`](https://wandb.ai/user72)) if you do not have access to the `epfl-optml` wandb team (which is most likely the case). For example, the command could look like this with my username:
```bash
python -m src.main \
  --wandb_entity user72 \
  --experiment_name nli \
  --experiment_version \
  'S2.21_model-bert_dataset-mnli_gamma-5.0_seed-72' \
  --model_name bert \
  --dataset mnli \
  --seed 72 \
  --optimizer_name adamw \
  --scheduler_name polynomial \
  --adam_epsilon 1e-06 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --gradient_clip_val 1.0 \
  --tokenizer_model_max_length 128 \
  --focal_loss_gamma 5.0 \
  --accumulate_grad_batches 1 \
  --lr 2e-05 \
  --batch_size 32 \
  --n_epochs 10 \
  --early_stopping_patience 30 \
  --precision 16 \
  --num_hans_train_examples 0 \
```

## Experiment logs

You can find the logs of all BERT experiments publicly available on [Weights & Biases](https://wandb.ai/epfl-optml/nli). For example, the baseline run with the best hyperparameters is [S2.21](https://wandb.ai/epfl-optml/nli/runs/S2.21_model-bert_dataset-mnli_gamma-5.0_seed-72_09.26_08.36.02). To filter interesting runs, you can use a regex like `S2` to filter the BERT runs related to the _Impact of focal loss_ experiment. Experiments that used InferSent did not use wandb for logging results but dumped the results to disk.

## Project structure

```
$ tree
.
├── LICENSE
├── README.md
├── requirements.txt
├── scripts/     # Scripts for generating experiments and collecting results
│   ├── compute_hardness_for_snli.py          # Script that computes the hardness annotations for the SNLI test subset
│   ├── compute_hardness_from_robustnli.py    # Script that computes the hardness annotations for the MNLI validation dataset
│   │
│   ├── collect_results_infersent_csv.py   # Script that collects InferSent CSV logs and generates the summaries we are interested in
│   ├── collect_results_wandb.py           # Script that collects the results from WANDB and generates the summaries
│   ├── merge_csv_files.py                 # Script to merge a list of CSV files into one CSV file
│   ├── collect_result_stddev.py           # Script that loads the CSV result summaries and computes the relevant standard deviations
│   │
│   ├── draw_focal_loss.py                 # Script that plots the focal loss paper figure
│   │
│   ├── evaluate_bert_wandb_checkpoint_on_hans.py    # Script that evaluates a BERT checkpoint from WANDB and evaluates it on HANS
│   │
│   ├── generate_sbatch.py        # Generate experiment configurations
│   └── sbatch/   # Batches of experiments
│       ├── sbatch_01/    # Batch 1
│       ├── (...)         # More batches
│       └── sbatch_04/    # Batch 4
│           ├── 72.S4.01_(...).sh    # First experiment in batch 4, used a seed of 72
│           ├── (...)                # More experiments in batch 4
│           └── 54.S4.60_(...).sh    # Last experiment in batch 4, used a seed of 54
│
├── src/     # Method codebase
│   ├── constants.py   # Global constants like enums
│   ├── main.py        # Training and argument parsing
│   │
│   ├── dataset/     # Dataset and dataloading
│   │   ├── mnli_datamodule.py   # MultiNLI with HANS datamodule
│   │   ├── snli_datamodule.py   # SNLI with HANS datamodule
│   │   └── util.py              # Utilites shared among datasets
│   │
│   ├── infersent    # InferSent codebase, adapted from (Mahabadi et al., 2020): https://github.com/rabeehk/robust-nli
│   │
│   ├── model/       # Model related code
│   │   ├── focalloss.py       # Focal loss implementation
│   │   └── nlitransformer.py  # BERT for NLI implementation
│   │
│   └── utils/       # General utilities
│       └── util.py       # Utilites like get_logger and horse plotting
│
└── tests     # Tests of some components
    └── test_focalloss.py     # Test that focal loss computes expected values
```

## License

Distributed under the MIT License. See LICENSE for more information.
