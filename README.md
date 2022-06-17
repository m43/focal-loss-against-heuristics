# Focal Loss Does Not Fight Shallow Heuristics

[![Python 3.7](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/axelmarmet/optml-proj/blob/master/LICENSE)

    There is no such thing as a perfect dataset.
    Sometimes there are underlying heuristics that allow deep neural networks to take shortcuts in the learning process, resulting in poor generalization.
    Instead of using cross entropy, we explore whether focal loss constrains the model not to use heuristics.
    Our findings suggest that focal loss gives slightly worse generalization results and is not more sample-efficient.

## Table of Contents

  - [Environment set-up](#environment-set-up)
  - [Reproducing results](#reproducing-results)
  - [Experiment logs](#experiment-logs)
  - [Project structure](#project-structure)
  - [License](#license)
  - [Authors](#authors)


## Environment set-up

This codebase has been tested with the packages and versions specified in `conda.yml` and Python 3.9.

Start by cloning the repository:
```bash
git clone https://github.com/axelmarmet/optml-proj.git
```

With the repository cloned, we recommend creating a new [conda](https://docs.conda.io/en/latest/) virtual environment:
```bash
conda create -n optml python=3.9 -y
conda activate optml
```

Then, install [PyTorch](https://pytorch.org/) 1.11.0 and [torchvision](https://pytorch.org/vision/stable/index.html)
0.12.0. For example:
```bash
conda install pytorch=1.11.0 torchvision=0.12.0 -c pytorch -y
```

Finally, install the required packages:
```bash
pip install -r requirements.txt
```

## Reproducing results

To reproduce any of the experiments, find the related run configuration in `scripts/sbatch`. The experimental results shown in the paper correspond to the following runs:

Denote with `S7` all the runs that correspond to the 7th set of experiments (`scripts/sbatch/sbatch_07`). Denote a particular run like `S8.01` -- the first run in the 8th set of experiments (`scripts/sbatch/sbatch_08/nli-08-01.sh`). With this notation, the correspondence between runs and paper experiments is the following:
1. Reproduction of HANS results:
   1. `S7`
2. Hyperparameter search for focal loss:
   1. `S7.03`
   2. `S8.01`
   3. `S9`
3. Impact of focal loss:
   1. `S7.03`
   2. `S8.01`
   3. `S10`
4. Impact of adding HANS examples to the training
   1. `S7.03`
   2. `S8.01`
   3. `S11`



For example, to reproduce the best baseline configuration `S7.03`, run the command from `scripts/sbatch/sbatch_07/nli-07-03.sh`, which is the following:
```bash
python -m src.main --experiment_name bertfornli-exp1 \
  --experiment_version 'S7.03_gamma=0.0_adamw-1e-06_lr=2e-05_e=10_precision=32' \
  --optimizer_name adamw --scheduler_name polynomial --gpus -1 --adam_epsilon 1e-06 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --gradient_clip_val 1.0 \
  --tokenizer_model_max_length 128 \
  --focal_loss_gamma 0.0 \
  --accumulate_grad_batches 1 \
  --lr 2e-05 \
  --batch_size 32 \
  --n_epochs 10 \
  --early_stopping_patience 10 \
  --precision 32 \
  --num_hans_train_examples 0
```

## Experiment logs

You can find the logs of all experiments publicly available on [Weights & Biases](https://wandb.ai/user72/bertfornli-exp1?workspace=user-user72). For example, the baseline run with the best hyperparameters is [S7.03](https://wandb.ai/user72/bertfornli-exp1/runs/S7.03_gamma-0.0_adamw-1e-06_lr-2e-05_e-10_precision-32_06.11_11.18.07). To filter interesting runs, you can use a regex like `S7.03|S8.01|S10` to filter the runs related to the _Impact of focal loss_ experiment.

## Project structure

```
$ tree
.
│
├── scripts/     # Scripts for generating experiments and collecting results
│   ├── get_data_from_wandb.py    # Collect results from wandb, needed for tables
│   ├── visualize_results.py      # Fetch data from wandb and create a ridgeplot
│   ├── generate_sbatch.py        # Generate experiment configurations
│   │
│   └── sbatch/   # Batches of experiments
│       ├── sbatch_01/    # Batch 1
│       ├── (...)         # More batches
│       └── sbatch_11/    # Batch 11
│           ├── nli-11-01.sh    # First experiment in batch 11
│           ├── (...)           # More experiments in batch 11
│           └── nli-11-06.sh    # Last experiment in batch 11
│
├── src/     # Method codebase
│   ├── constants.py   # Global constants like enums
│   ├── main.py        # Training and argument parsing
│   │
│   ├── dataset/     # Dataset and dataloading
│   │   └── datamodule.py    # MultiNLI and HANS datamodule
│   │
│   ├── model/       # Model related code
│   │   ├── focalloss.py       # Focal loss implementation
│   │   └── nlitransformer.py  # BERT for NLI implementation
│   │
│   ├── notebooks/       # Notebooks
│   │   └── histogram_plots.ipynb       # Notebook used for plotting
│   │
│   └── utils/       # General utilities
│       └── util.py       # Utilites like get_logger and horse plotting
│
└── tests     # Tests of some components
    └── test_focalloss.py     # Test that focal loss computes expected values
```

## License

Distributed under the MIT License. See LICENSE for more information.

## Authors

- [Axel Marmet](https://github.com/axelmarmet)
- [Tim Poštuvan](https://github.com/timpostuvan)
- [Frano Rajič](https://www.github.com/m43)

