#!/bin/bash
#SBATCH --chdir /scratch/izar/rajic/nli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#SBATCH -o /scratch/izar/rajic/nli/logs/slurm_logs/%x-%j.out

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)

# Modules
module purge
module load gcc/9.3.0-cuda
module load cuda/11.0.2

# Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate optml

# Run
date
printf "Run configured and environment setup. Gonna run now.\n\n"
python -m src.main \
  --wandb_entity epfl-optml \
  --experiment_name nli \
  --experiment_version \
  'S3.08_model-bert_nhans-100_gamma-1.0_seed-180' \
  --model_name bert \
  --dataset mnli \
  --seed 180 \
  --optimizer_name adamw \
  --scheduler_name polynomial \
  --adam_epsilon 1e-06 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --gradient_clip_val 1.0 \
  --tokenizer_model_max_length 128 \
  --focal_loss_gamma 1.0 \
  --accumulate_grad_batches 1 \
  --lr 2e-05 \
  --batch_size 32 \
  --n_epochs 10 \
  --early_stopping_patience 30 \
  --precision 16 \
  --num_hans_train_examples 100 \

echo FINISHED at $(date)

