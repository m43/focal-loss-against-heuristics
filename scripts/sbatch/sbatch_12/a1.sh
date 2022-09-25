#!/bin/bash
#SBATCH --chdir /scratch/izar/rajic/nli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

#SBATCH -o ./logs/slurm_logs/%x-%j.out

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
  --experiment_name bertfornli-exp1 \
  --experiment_version 'a1-adamw-lr5e-5-clark' \
  --optimizer_name adamw \
  --seed 36 \
  --dataset mnli \
  --scheduler_name polynomial \
  --adam_epsilon 1e-06 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --gradient_clip_val 1.0 \
  --tokenizer_model_max_length 128 \
  --focal_loss_gamma 0 \
  --accumulate_grad_batches 1 \
  --lr 5e-05 \
  --batch_size 32 \
  --n_epochs 3 \
  --early_stopping_patience 30 \
  --precision 32 \
  --num_hans_train_examples 0 \

echo FINISHED at $(date)

