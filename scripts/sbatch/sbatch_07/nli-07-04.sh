#!/bin/bash
#SBATCH --chdir /scratch/izar/rajic/nli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=18:00:00

#SBATCH -o ./logs/slurm_logs/slurm-sbatch_07-04-%j.out

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
conda activate OptML

# Run
date
printf "Run configured and environment setup. Gonna run now.\n\n"
python -m src.main --experiment_name bertfornli-exp1 --experiment_version 'S7.04_gamma=0.0_adamw-1e-08_lr=0.0001_e=10_precision=32' --optimizer_name adamw --scheduler_name polynomial --gpus -1 --adam_epsilon 1e-08 --weight_decay 0.01 --warmup_ratio 0.1 --gradient_clip_val 1.0 --tokenizer_model_max_length 128 --focal_loss_gamma 0.0 --accumulate_grad_batches 1 --lr 0.0001 --batch_size 32 --n_epochs 10 --early_stopping_patience 30 --precision 32 --num_hans_train_examples 0
echo FINISHED at $(date)

