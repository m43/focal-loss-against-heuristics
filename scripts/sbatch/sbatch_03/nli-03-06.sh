#!/bin/bash
#SBATCH --chdir /scratch/izar/rajic/nli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00

#SBATCH -o ./logs/slurm_logs/slurm-sbatch_03-06-%j.out

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
python -m src.main --experiment_name bertfornli-exp1 --experiment_version 'S3.06_gamma=2.0_wdecay=0.0001_gradclip=0.0_lr=0.001_bs=32_accum=64_warmup=5' --gpus -1 --focal_loss_gamma 2 --accumulate_grad_batches 64 --lr 1e-3 --batch_size 32 --warmup 958 --n_epochs 15 --early_stopping_patience 10 --weight_decay 0.0001 --gradient_clip 0 --adam_epsilon 1e-8 --precision 16
echo FINISHED at $(date)
