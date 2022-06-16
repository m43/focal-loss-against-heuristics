#!/bin/bash
#SBATCH --chdir /scratch/izar/rajic/nli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00

#SBATCH -o ./logs/slurm_logs/slurm-sbatch_09-03-%j.out

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
python -m src.main --experiment_name bertfornli-exp1 --experiment_version 'S9.03_gamma=2.0_adamw-1e-06_bs=1x32_lr=2e-06_wd=0.01_e=10_prec=16' --optimizer_name adamw --scheduler_name polynomial --gpus -1 --adam_epsilon 1e-06 --weight_decay 0.01 --warmup_ratio 0.1 --gradient_clip_val 1.0 --tokenizer_model_max_length 128 --focal_loss_gamma 2.0 --accumulate_grad_batches 1 --lr 2e-06 --batch_size 32 --n_epochs 10 --early_stopping_patience 10 --precision 16 --num_hans_train_examples 0 --bert_hidden_dropout_prob 0.1 --bert_attention_probs_dropout_prob 0.1 --bert_classifier_dropout 0.0
echo FINISHED at $(date)

