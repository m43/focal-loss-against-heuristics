import os
import pathlib

DEBUG_HEADER = """#SBATCH --chdir /scratch/izar/rajic/nli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=50G
#SBATCH --partition=debug
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00
"""

PRODUCTION_HEADER = """#SBATCH --chdir /scratch/izar/rajic/nli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
"""


def fill_template(i, sbatch_id, command, debug):
    return f"""#!/bin/bash
{DEBUG_HEADER if debug else PRODUCTION_HEADER}
#SBATCH -o ./logs/slurm_logs/slurm-{sbatch_id}-{i:02d}-%j.out

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
printf "Run configured and environment setup. Gonna run now.\\n\\n"
{command}
echo FINISHED at $(date)

"""


sbatch_configurations = {
    "sbatch_01": {
        "debug": False,
        "commands": [
            "python -m src.main"
            " --experiment_name bertfornli-exp1"
            " --experiment_version 'S1.01_g=0.0_e=100_lr=0.001_bs=32_accumulated=4'"
            " --gpus -1"
            " --focal_loss_gamma 0.0"
            " --accumulate_grad_batches 4"
            " --lr 1e-3"
            " --batch_size 32"
            " --warmup_steps 15000"
            " --n_epochs 100"
            " --early_stopping_patience 50"
            " --weight_decay 0.0"
            " --gradient_clip 0.0"
            " --adam_epsilon 1e-8"
            " --precision 16",

            "python -m src.main"
            " --experiment_name bertfornli-exp1"
            " --experiment_version 'S1.02_g=0.0_e=100_lr=0.001_bs=32_accumulated=16'"
            " --gpus -1"
            " --focal_loss_gamma 0.0"
            " --accumulate_grad_batches 16"
            " --lr 1e-3"
            " --batch_size 32"
            " --warmup_steps 3750"
            " --n_epochs 100"
            " --early_stopping_patience 50"
            " --weight_decay 0.0"
            " --gradient_clip 0.0"
            " --adam_epsilon 1e-8"
            " --precision 16",

            "python -m src.main"
            " --experiment_name bertfornli-exp1"
            " --experiment_version 'S1.03_g=0.0_e=100_lr=0.00002_bs=32_accumulated=4'"
            " --gpus -1"
            " --focal_loss_gamma 0.0"
            " --accumulate_grad_batches 4"
            " --lr 2e-5"
            " --batch_size 32"
            " --warmup_steps 15000"
            " --n_epochs 100"
            " --early_stopping_patience 50"
            " --weight_decay 0.0"
            " --gradient_clip 0.0"
            " --adam_epsilon 1e-8"
            " --precision 16",

            "python -m src.main"
            " --experiment_name bertfornli-exp1"
            " --experiment_version 'S1.04_g=0.0_e=100_lr=0.00002_bs=32_accumulated=16'"
            " --gpus -1"
            " --focal_loss_gamma 0.0"
            " --accumulate_grad_batches 16"
            " --lr 2e-5"
            " --batch_size 32"
            " --warmup_steps 3750"
            " --n_epochs 100"
            " --early_stopping_patience 50"
            " --weight_decay 0.0"
            " --gradient_clip 0.0"
            " --adam_epsilon 1e-8"
            " --precision 16",
        ]
    },
    "sbatch_02": {
        "debug": False,
        "commands": [
            f"python -m src.main"
            f" --experiment_name bertfornli-exp1"
            f" --experiment_version"
            f" 'S2.{i:02}_gamma={gamma:.1f}_wdecay={weight_decay}_gradclip={grad_clip}_lr=0.001_bs=32_accum=4'"
            f" --gpus -1"
            f" --focal_loss_gamma {gamma}"
            f" --accumulate_grad_batches 4"
            f" --lr 1e-3"
            f" --batch_size 32"
            f" --warmup_steps 15000"
            f" --n_epochs 15"
            f" --early_stopping_patience 10"
            f" --weight_decay {weight_decay}"
            f" --gradient_clip {grad_clip}"
            f" --adam_epsilon 1e-8"
            f" --precision 16"
            for i, (gamma, weight_decay, grad_clip) in enumerate([
                (2, 0, 0),
                (2, 0, 1),
                (2, 0.000001, 0),
                (2, 0.000001, 1),
                (5, 0, 0),
                (10, 0, 0),
            ])
        ]
    },
    "sbatch_03": {
        "debug": False,
        "commands": [
            f"python -m src.main"
            f" --experiment_name bertfornli-exp1"
            f" --experiment_version"
            f" 'S3.{i + 1:02}_gamma={gamma:.1f}_wdecay={weight_decay}_gradclip=0.0_lr=0.001_bs=32_accum={accu}_warmup=5'"
            f" --gpus -1"
            f" --focal_loss_gamma {gamma}"
            f" --accumulate_grad_batches {accu}"
            f" --lr 1e-3"
            f" --batch_size 32"
            f" --warmup_steps {(5 * 12272) // accu}"
            f" --n_epochs 15"
            f" --early_stopping_patience 10"
            f" --weight_decay {weight_decay}"
            f" --gradient_clip 0"
            f" --adam_epsilon 1e-8"
            f" --precision 16"
            for i, (gamma, weight_decay, accu) in enumerate([
                # Warmup of 5 epochs:
                #   392702 train samples
                #   ceil(392702/32) = 12272 batches
                #   5 epochs = 5*12272 batches = 61360 batches
                #   1 batch = 1 update step, if no gradients are accumulated
                # We accumulate gradients, thus: accumulate_grad_batches == 1 update step
                #   5 epochs --> (5*12272 / accumulate_grad_batches) update steps
                (2, 1e-5, 16),
                (2, 1e-5, 32),
                (2, 1e-5, 64),
                (2, 1e-4, 16),
                (2, 1e-4, 32),
                (2, 1e-4, 64),
            ])
        ]
    },
    "sbatch_04": {
        "debug": False,
        "commands": [
            f"python -m src.main"
            f" --experiment_name bertfornli-exp1"
            f" --experiment_version"
            f" 'S4.{i + 1:02}_gamma={gamma:.1f}_n-hans={n_hans}_wdecay={weight_decay}_gradclip=0.0_lr=0.001_bs=32_accum={accu}_warmup=5'"
            f" --gpus -1"
            f" --focal_loss_gamma {gamma}"
            f" --accumulate_grad_batches {accu}"
            f" --lr 1e-3"
            f" --batch_size 32"
            f" --warmup_steps {(5 * 12272) // accu}"
            f" --n_epochs 15"
            f" --early_stopping_patience 10"
            f" --weight_decay {weight_decay}"
            f" --gradient_clip 0"
            f" --adam_epsilon 1e-8"
            f" --precision 16"
            f" --num_hans_train_examples {n_hans}"
            for i, (gamma, weight_decay, accu, n_hans) in enumerate([
                # Warmup of 5 epochs:
                #   392702 train samples
                #   ceil(392702/32) = 12272 batches
                #   5 epochs = 5*12272 batches = 61360 batches
                #   1 batch = 1 update step, if no gradients are accumulated
                # We accumulate gradients, thus: accumulate_grad_batches == 1 update step
                #   5 epochs --> (5*12272 / accumulate_grad_batches) update steps
                (0, 0.0, 16, 15000),
                (2, 0.0, 16, 15000),
            ])
        ]
    },
    "sbatch_05": {
        "debug": False,
        "commands": [
            f"python -m src.main"
            f" --experiment_name bertfornli-exp1"
            f" --experiment_version"
            f" 'S5.{i + 1:02}_gamma={gamma:.1f}_n-hans={n_hans}_wdecay={weight_decay}_gradclip=0.0_lr=0.001_bs=32_accum={accu}_warmup=5'"
            f" --gpus -1"
            f" --focal_loss_gamma {gamma}"
            f" --accumulate_grad_batches {accu}"
            f" --lr 1e-3"
            f" --batch_size 32"
            f" --warmup_steps {(5 * 12272) // accu}"
            f" --n_epochs 15"
            f" --early_stopping_patience 10"
            f" --weight_decay {weight_decay}"
            f" --gradient_clip 0"
            f" --adam_epsilon 1e-8"
            f" --precision 16"
            f" --num_hans_train_examples {n_hans}"
            for i, (gamma, weight_decay, accu, n_hans) in enumerate([
                # Warmup of 5 epochs:
                #   392702 train samples
                #   ceil(392702/32) = 12272 batches
                #   5 epochs = 5*12272 batches = 61360 batches
                #   1 batch = 1 update step, if no gradients are accumulated
                # We accumulate gradients, thus: accumulate_grad_batches == 1 update step
                #   5 epochs --> (5*12272 / accumulate_grad_batches) update steps
                (0, 0.0, 16, 100),
                (0, 0.0, 16, 1000),
                (0, 0.0, 16, 5000),
                (2, 0.0, 16, 100),
                (2, 0.0, 16, 1000),
                (2, 0.0, 16, 5000),
            ])
        ]
    },
    "sbatch_06": {
        "debug": False,
        "commands": [
            f"python -m src.main"
            f" --experiment_name bertfornli-exp1"
            f" --experiment_version"
            f" 'S6.{i + 1:02}_gamma={gamma:.1f}_n-hans={n_hans}_wdecay={weight_decay}_bs=32_accum={accu}'"
            f" --gpus -1"
            f" --focal_loss_gamma {gamma}"
            f" --accumulate_grad_batches {accu}"
            f" --lr 1e-3"
            f" --batch_size 32"
            f" --warmup_steps {(5 * 12272) // accu}"
            f" --n_epochs 15"
            f" --early_stopping_patience 10"
            f" --weight_decay {weight_decay}"
            f" --gradient_clip 0"
            f" --adam_epsilon 1e-8"
            f" --precision 16"
            f" --num_hans_train_examples {n_hans}"
            for i, (gamma, weight_decay, accu, n_hans) in enumerate([
                # Warmup of 5 epochs:
                #   392702 train samples
                #   ceil(392702/32) = 12272 batches
                #   5 epochs = 5*12272 batches = 61360 batches
                #   1 batch = 1 update step, if no gradients are accumulated
                # We accumulate gradients, thus: accumulate_grad_batches == 1 update step
                #   5 epochs --> (5*12272 / accumulate_grad_batches) update steps
                (0, 0.0, 16, 0),
                (2, 0.0, 16, 0),
                (10, 0.0, 16, 0),
            ])
        ]
    },
    "sbatch_07": {
        "debug": False,
        "commands": [
            "python -m src.main"
            " --experiment_name bertfornli-exp1"
            " --experiment_version"
            " 'S7.{i:02d}_gamma={gamma:.1f}_{optimizer_name}'"
            " --optimizer_name {optimizer_name}"
            " --scheduler_name {scheduler_name}"
            " --gpus -1"
            " --adam_epsilon {adam_epsilon}"
            " --weight_decay {weight_decay}"
            " --warmup_ratio {warmup_ratio}"
            " --gradient_clip_val {grad_clip}"
            " --tokenizer_model_max_length {model_max_length}"
            " --focal_loss_gamma {gamma}"
            " --accumulate_grad_batches {accu}"
            " --lr {lr}"
            " --batch_size 32"
            " --n_epochs 3"
            " --early_stopping_patience 10"
            " --precision 16"
            " --num_hans_train_examples {n_hans}".format(
                optimizer_name=optimizer_name,
                i=i + 1,
                scheduler_name="polynomial",
                lr=2e-5,
                weight_decay=0.01,
                warmup_ratio=0.1,
                accu=1,
                gamma=0.0,
                grad_clip=1.0,
                n_hans=0,
                model_max_length=128,
                adam_epsilon=1e-6,
            )
            for i, optimizer_name in enumerate(["adamw", "adam"])
        ]
    },
}

SBATCH_ID = 'sbatch_07'
OUTPUT_FOLDER = f"./sbatch/{SBATCH_ID}"

sbatch_config = sbatch_configurations[SBATCH_ID]
if __name__ == '__main__':
    dirname = pathlib.Path(OUTPUT_FOLDER)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

    for i, cmd in enumerate(sbatch_config["commands"]):
        i += 1  # start from 1
        script_path = os.path.join(OUTPUT_FOLDER, f"nli-{SBATCH_ID.split('_')[-1]}-{i:02d}.sh")
        with open(script_path, "w") as f:
            f.write(fill_template(i=i, sbatch_id=SBATCH_ID, command=cmd, debug=sbatch_config["debug"]))
        print(f"Created script: {script_path}")
    print("Done")
