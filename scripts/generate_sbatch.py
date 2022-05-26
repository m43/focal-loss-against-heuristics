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
            " --warmup 15000"
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
            " --warmup 3750"
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
            " --warmup 15000"
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
            " --warmup 3750"
            " --n_epochs 100"
            " --early_stopping_patience 50"
            " --weight_decay 0.0"
            " --gradient_clip 0.0"
            " --adam_epsilon 1e-8"
            " --precision 16",

        ]

    },
}

SBATCH_ID = 'sbatch_01'
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
