"""
Script used to configure experiments and generate Sbatch files that can be run with SLURM.
"""

import os
import pathlib
import random

random.seed(72)

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

PRODUCTION_HEADER_1_GPU = """#SBATCH --chdir /scratch/izar/rajic/nli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=90G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
"""

PRODUCTION_HEADER_2_GPUS = """#SBATCH --chdir /scratch/izar/rajic/nli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
"""

PRODUCTION_HEADER_2_GPUS_W_RAM = """#SBATCH --chdir /scratch/izar/rajic/nli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=370G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
"""


def fill_template(command, header):
    return f"""#!/bin/bash
{header}
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
printf "Run configured and environment setup. Gonna run now.\\n\\n"
{command}
echo FINISHED at $(date)

"""


sbatch_configurations = {
    2: {
        "runs": [
            {
                "command": f"python -m src.main \\\n"
                           f"  --wandb_entity epfl-optml \\\n"
                           f"  --experiment_name nli \\\n"
                           f"  --experiment_version \\\n"
                           f"  '{{run_id}}_model-{model}_dataset-{dataset}_gamma-{gamma:.1f}_seed-{seed}' \\\n"
                           f"  --dataset {dataset} \\\n"
                           f"  --seed {seed} \\\n"
                           f"  --optimizer_name {optimizer_name} \\\n"
                           f"  --scheduler_name {scheduler_name} \\\n"
                           f"  --adam_epsilon {adam_epsilon} \\\n"
                           f"  --weight_decay {weight_decay} \\\n"
                           f"  --warmup_ratio {warmup_ratio} \\\n"
                           f"  --gradient_clip_val {grad_clip} \\\n"
                           f"  --tokenizer_model_max_length {model_max_length} \\\n"
                           f"  --focal_loss_gamma {gamma} \\\n"
                           f"  --accumulate_grad_batches {accu} \\\n"
                           f"  --lr {lr} \\\n"
                           f"  --batch_size {batch_size} \\\n"
                           f"  --n_epochs {n_epochs} \\\n"
                           f"  --early_stopping_patience 30 \\\n"
                           f"  --precision {precision} \\\n"
                           f"  --num_hans_train_examples {n_hans} \\\n",
                "header": PRODUCTION_HEADER_1_GPU,
                "run_id": None,
                "run_name": f"{seed}.{{run_id}}_{model[:1].upper()}{dataset[:1].upper()}_gamma-{gamma:.1f}_seed-{seed}",
            }
            # Varying
            for model in ["bert"]
            for dataset in ["mnli", "snli"]
            for gamma in [0, 0.5, 1.0, 2.0, 5.0, 10.0]
            for seed in [72, 36, 180, 360, 54]
            # Fixed
            for optimizer_name in ["adamw"]
            for scheduler_name in ["polynomial"]
            for warmup_ratio in [0.1]
            for batch_size in [32]
            for grad_clip in [1.0]
            for model_max_length in [128]
            for n_epochs in [10]
            for precision in [16]
            for adam_epsilon in [1e-6]
            for accu in [1]
            for lr in [2e-5]
            for weight_decay in [0.01]
            for n_hans in [0]
        ]
    },
}
for sbatch_id, sbatch_config in sbatch_configurations.items():
    for i, run in enumerate(sbatch_config["runs"]):
        run_id = f"S{sbatch_id:01d}.{i + 1:02d}"
        run["command"] = run["command"].format(run_id=run_id)
        run["run_name"] = run["run_name"].format(run_id=run_id)
        run["run_id"] = run_id

OUTPUT_FOLDER = f"./sbatch/sbatch_{{sbatch_id:02d}}"

if __name__ == '__main__':
    for sbatch_id in sbatch_configurations.keys():
        print(f"sbatch_id={sbatch_id}")
        dirname = pathlib.Path(OUTPUT_FOLDER.format(sbatch_id=sbatch_id))
        print(f"dirname={dirname}")
        if not dirname.is_dir():
            dirname.mkdir(parents=True, exist_ok=False)

        sbatch_config = sbatch_configurations[sbatch_id]
        for run in sbatch_config["runs"]:
            script_path = os.path.join(dirname, f'{run["run_name"]}.sh')
            with open(script_path, "w") as f:
                f.write(fill_template(command=run["command"], header=run["header"]))
            print(f"Created script: {script_path}")
        print(f"Done with sbatch_id={sbatch_id}.")
    print("Done.")
