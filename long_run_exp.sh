#!/bin/bash -l
#SBATCH --job-name=ne_lr
#SBATCH --output=/scratch/users/%u/ne_results/%A_%a.out
#SBATCH --partition=gpu
#SBATCH --array=0-10
#SBATCH --gres=gpu

anaconda_module="anaconda3/2022.10-gcc-10.3.0"
conda_prefix="/scratch/users/${USER}/conda_envs/jax"
script_path="/users/${USER}/optevo/main.py"
out_dir="/scratch/users/${USER}/ne_results"

task_id=$SLURM_ARRAY_TASK_ID

echo "SLURM_ARRAY_TASK_ID: $task_id"

# Load modules
module load $anaconda_module
source /users/${USER}/.bashrc
source activate $conda_prefix

# Run the script
python $script_path \
  --out_dir=$out_dir \
  --seed=$task_id \
  --max_gen=500 \
  --pop_size=100 \
  --crossover_p=0.5 \
  --tournament_size=20 \
  --elites=10 \
  --mutation_p=0.005 \
  --mutation_sigma=0.01 \
  --fitness_fn="acc" \
  --batch_size=1024 \
  --data_dir=/scratch/users/${USER}/data

