#!/bin/bash -l
#SBATCH --job-name=ne_param
#SBATCH --output=/scratch/users/%u/ne_results/param_var/%A_%a.out
#SBATCH --partition=gpu
#SBATCH --array=0-479
#SBATCH --gres=gpu

anaconda_module="anaconda3/2022.10-gcc-10.3.0"
conda_prefix="/scratch/users/${USER}/conda_envs/jax"
script_path="/users/${USER}/optevo/main.py"
out_dir="/scratch/users/${USER}/ne_results/param_var/"

task_id=$SLURM_ARRAY_TASK_ID

echo "SLURM_ARRAY_TASK_ID: $task_id"

# Load modules
# module load $anaconda_module
# source /users/${USER}/.bashrc
# source activate $conda_prefix

$seed_values=(1 2 3 4 5) # 5
$noise_p_values=(0.001 0.002 0.005 0.01) # 4
$tournament_size_values=(5 10 20) # 3
$batch_size_values=(128 256) # 2
$elites_values=(1 2 5 10) # 4

$seed=${seed_values[$((task_id % 5))]}
$noise_p=${noise_p_values[$(((task_id / 5) % 4))]}
$tournament_size=${tournament_size_values[$(((task_id / 20) % 3))]}
$batch_size=${batch_size_values[$(((task_id / 60) % 2))]}
$elites=${elites_values[$(((task_id / 120) % 4))]}

echo "seed: $seed"
echo "noise_p: $noise_p"
echo "tournament_size: $tournament_size"
echo "batch_size: $batch_size"
echo "elites: $elites"


Run the script
python $script_path \
  --seed=$seed \
  --out_dir=$HOME/optevo/results \
  --data_dir=$HOME/optevo/data \
  --arch="ff" \
  --max_gen=5000 \
  --pop_size=50 \
  --elites=$elites \
  --crossover_p=0.5 \
  --tournament_size=$tournament_size \
  --noise_p=$noise_p \
  --noise_sigma=0.01 \
  --criterion="batch_acc" \
  --batch_size=$batch_size 

