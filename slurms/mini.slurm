#!/bin/bash
#SBATCH --job-name=lerobot_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00           # Max walltime
#SBATCH --partition=gpu           # GPU partition (check with 'sinfo')
#SBATCH --gres=gpu:a100:1         # GPU type (e.g., a100, v100, or just 'gpu:1')
#SBATCH --cpus-per-task=8         # Reasonable for data loading
#SBATCH --mem=64G                 # More memory for datasets
#SBATCH --mail-type=ALL           # Get email on start/end/fail (if configured)
#SBATCH --mail-user=olin.goudey@tufts.edu

# Activate your conda env
source ~/miniconda3/bin/activate lerobot

# Move to your project dir
cd ~/lerobot

# Start training

python3 src/lerobot/scripts/train.py  --policy.path=lerobot/smolvla_base  --dataset.repo_id=$(realpath data/stationary_env)   --batch_size=64   --steps=20000   --policy.push_to_hub=false --save_freq 1000 --wandb.enable=true
