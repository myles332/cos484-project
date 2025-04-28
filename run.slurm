#!/bin/bash
#SBATCH --job-name=train_ppo_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=na1225@princeton.edu

module purge
module load anaconda3/2024.10
conda activate cos484

python attack.py \
    --model Llama-2-7b-chat-hf \
    --use_default \
    --use_system_prompt