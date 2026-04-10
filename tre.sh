#!/bin/bash
#SBATCH --job-name=hello_tre_gpu
#SBATCH --output=/job_scratch/%x.out
#SBATCH --error=/job_scratch/%x.error.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

module load cuda

nvidia-smi
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml



# normal 
# sbatch -p gpu hellogpu.sh


# test gpu
# srun -p interruptible_gpu --gres gpu --constraint a40 --pty /bin/bash -l
