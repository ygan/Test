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





#!/bin/bash
#SBATCH --job-name=probe_tre_paths
#SBATCH --output=/job_scratch/%x.out
#SBATCH --error=/job_scratch/%x.err
#SBATCH --partition=cpu
#SBATCH --time=00:05:00

set -e

echo "host=$(hostname)"
id

echo "--- mount probes ---"
findmnt /project || true
findmnt /storage || true
df -h /project || true
df -h /storage || true

echo "--- path probes ---"
for p in \
  /project \
  /project/LLMs \
  /project/LLMs/Qwen2.5-0.5B-Instruct \
  /storage/project \
  /storage/project/LLMs \
  /storage/project/LLMs/Qwen2.5-0.5B-Instruct
do
  echo "== $p =="
  ls -ld "$p" || true
done
