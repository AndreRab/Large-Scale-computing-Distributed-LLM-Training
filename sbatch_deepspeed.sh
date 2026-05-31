#!/bin/bash -l
#SBATCH --job-name=ds-trainer
#SBATCH --nodes=2                           # Number of GPU nodes
#SBATCH --ntasks-per-node=1                 # 1 main task per node (torchrun handles the GPU processes)
#SBATCH --gpus-per-node=2                   # Allocate number of GPUs on each node, max is 8
#SBATCH --cpus-per-task=8                   # CPU cores per task
#SBATCH --time=00:30:00                     # wall-clock limit
#SBATCH --account=<grant>-gpu-a100          # replace with your PL‑Grid grant
#SBATCH --partition=plgrid-gpu-a100         # partition on Athena with A100 GPUs
#SBATCH --output=ds_trainer_%j.out          # Standard output log file
#SBATCH --error=ds_trainer_%j.err           # Standard error log file

set -euo pipefail

# --- User disk quota protection ---
export HF_HOME="$SCRATCH/.cache/huggingface"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export TRITON_CACHE_DIR="$SCRATCH/.cache/triton" 
export TORCH_EXTENSIONS_DIR="$SCRATCH/.cache/torch_extensions"
export WANDB_DISABLED=true

module load CUDA/12.1.1
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate llm_env

# --- Determine master node and IP ---
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes[0]}
#head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" ip -4 addr show ib0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29555

# --- Network configuration ---
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

# OMP_NUM_THREADS = cpus-per-task (8) / gpus-per-node (2) = 4
export OMP_NUM_THREADS=4

rm -rf "$SCRATCH/outputs/ds_trainer"
mkdir -p "$SCRATCH/outputs/ds_trainer"

srun torchrun \
  --nnodes=2 \
  --nproc_per_node=2 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train_deepspeed.py \
  --model_name bigscience/bloom-3b \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --epochs 1 \
  --batch_size 2 \
  --block_size 512 \
  --lr 5e-5 \
  --deepspeed_config ds_config_zero3.json \
  --bf16 \
  --output_dir "$SCRATCH/outputs/ds"
