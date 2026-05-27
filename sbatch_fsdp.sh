#!/bin/bash -l
#SBATCH --job-name=llm-fsdp
#SBATCH --nodes=2                # number of GPU nodes to allocate
#SBATCH --ntasks-per-node=1      # ONE task per node (to run torchrun)
#SBATCH --gpus-per-node=8        # ALL 8 GPUs on the node for this one task
#SBATCH --cpus-per-task=32        # number of CPU cores per task (adjust as needed)
#SBATCH --time=00:30:00          # wall-clock time limit
#SBATCH --account=<PL-Grid grant>-gpu-a100  # replace with your PL‑Grid grant
#SBATCH --partition=plgrid-gpu-a100    # partition with A100 GPUs
#SBATCH --output=fsdp_%j.out     # standard output file
#SBATCH --error=fsdp_%j.err      # standard error file

# The following script prepares the environment, determines the master
# node's IP address and launches ``torchrun`` across all nodes.  It
# assumes that each node provides 8 GPUs and that your conda
# environment ``llm_env`` has been created ahead of time.

set -euo pipefail

echo "Job $SLURM_JOB_ID starting on nodes: $SLURM_JOB_NODELIST"

# Determine the head node and its IP address for rendezvous.  We
# capture only the first IP address returned by hostname --ip-address.
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500

echo "Master node is $head_node with IP $MASTER_ADDR"

# Load modules and activate your conda environment.  Adjust these
# module names to match Athena's software stack.  The CUDA module
# ensures that the correct NCCL libraries are available.
export HF_HOME="$SCRATCH/.cache/huggingface"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"

module load CUDA/12.1.1
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate llm_env

# Optional: set number of threads for CPU operations
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK/2:-4}
export OMP_NUM_THREADS=4 # najleoije podzielić SLURM_CPUS_PER_TASK/ilosc kart graficznych 

# WYMUSZENIE POPRAWNYCH KART SIECIOWYCH DO KOMUNIKACJI (Omija błędy InfiniBand/Localhost)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond,eth,ib,enp
export GLOO_SOCKET_IFNAME=bond,eth,ib,enp
export NCCL_IB_DISABLE=0
export FI_PROVIDER=mlx


export WANDB_DISABLED=true
# Launch the training across nodes.  torchrun will read RANK and
# WORLD_SIZE from the environment variables set by SLURM.  We pass
# rendezvous information explicitly via ``rdzv_id``, ``rdzv_backend``
# and ``rdzv_endpoint``.  Adjust model and dataset names as needed.

srun torchrun \
  --nnodes $SLURM_NNODES \
  --nproc_per_node ${SLURM_GPUS_ON_NODE:-8} \
  --rdzv_id $SLURM_JOB_ID \
  --rdzv_backend c10d \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  train_fsdp.py \
  --model_name bigscience/bloom-3b \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --epochs 1 \
  --batch_size 2 \
  --block_size 512 \
  --fp16 \
  --lr 5e-5 \
  --output_dir ./outputs/fsdp
