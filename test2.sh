#!/bin/bash -l
#SBATCH --job-name=test-sieci
#SBATCH --nodes=2                # Prosimy o 2 węzły (żeby wymusić komunikację sieciową)
#SBATCH --ntasks-per-node=1      # 1 task na węzeł (dla torchrun)
#SBATCH --gpus-per-node=1        # TYLKO 1 KARTA na każdym z dwóch węzłów!
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00          # Krótki czas = szybsza alokacja w SLURM
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output=siec_%j.out
#SBATCH --error=siec_%j.err

set -euo pipefail

# 0. BEZPIECZNE ZMIENNE
NNODES=${SLURM_NNODES:-2}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-1}  # Podstawi 1 kartę
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
JOB_ID=${SLURM_JOB_ID:-$RANDOM}
NODELIST=${SLURM_JOB_NODELIST:-$HOSTNAME}

echo "Start testu sieci na węzłach: $NODELIST"

# 1. KONFIGURACJA SIECI (Rendezvous)
nodes=( $(scontrol show hostnames $NODELIST) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29505

echo "Adres głównego węzła: $MASTER_ADDR:$MASTER_PORT"

# 2. ZABEZPIECZENIE DYSKU
export HF_HOME="$SCRATCH/.cache/huggingface"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"

# 3. ŚRODOWISKO I MODUŁY
module load CUDA/12.1.1
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate llm_env

export OMP_NUM_THREADS=$CPUS_PER_TASK

# WYMUSZENIE POPRAWNYCH KART SIECIOWYCH DO KOMUNIKACJI (Omija błędy InfiniBand/Localhost)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond,eth,ib,enp
export GLOO_SOCKET_IFNAME=bond,eth,ib,enp
export NCCL_IB_DISABLE=0

# 4. URUCHOMIENIE TRENINGU NA 2 KARTACH (na 2 różnych serwerach)
srun torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --rdzv_id=$JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train_fsdp.py \
  --model_name bigscience/bloom-560m \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --epochs 1 \
  --batch_size 2 \
  --max_steps 5 \
  --output_dir ./outputs/test_sieci
