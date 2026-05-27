#!/bin/bash -l
#SBATCH --job-name=llm-deepspeed
#SBATCH --nodes=2               # number of GPU nodes
#SBATCH --ntasks-per-node=1     # one task per node when using the deepspeed launcher
#SBATCH --gpus-per-node=8       # allocate all GPUs on each node
#SBATCH --cpus-per-task=32       # CPU cores per task
#SBATCH --time=00:30:00         # wall-clock limit
#SBATCH --account=<PL-Grid grant>-gpu-a100  # replace with your PL‑Grid grant
#SBATCH --partition=plgrid-gpu-a100    # partition on Athena with A100 GPUs
#SBATCH --output=ds_%j.out     # stdout file
#SBATCH --error=ds_%j.err      # stderr file

# Determine master node and IP
set -euo pipefail
echo "Job $SLURM_JOB_ID starting on nodes: $SLURM_JOB_NODELIST"

nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29501

echo "Master node is $head_node with IP $MASTER_ADDR:$MASTER_PORT"

#DeepSpeed używa kompilatora Triton, który zapisuje tam małe pliki tymczasowe. Dyski NFS są do tego trochę za wolne. Dlatego przekierowujemy go na scratcha
export TRITON_CACHE_DIR="$SCRATCH/.triton"
# Load modules and activate environment
export HF_HOME="$SCRATCH/.cache/huggingface"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"

module load CUDA/12.1.1
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate llm_env

# Optymalizacja użycia procesora (32 rdzenie / 8 kart = 4)
export OMP_NUM_THREADS=4

# WYMUSZENIE POPRAWNYCH KART SIECIOWYCH DO KOMUNIKACJI (Omija błędy InfiniBand/Localhost)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond,eth,ib,enp
export GLOO_SOCKET_IFNAME=bond,eth,ib,enp
export NCCL_IB_DISABLE=0
export FI_PROVIDER=mlx

export WANDB_DISABLED=true


# Number of GPUs per node as reported by SLURM; this should match the
# ``--gpus-per-node`` directive above.
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE

# Launch DeepSpeed across all nodes.  The ``deepspeed`` launcher will
# create the required workers on each GPU.  The JSON config file
# passed via ``--deepspeed_config`` defines ZeRO‑3 settings such as
# offloading and bucket sizes【796659090806970†L108-L116】.

#srun deepspeed \
#  --num_gpus $GPUS_PER_NODE \
#  --num_nodes $SLURM_NNODES \
#  --master_addr $MASTER_ADDR \
#  --master_port $MASTER_PORT \
#  train_deepspeed.py \
#  --model_name bigscience/bloom-3b \
#  --dataset_name wikitext \
#  --dataset_config wikitext-2-raw-v1 \
#  --epochs 1 \
#  --batch_size 2 \
#  --block_size 512 \
#  --deepspeed_config ds_config_zero3.json \
#  --fp16 \
#  --output_dir ./outputs/deepspeed

#nie działa, bo potrzeba pliku tekstowego hostfile z listą adresów IP, która w SLURM-ie nie jest tworzona

# Launch using torchrun instead of deepspeed launcher!
# Hugging Face Trainer will automatically pick up the DeepSpeed config.

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
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
  --deepspeed_config ds_config_zero3.json \
  --fp16 \
  --output_dir ./outputs/deepspeed