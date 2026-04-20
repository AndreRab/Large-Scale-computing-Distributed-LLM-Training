#!/bin/bash -l
#SBATCH --job-name=llm-deepspeed
#SBATCH --nodes=2               # number of GPU nodes
#SBATCH --ntasks-per-node=1     # one task per node when using the deepspeed launcher
#SBATCH --gpus-per-node=8       # allocate all GPUs on each node
#SBATCH --cpus-per-task=8       # CPU cores per task
#SBATCH --time=04:00:00         # wall-clock limit
#SBATCH --account=<grant_id>-gpu-a100  # replace with your PL‑Grid grant
#SBATCH --partition=plgrid-gpu-a100    # partition on Athena with A100 GPUs
#SBATCH --output=ds_%j.out     # stdout file
#SBATCH --error=ds_%j.err      # stderr file

# Determine master node and IP
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29501

echo "Starting DeepSpeed job on $SLURM_NNODES nodes; master at $MASTER_ADDR:$MASTER_PORT"

# Load modules and activate environment
module load cuda/12.0
module load miniconda3
conda activate athena_llm_env

# Number of GPUs per node as reported by SLURM; this should match the
# ``--gpus-per-node`` directive above.
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE

# Launch DeepSpeed across all nodes.  The ``deepspeed`` launcher will
# create the required workers on each GPU.  The JSON config file
# passed via ``--deepspeed_config`` defines ZeRO‑3 settings such as
# offloading and bucket sizes【796659090806970†L108-L116】.

srun deepspeed \
  --num_gpus $GPUS_PER_NODE \
  --num_nodes $SLURM_NNODES \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
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