#!/bin/bash -l
#SBATCH --job-name=llm-diag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1        # Jedna karta wystarczy do błyskawicznego testu
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output=diag_%j.out
#SBATCH --error=diag_%j.err

echo "=== DIAGNOSTYKA START: $(date) ==="

# 1. Zabezpieczenie limitów dyskowych (Quota)
export HF_HOME="$SCRATCH/.cache/huggingface"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"

# 2. Ładowanie modułów i środowiska
echo "--- Ładowanie modułów i środowiska ---"
module load CUDA/12.1.1
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate llm_env

echo "Python: $(which python)"

# 3. Test GPU i nowej wersji PyTorcha
echo "--- Test PyTorch i CUDA ---"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'GPU Count: {torch.cuda.device_count()}')
import transformers
print('Transformers OK')
"

# 4. Próba uruchomienia torchrun na 1 karcie
echo "--- Próba uruchomienia torchrun (mini model) ---"
export MASTER_ADDR=127.0.0.1  # Do testu na 1 węźle wystarczy localhost
export MASTER_PORT=29509

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train_fsdp.py \
  --model_name bigscience/bloom-560m \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --epochs 1 \
  --batch_size 1 \
  --max_steps 5 \
  --output_dir ./test_out

echo "=== DIAGNOSTYKA KONIEC: $(date) ==="
