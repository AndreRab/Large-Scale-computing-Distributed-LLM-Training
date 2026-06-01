# Distributed LLM Training on Athena with FSDP and DeepSpeed

This repository contains example code and Slurm job scripts for fine-tuning a
large language model on the Athena PL-Grid cluster with two distributed
training strategies:

* PyTorch Fully Sharded Data Parallel (FSDP)
* DeepSpeed ZeRO-3 through Hugging Face `Trainer`

Both examples train `bigscience/bloom-3b` on the Wikitext-2 raw training split
for causal language modeling. The scripts are configured for two GPU nodes with
two GPUs per node, using `srun torchrun` for distributed launch.

## Contents

* `train_fsdp.py` - Native PyTorch FSDP training script. It initializes the
  NCCL process group, tokenizes Wikitext, wraps BLOOM blocks with
  `FullyShardedDataParallel`, optionally uses BF16 mixed precision, runs a
  manual training loop, logs loss/runtime/throughput, and saves a sharded
  checkpoint.

* `train_deepspeed.py` - Hugging Face `Trainer` training script using a
  DeepSpeed config passed through `TrainingArguments`. It prepares the same
  causal language modeling dataset and lets DeepSpeed handle ZeRO-3
  partitioning.

* `ds_config_zero3.json` - DeepSpeed ZeRO-3 configuration. The current config
  uses automatic batch-size values, BF16 mixed precision, gradient clipping,
  reduce-scatter, overlap communication, and bucket-size settings. CPU/NVMe
  offload is not enabled.

* `fsdp_config.json` - Reference FSDP configuration. The native FSDP script
  does not load this file directly; equivalent behavior is configured in
  Python inside `train_fsdp.py`.

* `sbatch_fsdp.sh` - Slurm script for launching `train_fsdp.py` with
  `srun torchrun`.

* `sbatch_deepspeed.sh` - Slurm script for launching `train_deepspeed.py` with
  `srun torchrun` and `ds_config_zero3.json`.

* `report.md` - Project report with method descriptions and measured results
  from the logs.

* `logs/` - Example Slurm output/error logs from completed runs.

## Quick Start

1. Install the Python dependencies from `requirements.txt` in your cluster
   environment.

2. Edit the `#SBATCH` settings in `sbatch_fsdp.sh` or `sbatch_deepspeed.sh`,
   especially the account, partition, node count, GPU count, and wall time.

3. Submit one of the jobs:

   ```bash
   sbatch sbatch_fsdp.sh
   sbatch sbatch_deepspeed.sh
   ```

4. During execution, monitor the queue and logs:

   ```bash
   squeue -u $USER
   tail -f fsdp_<job_id>.out
   tail -f ds_trainer_<job_id>.out
   ```

5. Retrieve model outputs from scratch:

   ```text
   $SCRATCH/outputs/fsdp
   $SCRATCH/outputs/ds
   ```

## Current Run Configuration

| Item | Value |
| --- | --- |
| Model | `bigscience/bloom-3b` |
| Dataset | `wikitext`, `wikitext-2-raw-v1` |
| Split | `train` |
| Task | Causal language modeling |
| Epochs | `1` |
| Block size | `512` |
| Batch size | `2` per GPU |
| Precision | BF16 |
| Nodes | `2` |
| GPUs per node | `2` |
| Launcher | `srun torchrun` |

## Implementation Notes

FSDP sharding is configured directly in `train_fsdp.py` with
`FullyShardedDataParallel`, `transformer_auto_wrap_policy` for `BloomBlock`,
`use_orig_params=True`, and optional BF16 `MixedPrecision`. The script saves a
sharded state dict through `torch.distributed.checkpoint`.

DeepSpeed is enabled by passing `ds_config_zero3.json` to Hugging Face
`TrainingArguments`. The Python training loop stays close to normal `Trainer`
code while ZeRO-3 partitions training state across distributed workers.

Both scripts use the same dataset preprocessing pattern: tokenize text,
concatenate tokens into fixed-length blocks, then train causal language modeling
with shifted `input_ids` and `labels`.

For models larger than this example, memory can be reduced further with
activation checkpointing or CPU/NVMe offload, but those options are not enabled
in the current scripts.
