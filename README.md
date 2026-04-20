# Distributed LLM Training on Athena with FSDP and DeepSpeed

This repository contains example code and job scripts for fine‑tuning large
language models (LLMs) on the [Athena](https://plgrid.pl) cluster using
PyTorch’s **Fully Sharded Data Parallel** (FSDP) and **DeepSpeed ZeRO‑3**
distributed training strategies.  The goal is to demonstrate how to
efficiently fine‑tune models in the **3–7 billion parameter range** across
multiple GPU nodes on the PL‑Grid infrastructure.

## Contents

* `train_fsdp.py` – A PyTorch training script that wraps a Hugging Face
  language model with FSDP to shard its parameters, gradients and
  optimiser state across all GPUs.  It initialises the distributed
  process group, tokenises a text dataset and performs a simple
  fine‑tuning loop.  Model checkpoints are saved only from rank 0.

* `train_deepspeed.py` – A training script that uses Hugging Face’s
  `Trainer` together with a DeepSpeed configuration file to enable
  ZeRO‑3 partitioning.  The script prepares a causal language
  modelling dataset, constructs a `Trainer` and passes the
  `deepspeed_config` to `TrainingArguments`.

* `fsdp_config.json` – Example FSDP configuration.  The main script
  in this repo does not load this JSON directly; it is provided as a
  reference for fine‑tuning using the Hugging Face `Trainer` API.

* `ds_config_zero3.json` – DeepSpeed configuration enabling ZeRO
  Stage 3 with CPU offload and typical bucket sizes.  This file is
  consumed by `train_deepspeed.py`.

* `sbatch_fsdp.sh` – A SLURM batch script that runs `train_fsdp.py`
  across multiple nodes using `torchrun`.  The script determines the
  master node’s IP address, loads CUDA and conda modules and
  launches the training across all GPUs.  Adapt the `#SBATCH`
  directives to match your grant ID and desired number of nodes.

* `sbatch_deepspeed.sh` – A SLURM batch script that runs
  `train_deepspeed.py` using the `deepspeed` launcher.  This script
  requests one task per node and uses all GPUs on each node.  Like
  the FSDP script, it discovers the master node IP, activates your
  conda environment and executes the job.

* `requirements.txt` – Minimal Python dependencies.  Install these
  into your own environment (for example via `pip install -r
  requirements.txt`) on a login node.

## Quick start

1. **Create a conda environment and install dependencies** (see
   `report.md` for exact commands).  You will need PyTorch with
   CUDA support, the Hugging Face `transformers` and `datasets`
   libraries, the `accelerate` library and `deepspeed`.

2. **Clone this repository** into your PL‑Grid home directory or
   scratch space.

3. **Edit the `#SBATCH` directives** in either `sbatch_fsdp.sh` or
   `sbatch_deepspeed.sh` to specify your grant ID, desired number of
   nodes and wall‑clock time.

4. **Submit the job** from the command line:

   ```bash
   sbatch sbatch_fsdp.sh      # for FSDP training
   sbatch sbatch_deepspeed.sh # for DeepSpeed ZeRO‑3 training
   ```

   The scripts use `srun` to launch either `torchrun` or `deepspeed`
   across all allocated nodes.  The rendezvous address is computed
   automatically from the first host in the SLURM node list.  During
   execution you can monitor progress with `squeue -u $USER`.

5. **Retrieve your outputs**.  Results are saved under the
   `outputs/` directory in the working directory you launched from.
   Standard output and error logs are written to files with the job
   number in their names.

## Notes

* FSDP sharding strategy is configured directly in the code via
  calls to `fully_shard(model)` and does not require an external
  configuration file.  The provided `fsdp_config.json` can be used
  with Hugging Face’s `Trainer` API if you prefer that high‑level
  interface.

* DeepSpeed requires passwordless SSH between nodes when using the
  default launcher【796659090806970†L47-L60】.  On PL‑Grid this is
  configured automatically when launching through `srun`; you do not
  need to set up a hostfile manually.

* The default dataset used in both scripts is **Wikitext‑2**; you can
  substitute any Hugging Face dataset or provide your own text files
  by modifying the arguments in the SLURM script.

* For models larger than **7 B parameters**, consider using
  activation checkpointing or CPU/NVMe offload (not enabled in these
  examples) to further reduce memory usage.