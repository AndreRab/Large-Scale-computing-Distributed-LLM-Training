# Commands to run the distributed LLM training

The following commands guide you through preparing the environment on the
Athena cluster, cloning this repository and submitting the example
training jobs.  Each command is wrapped in a ``command`` block
followed by a placeholder ``[result]`` where you can paste the actual
output after execution.

## 1. Prepare the environment

Load the necessary modules, create a conda environment and install
dependencies.  Adjust module names to match the software versions
available on Athena.

```command
module load cuda/12.0
module load miniconda3
conda create -n athena_llm_env python=3.10 -y
conda activate athena_llm_env
pip install torch --extra-index-url https://download.pytorch.org/whl/cu120
pip install -r requirements.txt
```
[result]

## 2. Clone the project repository

Assuming you have pushed this project to your own GitHub account,
replace ``<URL_of_your_GitHub_repo>`` with the actual URL.  Move into
the project directory after cloning.

```command
git clone <URL_of_your_GitHub_repo>
cd distributed_llm_training_athena
```
[result]

## 3. Submit an FSDP training job

Submit the SLURM script that launches `train_fsdp.py` with
PyTorch’s Fully Sharded Data Parallel.  Edit `sbatch_fsdp.sh` to set
your PL‑Grid grant ID and desired resources before running.

```command
sbatch sbatch_fsdp.sh
```
[result]

Check the status of your job (replace ``$USER`` with your username):

```command
squeue -u $USER
```
[result]

## 4. Submit a DeepSpeed ZeRO‑3 training job

Submit the script that runs `train_deepspeed.py` with the ZeRO‑3
optimizer.  Again, ensure the `#SBATCH` directives in
`sbatch_deepspeed.sh` match your account and resource requirements.

```command
sbatch sbatch_deepspeed.sh
```
[result]

Monitor the job queue as above.  Once the jobs complete, the output
directories (`outputs/fsdp` and `outputs/deepspeed`) will contain the
fine‑tuned models and logs, and the `fsdp_<jobid>.out`,
`fsdp_<jobid>.err`, `ds_<jobid>.out` and `ds_<jobid>.err` files will
contain the stdout and stderr of each run.