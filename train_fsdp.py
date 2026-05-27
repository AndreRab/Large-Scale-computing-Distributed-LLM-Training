r"""
Fine‑tuning script for large language models using PyTorch's Fully Sharded
Data Parallel (FSDP) API.  This example is designed to run on the Athena
cluster of the Polish national PL‑Grid infrastructure.  Models in the
3 – 7 billion parameter range (for example, Falcon‑7B or similar Hugging Face
models) are too large to fit on a single A100 GPU, so FSDP shards the
model's parameters, gradients and optimiser state across multiple GPUs and
multiple nodes【436250730084965†L279-L320】.  When combined with the SLURM
scheduler, you can use this script to perform distributed fine‑tuning on
several GPU nodes simultaneously.

The training loop is intentionally simple: it tokenises a text dataset
for causal language modelling, partitions it across the distributed
processes and performs a standard forward/backward/optimizer step on
each mini‑batch.  Checkpoints are saved only from the rank‑0 process
to avoid file contention.

Usage within a SLURM batch script (see ``sbatch_fsdp.sh`` in this
repository for a full example):

    srun --nodes=$SLURM_NNODES --ntasks‑per‑node=$SLURM_GPUS_PER_NODE \ 
         torchrun \
           --nnodes $SLURM_NNODES \
           --nproc_per_node $SLURM_GPUS_PER_NODE \
           --rdzv_id $SLURM_JOB_ID \
           --rdzv_backend c10d \
           --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
           train_fsdp.py --model_name bigscience/bloom-3b --dataset_name wikitext \
           --dataset_config wikitext-2-raw-v1 --epochs 1

The script does not depend on any PL‑Grid specifics; the scheduler
allocates resources and ``torchrun`` handles the rendezvous across nodes.
"""

import argparse
import os
from functools import partial
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Dodane importy do prawidłowego zapisu wag w FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy



def tokenize_function(examples, tokenizer, block_size: int):
    """Tokenise and concatenate a batch of texts for causal language modelling.

    The function concatenates all texts in the batch, tokenises them and
    splits the resulting token stream into fixed‑length blocks.  Any
    remainder that does not fill a complete block is dropped.  This
    approach avoids padding and ensures that each sequence is the same
    length across all distributed processes.
    """
    concatenated = tokenizer(examples["text"], return_attention_mask=False, truncation=False)
    input_ids = []
    for ids in concatenated["input_ids"]:
        input_ids.extend(ids)
    total_length = (len(input_ids) // block_size) * block_size
    input_ids = input_ids[:total_length]
    result = {"input_ids": [input_ids[i : i + block_size] for i in range(0, len(input_ids), block_size)]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine‑tune a Hugging Face model using FSDP")
    parser.add_argument("--model_name", type=str, required=True, help="Model identifier on the Hugging Face hub")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name on the Hugging Face hub")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration (if any)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Local batch size per GPU")
    parser.add_argument("--block_size", type=int, default=512, help="Sequence length for language modelling")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./fsdp_output", help="Directory to save the fine‑tuned model")
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many optimizer steps; useful for cluster smoke tests")
    args = parser.parse_args()

    # Initialise the distributed process group.  ``torchrun`` sets
    # environment variables such as RANK, WORLD_SIZE, LOCAL_RANK,
    # MASTER_ADDR and MASTER_PORT which are needed for rendezvous.  NCCL
    # is the recommended backend for multi‑GPU training on CUDA devices.
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Download tokenizer and model.  The first process on each node
    # downloads the model weights; subsequent processes reuse cached
    # files.  When ``fp16`` is enabled we convert weights to FP16 to
    # reduce the memory footprint.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if args.fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)

    # Use one FSDP API consistently. The previous composable FSDP wrapping
    # crashed during classic FSDP full-state-dict checkpoint saving.
    # and optimiser state across all ranks【436250730084965†L279-L320】.  The result is an
    auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=100_000_000)
    mixed_precision = None
    if args.fp16:
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        mixed_precision=mixed_precision,
        use_orig_params=True,
    )

    # Load and preprocess the dataset.  We use Wikitext by default but any
    # text dataset on the hub can be specified.  The map call will
    # execute on each worker independently (we set ``num_proc=1`` to
    # avoid nested multiprocessing).
    dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.block_size),
        batched=True,
        remove_columns=dataset.column_names,
    )
    sequences = [torch.tensor(seq, dtype=torch.long) for seq in tokenized_dataset["input_ids"]]

    class CausalDataset(torch.utils.data.Dataset):
        """Simple dataset wrapper that returns input/label pairs for causal LM."""

        def __init__(self, seqs):
            self.seqs = seqs

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, idx):
            seq = self.seqs[idx]
            return {"input_ids": seq[:-1], "labels": seq[1:]}

    train_dataset = CausalDataset(sequences)
    sampler = DistributedSampler(train_dataset)
    dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, drop_last=True)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            global_step += 1
            if step % 10 == 0 and torch.distributed.get_rank() == 0:
                print(f"Epoch {epoch+1}, step {step}, loss={loss.item():.4f}")
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Save only on rank 0 to avoid race conditions.  FSDP returns
    # sharded state dicts by default; gather them to CPU before saving.


    # Używamy context managera, który musi być wykonany przez WSZYSTKIE procesy.
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        # Ta linia wymaga komunikacji między węzłami, więc wywołujemy ją przed if'em
        state_dict = model.state_dict()

    if torch.distributed.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        #state_dict = model.state_dict(gather_dtensor=True)  # type: ignore[call-arg]
        torch.save(state_dict, os.path.join(args.output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(args.output_dir)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
