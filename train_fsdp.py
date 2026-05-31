"""
Fine-tuning script for large language models using PyTorch's Fully Sharded
Data Parallel (FSDP) API.  This example is designed to run on the Athena
cluster of the Polish national PL-Grid infrastructure.  Models in the
3-7 billion parameter range (for example, Falcon-7B or similar Hugging Face
models) are too large to fit on a single A100 GPU, so FSDP shards the
model's parameters, gradients and optimiser state across multiple GPUs and
multiple nodes.  When combined with the SLURM
scheduler, you can use this script to perform distributed fine-tuning on
several GPU nodes simultaneously.

The training loop is intentionally simple: it tokenises a text dataset
for causal language modelling, partitions it across the distributed
processes and performs a standard forward/backward/optimizer step on
each mini-batch.  Checkpoints are saved only from the rank-0 process
to avoid file contention.

Usage within a SLURM batch script (see ``sbatch_fsdp.sh`` in this
repository for a full example):

    srun torchrun \
      --nnodes $SLURM_NNODES \
      --nproc_per_node $SLURM_GPUS_PER_NODE \
      --rdzv_id $SLURM_JOB_ID \
      --rdzv_backend c10d \
      --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
      train_fsdp.py --model_name bigscience/bloom-3b --dataset_name wikitext \
      --dataset_config wikitext-2-raw-v1 --epochs 1 --bf16

The script does not depend on any PL-Grid specifics; the scheduler
allocates resources, and ``torchrun`` handles the rendezvous across nodes.
"""

import argparse
import os
import time
from functools import partial
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, StateDictType, ShardedStateDictConfig
import torch.distributed.checkpoint as dist_cp
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.bloom.modeling_bloom import BloomBlock

def tokenize_function(examples, tokenizer, block_size: int):
    """Tokenise and concatenate a batch of texts for causal language modelling."""
    concatenated = tokenizer(examples["text"], return_attention_mask=False, truncation=False)
    input_ids = []
    for ids in concatenated["input_ids"]:
        input_ids.extend(ids)
    total_length = (len(input_ids) // block_size) * block_size
    input_ids = input_ids[:total_length]
    result = {"input_ids": [input_ids[i : i + block_size] for i in range(0, len(input_ids), block_size)]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model using FSDP")
    parser.add_argument("--model_name", type=str, required=True, help="Model identifier on the Hugging Face hub")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name on the Hugging Face hub")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration (if any)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Local batch size per GPU")
    parser.add_argument("--block_size", type=int, default=512, help="Sequence length for language modelling")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./fsdp_output", help="Directory to save the fine-tuned model")
    args = parser.parse_args()

    # Initialise the distributed process group.  ``torchrun`` sets
    # environment variables such as RANK, WORLD_SIZE, LOCAL_RANK,
    # MASTER_ADDR and MASTER_PORT which are needed for rendezvous.  NCCL
    # is the recommended backend for multi‑GPU training on CUDA devices.
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Download tokenizer and model. Processes reuse files from the
    # Hugging Face cache when available.  
    # When ``bf16`` is enabled, we convert weights to BF16 to
    # reduce the memory footprint.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={BloomBlock},
    )
    mixed_precision = None
    if args.bf16:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        mixed_precision=mixed_precision,
        use_orig_params=True,
    )

    # Load and preprocess the dataset.  We use Wikitext by default but any
    # text dataset on the hub can be specified.
    dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.block_size),
        batched=True,
        remove_columns=dataset.column_names,
    )
    sequences = [torch.tensor(seq, dtype=torch.long) for seq in tokenized_dataset["input_ids"]]

    class CausalDataset(torch.utils.data.Dataset):
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
    
    running_loss = 0.0

    if torch.distributed.get_rank() == 0:
        start_time = time.time()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            optimiser.zero_grad()
            loss.backward()

            model.clip_grad_norm_(1.0)
            optimiser.step()
            
            global_loss = loss.detach().clone()

            torch.distributed.all_reduce(global_loss, op=torch.distributed.ReduceOp.AVG)

            if torch.distributed.get_rank() == 0:
                running_loss += global_loss.item()

            if step % 10 == 0 and torch.distributed.get_rank() == 0:
                avg_loss = running_loss / 10 if step > 0 else running_loss
                print(f"Epoch {epoch+1}, step {step}, loss={avg_loss:.4f}")
                running_loss = 0.0

    if torch.distributed.get_rank() == 0:
        end_time = time.time()
        training_time = end_time - start_time
        
        num_steps = len(dataloader) * args.epochs

        world_size = torch.distributed.get_world_size()
        total_samples = num_steps * args.batch_size * world_size
        samples_per_second = total_samples / training_time
        
        print("\n" + "="*40)
        print("FSDP TRAINING SUMMARY")
        print("="*40)
        print(f"Total training time:   {training_time:.1f} s")
        print(f"Training speed:        {samples_per_second:.2f} samples/s")
        print(f"Final training loss:   {global_loss.item():.4f}")
        print("="*40 + "\n")
        
        os.makedirs(args.output_dir, exist_ok=True)

    torch.distributed.barrier()

    save_policy = ShardedStateDictConfig(offload_to_cpu=False)
    FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT, save_policy)
    
    model_state_dict = model.state_dict()
    
    state_dict = {"model": model_state_dict}
    
    dist_cp.save(
        state_dict=state_dict,
        checkpoint_id=args.output_dir,
    )

    # Save only on rank 0 to avoid race conditions. 
    if torch.distributed.get_rank() == 0:
        tokenizer.save_pretrained(args.output_dir)
        print(f"=== Checkpoints SHARDED saved successfully to {args.output_dir} ===")

    torch.distributed.destroy_process_group()
if __name__ == "__main__":
    main()
