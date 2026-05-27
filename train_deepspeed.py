"""
Fine‑tuning script that leverages the Hugging Face ``Trainer`` API in
combination with DeepSpeed ZeRO‑3 for memory‑efficient distributed
training.  DeepSpeed's ZeRO‑3 stage partitions model parameters,
gradients and optimiser state across data‑parallel processes, enabling
training of models well beyond the memory limits of a single GPU
【796659090806970†L44-L60】.  This script is designed for use on the
Athena (PL‑Grid) cluster but does not depend on any cluster
specifics—it will run anywhere you have Python, PyTorch and
DeepSpeed installed.

To run on multiple GPU nodes managed by SLURM, you can either invoke
the DeepSpeed launcher directly or wrap it with ``srun``.  An
example SLURM batch script is provided in ``sbatch_deepspeed.sh``.  On
the head node, you would execute something like:

    srun --nodes=$SLURM_NNODES --ntasks‑per‑node=1 \ 
         deepspeed --num_gpus $SLURM_GPUS_PER_NODE --num_nodes $SLURM_NNODES \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \ 
         train_deepspeed.py --model_name bigscience/bloom-3b --dataset_name wikitext \
         --dataset_config wikitext-2-raw-v1 --epochs 1 --deepspeed_config ds_config_zero3.json

The script tokenises a text dataset for causal language modelling, builds a
``Trainer`` and passes a ZeRO‑3 configuration file to the ``deepspeed``
parameter of ``TrainingArguments``.  DeepSpeed and the scheduler will
take care of launching the distributed processes and establishing
communications across the cluster.【796659090806970†L108-L116】
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset


def tokenize_function(examples, tokenizer, block_size: int):
    """Concatenate and split texts into fixed‑length blocks for causal LM."""
    concatenated = tokenizer(examples["text"], return_attention_mask=False, truncation=False)
    input_ids = []
    for ids in concatenated["input_ids"]:
        input_ids.extend(ids)
    total_length = (len(input_ids) // block_size) * block_size
    input_ids = input_ids[:total_length]
    return {"input_ids": [input_ids[i : i + block_size] for i in range(0, len(input_ids), block_size)]}


class CausalDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for causal language modelling."""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {"input_ids": seq[:-1], "labels": seq[1:]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine‑tune with DeepSpeed ZeRO‑3")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model identifier")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Name of the dataset on the HF hub")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration (if any)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device batch size")
    parser.add_argument("--block_size", type=int, default=512, help="Sequence length for language modelling")
    parser.add_argument("--deepspeed_config", type=str, required=True, help="Path to the DeepSpeed JSON config file")
    parser.add_argument("--output_dir", type=str, default="./ds_output", help="Where to save the fine‑tuned model")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision")
    args = parser.parse_args()

    # TrainingArguments MUSZĄ być zadeklarowane przed modelem!
    # Dzięki temu Hugging Face wie, że ma aktywować kontekst ZeRO-3 
    # i nie załaduje całego modelu do pamięci RAM na raz.
    # Define training arguments; the deepspeed config enables ZeRO‑3.  We
    # disable reporting to external trackers like WandB by passing an
    # empty list to ``report_to``.  Mixed precision can be enabled
    # globally via the ``fp16`` flag.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=1,
        logging_steps=10,
        fp16=args.fp16,
        deepspeed=args.deepspeed_config,
        report_to=[],
    )

    # Load tokenizer and model.  We intentionally avoid automatic
    # sharding or wrapping here because DeepSpeed handles parameter
    # partitioning internally via the ZeRO‑3 engine.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Load and preprocess the dataset.  We drop incomplete blocks to
    # ensure consistent sequence lengths across all processes.
    dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")
    tokenised = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.block_size),
        batched=True,
        remove_columns=dataset.column_names,
    )
    sequences = [torch.tensor(s, dtype=torch.long) for s in tokenised["input_ids"]]
    train_dataset = CausalDataset(sequences)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    # Save the resulting model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()