# Easy Distributed LLM Training with Transformers, DeepSpeed, and FSDP

## Goal

Fine-tuning a large language model is easy to express in Python, but the model
quickly becomes too large for a single GPU. The goal of this project is to show
that the training workflow stays almost the same while distributed methods add
the memory handling needed for multi-GPU training.

The model, dataset, tokenizer, and training objective stay the same. The main
difference is how model parameters, gradients, and optimizer states are
distributed across GPUs.

We compare two distributed approaches implemented in this repository:

| Method | Best for | Code complexity | Main idea |
| --- | --- | --- | --- |
| Default Hugging Face `Trainer` | Simple single-GPU or small-model runs | Low | Let Transformers handle the training loop |
| DeepSpeed ZeRO-3 + `Trainer` | Easy large-model training | Low to medium | Add a DeepSpeed config to `TrainingArguments` |
| Native PyTorch FSDP | Maximum control | High | Manually initialize distributed training and shard the model |

## Experimental Setup

Both distributed jobs use the same model, dataset, and training objective so the
comparison focuses on the distributed training method rather than a difference
in data or architecture.

| Item | Value |
| --- | --- |
| Model | `bigscience/bloom-3b` |
| Task | Causal language modeling |
| Dataset | `wikitext` |
| Dataset configuration | `wikitext-2-raw-v1` |
| Split | `train` |
| Epochs | `1` |
| Sequence length | `512` tokens |
| Per-device batch size | `2` |

## Baseline: Normal Transformers Training

The default Hugging Face workflow is short and readable. After preparing a
tokenizer, model, and dataset, training can be launched with `Trainer`.

```python
model_name = "bigscience/bloom-3b"
dataset_name = "wikitext"
dataset_config = "wikitext-2-raw-v1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
dataset = load_dataset(dataset_name, dataset_config, split="train")

training_args = TrainingArguments(
    output_dir="./outputs/baseline",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

This is the reference point for all distributed methods. The goal is not to
change the training objective, but to make the same workflow possible when the
model no longer fits comfortably on one GPU.

Distributed training should feel like scaling the baseline, not rewriting the
whole experiment from zero.

## Method 1: DeepSpeed ZeRO-3 with Hugging Face Trainer

DeepSpeed integrates directly with Hugging Face `Trainer`. The training code
stays almost identical to the baseline. The important change is the
`deepspeed` argument, which points to a JSON configuration file.

```python
training_arg_values = {
    "output_dir": args.output_dir,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": args.batch_size,
    "num_train_epochs": args.epochs,
    "gradient_accumulation_steps": 1,
    "learning_rate": args.lr,
    "logging_steps": 10,
    "bf16": args.bf16,
    "deepspeed": args.deepspeed_config,
    "report_to": [],
    "max_steps": args.max_steps,
}

supported_args = set(inspect.signature(TrainingArguments.__init__).parameters)
training_args = TrainingArguments(
    **{key: value for key, value in training_arg_values.items() if key in supported_args}
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

The DeepSpeed configuration enables ZeRO Stage 3, where model parameters,
gradients, and optimizer states are partitioned across GPUs.

With DeepSpeed, the Python training code remains close to the normal `Trainer`
version. Most distributed behavior is moved into configuration.

### DeepSpeed configuration file

DeepSpeed needs a separate JSON file because ZeRO-3 behavior is configured
outside the Python training loop. In this project, `ds_config_zero3.json`
contains the important distributed-memory decisions:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "steps_per_print": 10,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 500000000,
    "stage3_prefetch_bucket_size": 500000000,
    "stage3_param_persistence_threshold": 1000000,
    "sub_group_size": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "fp16": { "enabled": false },
  "bf16": { "enabled": true },
  "wall_clock_breakdown": false
}
```

The most important setting is `"stage": 3`. According to the DeepSpeed ZeRO
documentation, Stage 3 partitions the optimizer states, gradients, and model
parameters across data-parallel workers, which is why it can reduce per-GPU
memory enough for larger models. In the current configuration, batch sizes are
left on `"auto"`, BF16 mixed precision is enabled, and CPU/NVMe offload is not
enabled.

## Method 2: Native PyTorch FSDP

Fully Sharded Data Parallel (FSDP) is PyTorch's native sharding approach. It is
more explicit than DeepSpeed: the code initializes distributed execution,
assigns each process to a GPU, wraps the model, and runs the training loop.

```python
torch.distributed.init_process_group(backend="nccl")
# added: initialize one distributed process per GPU

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
# added: bind each process to its local GPU

dtype = torch.bfloat16 if args.bf16 else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=dtype,
)

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
# added: wrap Bloom blocks with FSDP and shard model state across workers
```

The training loop is manual:

```python
optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

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
```

Native FSDP gives more control over the distributed training details, but it
also requires more code than the high-level `Trainer` approaches.

FSDP is powerful when we want control over the training loop, but this control
makes the implementation longer and more sensitive to distributed setup
details.

### Native FSDP configuration

The native PyTorch FSDP script in this project configures FSDP mostly in Python:
it initializes the NCCL process group, chooses the local GPU, wraps the model
with `FullyShardedDataParallel`, uses `transformer_auto_wrap_policy` for
`BloomBlock` layers, and runs a manual training loop. Therefore, the native
script does not need an external JSON config file in the same way that
DeepSpeed does.

PyTorch's FSDP documentation describes this method as a way to shard training
state across ranks. Practically, this means less replicated memory per GPU, but
more communication and more responsibility in the user code.

### FSDP configuration file

The repository also contains `fsdp_config.json`. The current native PyTorch
FSDP script does not load it directly, because its sharding behavior is defined
in Python with `FSDP(...)`. The file is still useful as a compact reference for
the same ideas: full sharding, backward prefetching, BF16 mixed precision, and
state-dict behavior. The running script saves sharded checkpoints with
`StateDictType.SHARDED_STATE_DICT`.

## Results: One-Epoch Comparison

Both methods were prepared for the same type of one-epoch language-model
fine-tuning task. The log files show that both methods completed the epoch, but
the measured runtime and final loss differ. In this run, native PyTorch FSDP was
faster and reached a lower final loss, while the implementation style remained
much more manual than the DeepSpeed `Trainer` workflow.

| Method | Training result | Speed result | What it shows |
| --- | ---: | ---: | --- |
| DeepSpeed ZeRO-3 + `Trainer` | `4.725` training loss | `851.5 s`, `5.466 samples/s` | Clean high-level distributed training with minimal code changes |
| Native PyTorch FSDP | `4.2292` final training loss | `752.1 s`, `6.19 samples/s` | Better measured result in this run, with more manual distributed code |

The run summary:

```text
DeepSpeed ZeRO-3:
  train_runtime = 851.5 s
  train_samples_per_second = 5.466
  train_steps_per_second = 0.684
  train_loss = 4.725

Native PyTorch FSDP:
  total_training_time = 752.1 s
  training_speed = 6.19 samples/s
  final_training_loss = 4.2292
```

The practical difference is still strongly about developer experience: how much
code and configuration must be managed to run the same training objective at
scale. The measured run also shows that implementation choices and runtime
configuration can have a large performance impact.

### Speed interpretation

Speed is not only about the final runtime. For distributed LLM training, we also
care about throughput, memory pressure, and communication overhead.

| Method | Runtime view | Throughput view | Memory view | Speed tradeoff |
| --- | --- | --- | --- | --- |
| DeepSpeed ZeRO-3 + `Trainer` | `851.5 s` for one epoch | `5.466 samples/s` and `0.684 steps/s` | Strong memory reduction through ZeRO-3 partitioning and BF16 precision | ZeRO-3 reduces replicated state, but communication and checkpoint overhead can reduce raw speed |
| Native PyTorch FSDP | `752.1 s` for one epoch | `6.19 samples/s` | Strong memory reduction through full sharding | More direct PyTorch control, but all-gather/reduce-scatter communication can dominate |

The fastest-looking method is not always the most useful one. For large models,
the first success criterion is often "does it fit in GPU memory?" Once the
model fits, we compare step time and throughput.

## Comparison

| Question | DeepSpeed ZeRO-3 | Native PyTorch FSDP |
| --- | --- | --- |
| How much code changes from baseline? | Small | Large |
| Who manages the training loop? | `Trainer` | User code |
| Where is sharding configured? | JSON config | Python code |
| Main advantage | Easiest large-model path | Maximum control |
| Main tradeoff | Extra DeepSpeed dependency | More manual code |

## Strengths and Weaknesses

| Method | Pluses | Minuses |
| --- | --- | --- |
| DeepSpeed ZeRO-3 + `Trainer` | Minimal Python changes; strong memory savings; good Hugging Face integration; config-driven ZeRO-3 setup | Adds a DeepSpeed dependency; performance depends heavily on config; communication and checkpointing overhead can slow training |
| Native PyTorch FSDP | Native PyTorch solution; maximum control over model wrapping, data loading, checkpointing, and training loop; good for custom research code | More code; more distributed details to manage; speed depends on communication efficiency and correct sharding/wrapping choices |

DeepSpeed is usually the easiest path. Native FSDP is the most controllable
path.

## Conclusion

Both distributed methods complete the same one-epoch fine-tuning task on the
same model and data. In the recorded logs, FSDP finishes faster and with lower
final loss, while DeepSpeed keeps the training code closer to the normal
Transformers `Trainer` workflow. In practice, the comparison is not only model
quality, but also how much code and configuration the user must manage.

**DeepSpeed ZeRO-3 is the easiest path from normal Transformers training to
large-model distributed training.** It keeps the `Trainer` workflow and moves
most distributed behavior into a config file.

**Native PyTorch FSDP is the most flexible option.** It is useful when the
training loop needs custom behavior, but the implementation is longer.

Final message: distributed LLM training does not require changing the whole
experiment. The model and dataset stay the same. We mainly choose how much of
the distributed system we want to manage ourselves.
