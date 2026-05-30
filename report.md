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

## Baseline: Normal Transformers Training

The default Hugging Face workflow is short and readable. After preparing a
tokenizer, model, and dataset, training can be launched with `Trainer`.

```python
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
training_args = TrainingArguments(
    output_dir="./outputs/deepspeed",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,                         # added: mixed precision
    deepspeed="ds_config_zero3.json",  # added: ZeRO-3 distributed config
    report_to=[],
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
  "train_micro_batch_size_per_gpu": 1,
  "zero_optimization": {
    "stage": 3,
    "offload_param": { "device": "cpu", "pin_memory": true },
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "overlap_comm": true
  },
  "fp16": { "enabled": true }
}
```

The most important setting is `"stage": 3`. According to the DeepSpeed ZeRO
documentation, Stage 3 partitions the optimizer states, gradients, and model
parameters across data-parallel workers, which is why it can reduce per-GPU
memory enough for larger models.

## Method 2: Native PyTorch FSDP

Fully Sharded Data Parallel (FSDP) is PyTorch's native sharding approach. It is
more explicit than DeepSpeed: the code initializes distributed execution,
assigns each process to a GPU, wraps the model, and runs the training loop.

```python
torch.distributed.init_process_group(backend="nccl")
# added: initialize one distributed process per GPU

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
# added: bind each process to its local GPU

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)

fully_shard(model)
model.to(device)
# added: shard model parameters across GPU workers
```

The training loop is manual:

```python
for epoch in range(epochs):
    sampler.set_epoch(epoch)

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Native FSDP gives more control over the distributed training details, but it
also requires more code than the high-level `Trainer` approaches.

FSDP is powerful when we want control over the training loop, but this control
makes the implementation longer and more sensitive to distributed setup
details.

### Native FSDP configuration

The native PyTorch FSDP script in this project configures FSDP mostly in Python:
it initializes the NCCL process group, chooses the local GPU, wraps the model
with `fully_shard(model)`, and runs a manual training loop. Therefore, the
native script does not need an external JSON config file in the same way that
DeepSpeed does.

PyTorch's FSDP documentation describes this method as a way to shard training
state across ranks. Practically, this means less replicated memory per GPU, but
more communication and more responsibility in the user code.

### FSDP configuration file

The repository also contains `fsdp_config.json`. The current native PyTorch
FSDP script does not load it directly, because its sharding behavior is defined
in Python with `fully_shard(model)`. The file is still useful as a compact
reference for the same ideas: full sharding, backward prefetching, mixed
precision, and full state dict checkpointing.

## Results: One-Epoch Comparison

Both methods were prepared for the same type of one-epoch language-model
fine-tuning task. The important result is that the training behavior is close:
the loss moves into the same range, while the implementation style is very
different.

| Method | Training result | Speed result | What it shows |
| --- | ---: | ---: | --- |
| DeepSpeed ZeRO-3 + `Trainer` | `4.837` training loss | `127.7 s`, `36.45 samples/s` | Clean high-level distributed training with minimal code changes |
| Native PyTorch FSDP | `4.78` training loss | `130.5 s`, similar step scale | Similar optimization behavior with more manual distributed code |

The run summary:

```text
DeepSpeed ZeRO-3:
  train_runtime = 127.7 s
  train_samples_per_second = 36.45
  train_steps_per_second = 1.144
  train_loss = 4.837

Native PyTorch FSDP:
  train_runtime = 130.5 s
  train_loss = 4.78
```

The training quality is very similar. The practical difference is the developer
experience: how much code and configuration must be managed to run the same
training objective at scale.

### Speed interpretation

Speed is not only about the final runtime. For distributed LLM training, we also
care about throughput, memory pressure, and communication overhead.

| Method | Runtime view | Throughput view | Memory view | Speed tradeoff |
| --- | --- | --- | --- | --- |
| DeepSpeed ZeRO-3 + `Trainer` | `127.7 s` for one epoch | `36.45 samples/s` and `1.144 steps/s` | Strong memory reduction through ZeRO-3 partitioning and CPU offload | Offload can make larger models fit, but CPU transfers may reduce raw speed |
| Native PyTorch FSDP | `130.5 s` for one epoch | Similar step scale | Strong memory reduction through full sharding | More direct PyTorch control, but all-gather/reduce-scatter communication can dominate |

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
| DeepSpeed ZeRO-3 + `Trainer` | Minimal Python changes; strong memory savings; good Hugging Face integration; config-driven CPU offload | Adds a DeepSpeed dependency; performance depends heavily on config; CPU offload can slow training if communication or transfer overhead is high |
| Native PyTorch FSDP | Native PyTorch solution; maximum control over model wrapping, data loading, checkpointing, and training loop; good for custom research code | More code; more distributed details to manage; speed depends on communication efficiency and correct sharding/wrapping choices |

DeepSpeed is usually the easiest path. Native FSDP is the most controllable
path.

## Conclusion

Both distributed methods reach very similar loss after one epoch because they
optimize the same model on the same data. In practice, the main difference is
not model quality, but how much code and configuration the user must manage.

**DeepSpeed ZeRO-3 is the easiest path from normal Transformers training to
large-model distributed training.** It keeps the `Trainer` workflow and moves
most distributed behavior into a config file.

**Native PyTorch FSDP is the most flexible option.** It is useful when the
training loop needs custom behavior, but the implementation is longer.

Final message: distributed LLM training does not require changing the whole
experiment. The model and dataset stay the same. We mainly choose how much of
the distributed system we want to manage ourselves.