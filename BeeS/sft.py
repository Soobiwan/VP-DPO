from dataclasses import dataclass, field
from typing import Optional
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,7"
import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer

os.environ['WANDB_PROJECT'] = "adpo"


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, \
    what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. \
            You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.0)
    model_name: Optional[str] = field(
        # default="./models/meta-llama-3-8b",
        default="./models/Llama-3.2-1B",
        # default="./models/Llama-3.2-3B",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default="./dataset/SFT-OpenHermes-2.5-Standard",
        metadata={
            "help": "",
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[float] = field(
        default=2,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )

    max_training_samples: Optional[int] = field(default=-1, metadata={"help": "the maximum sample size"})

    max_length: Optional[int] = field(default=4096)
    output_dir: Optional[str] = field(default="./models/Llama-3.2-1B-sft")

    run_name: Optional[str] = field(default="1b-sft-20w")


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    save_strategy="no",
    save_steps=1000000000,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    gradient_checkpointing_kwargs= {
            "use_reentrant": False
        },
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=True,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.1,
    report_to="wandb",
    run_name=script_args.run_name
)


dataset = load_dataset(script_args.dataset_name, split="train")
dataset = dataset.shuffle(seed=42).select(range(200000))

if script_args.max_training_samples > 0:
    dataset = dataset.select(range(script_args.max_training_samples))

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
).to("cuda") # for llama series


model.config.use_cache = not script_args.gradient_checkpointing
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
print("We set the pad token as the eos token by default....")
# tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length
# llama template
tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"


def formatting_prompts_func(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


ds = dataset.map(formatting_prompts_func, batched=False)
# formatting_prompts_func

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=training_args,
    dataset_text_field="text",
    # formatting_func=,
    max_seq_length=script_args.max_length,
    packing=True,
)

trainer.train()
print("Saving last checkpoint of the model")

trainer.save_model(script_args.output_dir)
# trainer.model.save_pretrained(script_args.output_dir)
# tokenizer.save_pretrained(script_args.output_dir)