import os
from typing import Any, List, Literal, Optional
from dataclasses import dataclass, field
import wandb
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainingArguments
import torch
from trl import KTOTrainer, KTOConfig
from adpo.sampler import Sampling
from adpo.m_sampler import Sampling_ensemble
os.environ['WANDB_PROJECT'] = "adpo-final"

def transform_dataset(dataset, tokenizer):
    new_dict = {
        "prompt": [],
        "completion": [],
        "label": [],
    }
    for example in dataset:
        new_dict["prompt"].append(tokenizer.apply_chat_template(example["chosen"][:-1], tokenize=False, add_generation_prompt=True))
        new_dict["completion"].append(example["chosen"][-1]["content"] + tokenizer.eos_token)
        new_dict["label"].append(True)
        new_dict["prompt"].append(tokenizer.apply_chat_template(example["rejected"][:-1], tokenize=False, add_generation_prompt=True))
        new_dict["completion"].append(example["rejected"][-1]["content"] + tokenizer.eos_token)
        new_dict["label"].append(False)
    return Dataset.from_dict(new_dict)

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
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=8)
    learning_rate: Optional[float] = field(default=5e-7) # 5e-7, 1e-6, 5e-6
    weight_decay: Optional[float] = field(default=0.0)
    model_name: Optional[str] = field(
        default="",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default="tldr",
        metadata={
            "help": "the path of the preference datasets",
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the model."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        #default="paged_adamw_32bit",
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        # default="constant_with_warmup",
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )

    max_training_samples: Optional[int] = field(default=-1, metadata={"help": "the maximum sample size"})

    max_length: Optional[int] = field(default=2048)
    output_dir: Optional[str] = field(default="")
    run_name: Optional[str] = field(default="")
    beta: Optional[float] = field(default=0.1)
    strategy: Literal["random", "ex_reward_margin", "im_reward_margin", "ppl_margin", "add", "mul"] = field(default="random")
    part: Literal["top", "mid","bot", ""] = field(default="")
    num_samples: Optional[int] = field(default=2000)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]



tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# load dataset and mapping
if script_args.dataset_name == 'uf':
    ds = load_from_disk('dataset/ultrafeedback_binarized')
elif script_args.dataset_name == 'llama_uf':
    ds = load_from_disk('dataset/llama3-ultrafeedback')['train']
elif script_args.dataset_name == 'mistral_uf':
    ds = load_from_disk('dataset/mistral-ultrafeedback')['train']


if script_args.strategy in ['add', 'mul']:
    ds_tr = Sampling_ensemble(script_args.dataset_name, ds, script_args.strategy, script_args.part, script_args.num_samples)
else:
    ds_tr = Sampling(script_args.dataset_name, ds, script_args.strategy, script_args.part, script_args.num_samples)
print(ds_tr)
# ds_tr = ds_tr.map(apply_chat_template,fn_kwargs={"tokenizer":tokenizer, "name":script_args.dataset_name}, num_proc=8)

# ds_tr = Sampling(script_args.dataset_name, ds, script_args.strategy, script_args.part, script_args.num_samples)
ds_tr = transform_dataset(ds_tr, tokenizer)
print(ds_tr)


model = AutoModelForCausalLM.from_pretrained(script_args.model_name,
    torch_dtype = torch.bfloat16,
    # load_in_4bit = True,
    attn_implementation='flash_attention_2',  
    trust_remote_code = True,
    use_cache = False
)

model_ref = AutoModelForCausalLM.from_pretrained(script_args.model_name,
    torch_dtype = torch.bfloat16,
    # load_in_4bit = True,
    attn_implementation='flash_attention_2',  
    trust_remote_code = True,
    use_cache = False
)


kto_trainer = KTOTrainer(
    model = model,
    ref_model = model_ref,
    args = KTOConfig(
        per_device_train_batch_size = script_args.per_device_train_batch_size,
        gradient_accumulation_steps = script_args.gradient_accumulation_steps,
        warmup_ratio = 0.1,
        # label_smoothing = 0.1,
        num_train_epochs = script_args.num_train_epochs,
        learning_rate = script_args.learning_rate,
        bf16 = script_args.bf16,
        optim = script_args.optim,
        weight_decay = script_args.weight_decay,
        lr_scheduler_type = script_args.lr_scheduler_type,
        seed = 42,
        output_dir = script_args.output_dir,
        gradient_checkpointing = script_args.gradient_checkpointing, # True or "unsloth" for very long context
        gradient_checkpointing_kwargs= {
            "use_reentrant": False
        },
        report_to="wandb",
        run_name=script_args.run_name,
        save_strategy="steps",
        save_steps=1000000000,
        loss_type="kto",
        remove_unused_columns=True,
        logging_strategy="steps",
        logging_steps=5,
        beta = script_args.beta,
        max_length = script_args.max_length,
        max_prompt_length =1500,
    ),
    train_dataset = ds_tr,
    tokenizer = tokenizer,
)

kto_trainer.train()