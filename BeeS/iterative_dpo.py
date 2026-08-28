import os
from typing import Any, List, Literal, Optional
from dataclasses import dataclass, field
import wandb
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainingArguments
import torch
from trl import DPOTrainer, DPOConfig

os.environ['WANDB_PROJECT'] = "adpo-online"

def is_openai_format(messages: Any) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
        return all("role" in message and "content" in message for message in messages)
    return False

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]



def apply_chat_template(
    example,
    tokenizer
):

    if all(k in example.keys() for k in ("chosen", "rejected")):
        if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
            raise ValueError(
                f"Could not format example as dialogue for the task! Require OpenAI format for all messages"
            )

        # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
        # We therefore need to extract the N-1 turns to form the prompt (for multi-turn dialogues)
        prompt_messages = example["chosen"][:-1]
        # Now we extract the final turn to define chosen/rejected responses
        chosen_messages = example["chosen"][-1]['content']
        rejected_messages = example["rejected"][-1]['content']
        
        example["prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True).replace(tokenizer.bos_token, "")
        example["chosen"] = chosen_messages # + tokenizer.eos_token
        example["rejected"] = rejected_messages # + tokenizer.eos_token
    else:
        raise ValueError(
            f"Could not format example as dialogue for the task! Require either the "
            f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
        )
   
    return example


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
        default="r0",
        metadata={
            "help": "the path of the preference datasets",
        },
    )
    dataset_path: Optional[str] = field(
        default="",
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
    strategy: Literal["random", "ex_reward_margin", "im_reward_margin", "ppl_margin"] = field(default="random")
    part: Literal["top", "mid","bot", ""] = field(default="")
    num_samples: Optional[int] = field(default=5000)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]



tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# load dataset and mapping

ds0 = load_from_disk(f'{script_args.dataset_path}{script_args.dataset_name}/0')
ds1 = load_from_disk(f'{script_args.dataset_path}{script_args.dataset_name}/1')
ds2 = load_from_disk(f'{script_args.dataset_path}{script_args.dataset_name}/2')
ds3 = load_from_disk(f'{script_args.dataset_path}{script_args.dataset_name}/3')
ds = concatenate_datasets([ds0, ds1, ds2, ds3])

if script_args.num_samples != -1:
    # top 5k
    import numpy as np
    score_list = np.zeros(20000)
    for i in range(20000):
        scores = ds[i]['all_rm_scores']
        score_list[i] = max(scores) - min(scores)
    indexs = np.argsort(score_list)[-script_args.num_samples:]
    print(f'lowest score:{score_list[indexs[0]]}')
    ds_tr = ds.select(indexs)
else:
    ds_tr = ds.shuffle(seed=42)

ds_tr = ds_tr.map(apply_chat_template,fn_kwargs={"tokenizer":tokenizer}, num_proc=8)


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


dpo_trainer = DPOTrainer(
    model = model,
    ref_model = model_ref,
    args = DPOConfig(
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
        loss_type="sigmoid",
        remove_unused_columns=True,
        logging_strategy="steps",
        logging_steps=5,
    ),
    beta = script_args.beta,
    train_dataset = ds_tr,
    tokenizer = tokenizer,
    max_length = script_args.max_length,
    max_prompt_length =1500,
    # peft_config=peft_config,
)

dpo_trainer.train()

# Save the model state
# model = dpo_trainer.model
# model.save_pretrained(script_args.output_dir, safe_serialization=True)
# tokenizer.save_pretrained(script_args.output_dir)