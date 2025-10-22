from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
import json
import os


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    model_name_or_path: Optional[str] = field(
        default="models/model/Llama-3-8B-sft",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="dataset/ultrafeedback_binarized",
        metadata={"help": "the location of the dataset name or path"},
    )
    round: Optional[int] = field(
        default=0,
        metadata={"help": "the round of the generation"},
    )
    local_index: Optional[int] = field(
        default=0,
        metadata={"help": "the local index of the agent"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=1,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=5,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=4096,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="prompt",
        metadata={"help": "the key of the dataset"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_path = script_args.model_name_or_path
print("model_path", model_path)
seed = script_args.seed
# set seed
torch.manual_seed(seed)
np.random.seed(seed)

llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",
    max_model_len=script_args.max_input_length,
    gpu_memory_utilization=0.9,
    load_format="auto",
    seed=42,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(
    temperature=script_args.temperature,
    top_p=1.0,
    max_tokens=script_args.max_new_tokens,
    n=script_args.K,
    stop_token_ids=[128001, 128009],
    #stop=["<|user|>"],
)


# ds = load_dataset(script_args.dataset_name_or_path, split="train")
ds = load_from_disk(script_args.dataset_name_or_path)
# ds = ds.select(range(20000*script_args.round,20000*(script_args.round+1)))
ds = ds.select(range(script_args.round,60000,3))

ds = ds.select(range(5000*script_args.local_index,5000*(script_args.local_index+1)))

ds = ds.map(
    lambda x: {
        "prompts": tokenizer.apply_chat_template([{"role": "user", "content": x[script_args.dataset_key]}], tokenize=False, add_generation_prompt=True)
    }
)

# data_size = len(ds["prompt"])
# one_num_share = int(data_size / script_args.my_world_size)
# ds = ds.select(np.arange(script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share))

# print([script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share])
# print(f'generation range: {script_args.local_index * one_num_share} - {(script_args.local_index + 1) * one_num_share}')
print(ds)
print(ds[0]['prompts'])


prompts = ds["prompts"]
outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)


completions = []
used_prompts = []
gathered_data = []
for i, output in enumerate(outputs):
    tmp_data = {"prompt": ds[i][script_args.dataset_key], "responses": [out.text for out in output.outputs]}
    gathered_data.append(tmp_data)


print("I collect ", len(gathered_data), "samples")
os.makedirs(script_args.output_dir, exist_ok=True)

with open(script_args.output_dir + 'round' + str(script_args.round) + '-idx' + str(script_args.local_index) + '.json', "w", encoding="utf8") as f:
    for i in range(len(gathered_data)):
        json.dump(gathered_data[i], f, ensure_ascii=False)
        f.write('\n')