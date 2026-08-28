#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
from datasets import load_dataset,load_from_disk,concatenate_datasets
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    model_name_or_path: Optional[str] = field(
        default="",
        metadata={"help": "the location of the SFT model name or path"},
    )
    task: Optional[str] = field(
        default="uf",
        metadata={"help": "the task to use"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    K: Optional[int] = field(
        default=1,
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
    


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_path = './models/' + script_args.model_name_or_path
seed = script_args.seed
# set seed
torch.manual_seed(seed)
np.random.seed(seed)


llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",
    max_model_len=script_args.max_input_length,
    trust_remote_code=True,
    gpu_memory_utilization=0.5,
    seed=42,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(
    temperature=script_args.temperature,
    top_p=1.0,
    max_tokens=script_args.max_new_tokens,
    n=script_args.K,
    stop_token_ids=[128001, 128009],
)

if script_args.task == 'tldr':
    ds = load_dataset('json',data_files='dataset/test.jsonl')['train']
    ds = ds.select(range(0,4000,10))
    chosen_response_list = [ds[i]['summary'] for i in range(len(ds))]
    ds = ds.map(
        lambda x: {
            "prompt": tokenizer.apply_chat_template([{'role':'user','content':f"Summarize the following text with a TL;DR:\n\n{x['post']}"}], tokenize=False, add_generation_prompt=True)
        }
    )
elif script_args.task == 'hh':
    ds1 = load_from_disk('dataset/HH/hh-helpful')['test']
    ds2 = load_from_disk('dataset/HH/hh-harmless')['test']
    ds = concatenate_datasets([ds1,ds2])
    ds = ds.select(range(0,4000,10))
    chosen_response_list = [ds[i]['chosen'][len(extract_anthropic_prompt(ds[i]['chosen'])):] for i in range(len(ds))]
    ds = ds.map(
        lambda x: {
            "prompt": tokenizer.apply_chat_template([{'role':'user','content':f"Complete the following dialogue:{extract_anthropic_prompt(x['chosen'])}"}], tokenize=False, add_generation_prompt=True)
        }
    )
elif 'uf' in script_args.task:
    ds = load_dataset('json',data_files='dataset/alpaca_eval/alpaca_eval.json')['train']
    chosen_response_list = [ds[i]['output'] for i in range(len(ds))]
    ds = ds.map(
        lambda x: {
            "prompt": tokenizer.apply_chat_template([{'role':'user','content':x['instruction']}], tokenize=False, add_generation_prompt=True)
        }
    )
N = len(ds["prompt"])
prompts = ds["prompt"]
outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)


completions = []
gathered_data = []
for i, output in enumerate(outputs):
    if script_args.task == 'tldr':
        tmp_data = {"prompt": ds[i]['post'], "model_response": [out.text for out in output.outputs], "chosen_response": [chosen_response_list[i]]}
    elif script_args.task == 'hh':
        tmp_data = {"prompt": extract_anthropic_prompt(ds[i]['chosen']), "model_response": [out.text for out in output.outputs], "chosen_response": [chosen_response_list[i]]}
    elif 'uf' in script_args.task:
        tmp_data = {"prompt": ds[i]['instruction'], "model_response": [out.text for out in output.outputs], "chosen_response": [chosen_response_list[i]]}
    gathered_data.append(tmp_data)


print("I collect ", len(gathered_data), "samples")


with open(f"./adpo/eval/result/gen/{script_args.output_dir}.json", "w", encoding="utf8") as f:
    for i in range(len(gathered_data)):
        json.dump(gathered_data[i], f, ensure_ascii=False)
        f.write('\n')