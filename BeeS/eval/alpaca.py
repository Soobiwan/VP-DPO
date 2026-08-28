import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
import json
import argparse
import logging
import random
import torch
import datasets
import vllm
import yaml

# with open('adpo/config/qwen-7b-vllm-gen.yaml', 'r') as file:
with open('adpo/config/llama3-8b-vllm-gen.yaml', 'r') as file:
# with open('adpo/config/mistral-7b-vllm-gen.yaml', 'r') as file:
    configs = yaml.safe_load(file)

config = configs['Llama']
# config = configs['Mistral']
# config = configs['Qwen']


with open(config['prompt_template'], 'r') as file:
    prompt_template = file.read()

def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    logging.info("loading data and model...")
    alpaca_eval_data = datasets.load_dataset(args.reference_path,'alpaca_eval',trust_remote_code=True)['eval'] # text-davinci-003 as the generator
    # alpaca_eval_data = datasets.load_dataset(args.reference_path,'alpaca_eval_gpt4_baseline',trust_remote_code=True)['eval'] # gpt4-turbo as the generator
    
    prompts = []
    
    for example in alpaca_eval_data:
        prompt = example["instruction"] 
        prompt = prompt_template.format(instruction=prompt)
        prompts.append(prompt)

    
    # model = vllm.LLM(
    #     model=args.model_name_or_path,
    #     tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
    #     # tokenizer_mode="slow",
    #     tensor_parallel_size=torch.cuda.device_count(),
    #     gpu_memory_utilization=0.7,
    # )
    model = vllm.LLM(model=config['completions_kwargs']['model_name'],
                    dtype=config['completions_kwargs']['model_kwargs']['torch_dtype'],
                    trust_remote_code=True,
                    gpu_memory_utilization=0.9,
                    # enforce_eager=True,  # for gemma2 model
                    tensor_parallel_size=torch.cuda.device_count())
    sampling_params = vllm.SamplingParams(
        temperature=config['completions_kwargs']['temperature'],
        top_p=config['completions_kwargs']['top_p'],
        max_tokens=config['completions_kwargs']['max_new_tokens'],
        stop_token_ids=config['completions_kwargs']['stop_token_ids']
    )
    outputs = model.generate(prompts, sampling_params)
    outputs = [it.outputs[0].text for it in outputs]
       

    # model_name = config['completions_kwargs']['model_name'].split('/')[-1]
    model_name = config['completions_kwargs']['save_name']
    model_results = []
    for example, output in zip(alpaca_eval_data, outputs):
        example["output"] = output
        example["generator"] = f"{model_name}"
        # fout.write(json.dumps(example) + "\n")
        model_results.append(example)
    with open(os.path.join(args.save_dir, f"{model_name}.json"), "w") as fout:
        json.dump(model_results, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_path",
        type=str,
        default="dataset/alpaca_eval",
        help="Path to the reference outputs. "
             "Alpaca_eval leaderboard use davinci_003 to generate the reference outputs, "
             "but they limit the max_tokens to 300. Here we regenerated reference outputs with max_tokens=2048.",
    )
    parser.add_argument(
        "--save_dir",
        type=str, 
        default="adpo/eval/result/alpaca_eval2/")
    args = parser.parse_args()

    main(args)