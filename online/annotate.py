import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import argparse
import tqdm
import numpy as np
import datasets

parser = argparse.ArgumentParser()
parser.add_argument("--generation_file", type=str, default="", help="Path to the output generation file")
parser.add_argument("--reward_model", type=str, default="models/model/skywork-8b-rm", help="Path to reward model")
parser.add_argument("--output_dir", type=str, default="", help="Path to output directory")
args = parser.parse_args()

print(args)
print('start annotate.')
generation_file = args.generation_file
# with open(generation_file, 'r') as f:
#     output_data = json.load(f)
with open(generation_file, 'r') as f:
    # 按行读取并解析每个JSON对象
    output_data = []
    for line in f:
        if line.strip():  # 跳过空行
            output_data.append(json.loads(line))

# inputs = [data["prompt"] for data in output_data]
# candidates_texts = [data["responses"] for data in output_data]

model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, 
                                                           device_map="cuda", 
                                                           trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True)

os.makedirs(args.output_dir, exist_ok=True)

for data in tqdm.tqdm(output_data):
    prompt = data["prompt"]
    candidates = data["responses"]
    scores = []
    for candidate in candidates:
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": candidate}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model(input_ids)
            # score = output.score.float().item()
            score = output.logits[0][0].item()
            scores.append(score)
    data["all_rm_scores"] = scores

file_name = os.path.basename(args.generation_file).split('.json')[0] + "_rm.json"
with open(os.path.join(args.output_dir, file_name), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Annotated outputs saved to {os.path.join(args.output_dir, file_name)}")

# Binarize data: win = highest scoring reponse; lose = lowest scoring response
for data in output_data:
    chosen_idx = np.argmax(data["all_rm_scores"])
    rejected_idx = np.argmin(data["all_rm_scores"])
    chosen = []
    chosen.append({
        "role": "user",
        "content": data["prompt"]
    })
    chosen.append({
        "role": "assistant",
        "content": data["responses"][chosen_idx]
    })
    rejected = []
    rejected.append({
        "role": "user",
        "content": data["prompt"]
    })
    rejected.append({
        "role": "assistant",
        "content": data["responses"][rejected_idx]
    })
    data.update({
        "chosen": chosen,
        "rejected": rejected,
    })

output_file = os.path.basename(args.generation_file).split('.json')[0] + "_bin.json"
with open(os.path.join(args.output_dir, file_name), 'w') as f:
    json.dump(output_data, f, indent=4)
print(f"Binarized outputs saved to {output_file}")

# Convert the data to Hugging Face datasets format
dataset = datasets.Dataset.from_list(output_data)
dataset.save_to_disk(os.path.join(args.output_dir))
print(f"Binarized dataset saved to {os.path.join(args.output_dir)}")