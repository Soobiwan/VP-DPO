import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='llama_uf_armo')
    parser.add_argument("--save_path", type=str, default='./adpo/metric/llama_uf_armo_rm_scores.jsonl')
    parser.add_argument("--model_name", type=str, default='models/model/skywork-8b-rm')
    parser.add_argument("--device", type=str, default="cuda:6")
    args = parser.parse_args()
    return args

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def get_reward_score(rm_model, rm_tokenizer, prompt_messages, response, device):
    conv = prompt_messages + [{"role": "assistant", "content": response}]
    conv_tokenized = rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        score = rm_model(conv_tokenized).logits[0][0].item()
    return score

def main():
    args = parse_args()
    print(args)
    
    # Load model and tokenizer
    device = args.device
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    rm.eval()

    # Load dataset
    if args.task == 'tldr':
        ds = load_from_disk('dataset/TLDR')['train']
    elif args.task == 'hh':
        ds1 = load_from_disk('dataset/HH/hh-helpful')['train']
        ds2 = load_from_disk('dataset/HH/hh-harmless')['train']
        ds = concatenate_datasets([ds1,ds2])
    elif args.task == 'uf':
        ds = load_from_disk('dataset/ultrafeedback_binarized')
    elif args.task == 'llama_uf':
        ds = load_from_disk('dataset/llama3-ultrafeedback')['train']
    elif args.task == 'mistral_uf':
        ds = load_from_disk('dataset/mistral-ultrafeedback')['train']
    elif args.task == 'llama_uf_armo':
        ds = load_from_disk('dataset/llama3-ultrafeedback-armo')['train']

    # Create save file if it doesn't exist
    if not os.path.exists(args.save_path):
        with open(args.save_path, "w") as file:
            pass

    # Resume from last saved position
    with open(args.save_path, "r") as file:
        existing_num = sum(1 for _ in file)
        print(f'existing row:{existing_num}')
    ds = ds.select(range(existing_num, len(ds)))

    for i in tqdm(range(len(ds))):
        data_i = ds[i]
        
        # Process different datasets
        if args.task == 'tldr':
            prompt_messages = [{'role':'user','content':f"Summarize the following text with a TL;DR:\n\n{data_i['info']['post']}"}]
            chosen_messages = data_i['summaries'][data_i['choice']]['text']
            rejected_messages = data_i['summaries'][1-data_i['choice']]['text']
        elif args.task == 'hh':
            prompt = extract_anthropic_prompt(data_i['chosen'])
            prompt_messages = [{'role':'user','content':f"Complete the following dialogue:{prompt}"}]
            chosen_messages = data_i['chosen'][len(prompt):]
            rejected_messages = data_i['rejected'][len(prompt):]
        elif args.task == 'uf' or args.task == 'llama_uf' or args.task == 'mistral_uf' or args.task == 'llama_uf_armo':
            prompt_messages = data_i["chosen"][:-1]
            chosen_messages = data_i["chosen"][-1]['content']
            rejected_messages = data_i["rejected"][-1]['content']

        # Calculate reward scores
        chosen_score = get_reward_score(rm, rm_tokenizer, prompt_messages, chosen_messages, device)
        rejected_score = get_reward_score(rm, rm_tokenizer, prompt_messages, rejected_messages, device)
        
        # Save results
        temp_data_i = {
            'chosen_score': chosen_score,
            'rejected_score': rejected_score,
            'reward_diff': chosen_score - rejected_score
        }

        with open(args.save_path, "a") as file:
            file.write(json.dumps(temp_data_i) + '\n')

    rm.cpu()
    print('Done: Data Analysis:', args.task)

if __name__ == "__main__":
    main()