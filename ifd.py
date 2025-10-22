import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_from_disk,concatenate_datasets
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from transformers import AutoTokenizer, AutoModelForCausalLM

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='mistral_uf')
    parser.add_argument("--save_path", type=str, default='./adpo/metric/dpo_2k_mistral_uf_ppl.jsonl')
    parser.add_argument("--model_name", type=str, default='./models/model-uf-mis/mis-uf-3b-rand-2k/checkpoint-250')
    parser.add_argument("--tokenizer_name", type=str, default="./models/model/Llama-3.2-3B-sft")
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()
    return args

# Used to get the ppl and emb for the whole input
def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):

    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        with torch.no_grad(): 
            outputs = model(input_ids, labels=input_ids.contiguous())
        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item()

    except:
        return 0, 0

# Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):

    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        start_index = text.rfind(target_span)
        start_token = len(tokenizer.encode(text[:start_index]))
        end_token = input_ids.shape[1]

        labels = input_ids.clone()
        labels[0, :start_token] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)

        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item(), loss.to('cpu').item() * -(end_token - start_token)
    
    except:
        return 0, 0

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def main():

    args = parse_args()
    print(args)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True).to("cuda") # for llama series
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    model.eval()
    if args.dataset_name == 'tldr':
        ds = load_from_disk('dataset/TLDR')['train']
        # ds = ds.select(range(10))
    elif args.dataset_name == 'hh':
        ds1 = load_from_disk('dataset/HH/hh-helpful')['train']
        ds2 = load_from_disk('dataset/HH/hh-harmless')['train']
        ds = concatenate_datasets([ds1,ds2])
    elif args.dataset_name == 'uf':
        ds = load_from_disk('dataset/ultrafeedback_binarized')
    elif args.dataset_name == 'llama_uf':
        ds = load_from_disk('dataset/llama3-ultrafeedback')['train']
    elif args.dataset_name == 'mistral_uf':
        ds = load_from_disk('dataset/mistral-ultrafeedback')['train']

    if not os.path.exists(args.save_path):
        with open(args.save_path, "w") as file:
            pass  # Creates an empty file

    with open(args.save_path, "r") as file:
        exsisting_num =  sum(1 for _ in file)
        print(f'existing row:{exsisting_num}')
    ds = ds.select(range(exsisting_num, len(ds)))


    for i in tqdm(range(len(ds))):

        data_i = ds[i]
        if args.dataset_name == 'tldr':
            prompt_messages = [{'role':'user','content':f"Summarize the following text with a TL;DR:\n\n{data_i['info']['post']}"}]
            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = data_i['summaries'][data_i['choice']]['text']
            rejected_messages = data_i['summaries'][1-data_i['choice']]['text']
        elif args.dataset_name == 'hh':
            prompt = extract_anthropic_prompt(data_i['chosen'])
            prompt_messages = [{'role':'user','content':f"Complete the following dialogue:{prompt}"}]
            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = data_i['chosen'][len(prompt):]
            rejected_messages = data_i['rejected'][len(prompt):]
        elif 'uf' in args.dataset_name:
            prompt_messages = data_i["chosen"][:-1]
            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = data_i["chosen"][-1]['content']
            rejected_messages = data_i["rejected"][-1]['content']
        instruct_i = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True).replace(tokenizer.bos_token, "")
        whole_text_chosen = instruct_i + chosen_messages
        whole_text_rejected = instruct_i + rejected_messages

        
        ppl_chosen_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, chosen_messages, args.max_length)
        ppl_rejected_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, rejected_messages, args.max_length)
        ppl_chosen_condition, loss_chosen_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text_chosen, chosen_messages, args.max_length)
        ppl_rejected_condition, loss_rejected_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text_rejected, rejected_messages, args.max_length)


        temp_data_i = {}
        temp_data_i['ppl_chosen'] = [ppl_chosen_alone,ppl_chosen_condition]
        temp_data_i['ppl_rejected'] = [ppl_rejected_alone,ppl_rejected_condition]
        temp_data_i['reward'] = [loss_chosen_condition - loss_rejected_condition]

        with open(args.save_path, "a") as file:
            file.write(json.dumps(temp_data_i) + '\n')

    model.cpu()
    
    print('Done: Data Analysis:',args.dataset_name)

if __name__ == "__main__":
    main()

# nohup python adpo/ifd.py >>./adpo/log/metric_mistral_uf.log &