import json
# import matplotlib.pyplot as plt
import numpy as np


def Sampling_ensemble(name, ds, strategy, part, Num):
    
    # 存储skywork reward difference的列表
    rewards_real = []
    f_reward = open(f'./adpo/metric/{name}_rm_scores.jsonl', 'r')

    for line in f_reward:
        reward_data = json.loads(line)
        reward_diff = reward_data['reward_diff']
        rewards_real.append(reward_diff)

    f_reward.close()
    rewards_real = np.array(rewards_real)
    # 处理nan
    if np.isnan(rewards_real).any():
        rewards_real = np.nan_to_num(rewards_real, nan=0)
    else:
        print("skywork reward difference 矩阵中没有 NaN 值")


    # 存储implicit reward difference的列表
    rewards = []

    f_dpo = open(f'./adpo/metric/dpo_2k_{name}_ppl.jsonl', 'r')
    f_sft = open(f'./adpo/metric/sft_{name}_ppl.jsonl', 'r')

    for dpo_line, sft_line in zip(f_dpo, f_sft):
        dpo_data = json.loads(dpo_line)
        sft_data = json.loads(sft_line)
        
        reward = dpo_data['reward'][0] - sft_data['reward'][0]
        rewards.append(reward)

    f_dpo.close()
    f_sft.close()
    rewards = np.array(rewards)
    # 处理nan
    if np.isnan(rewards).any():
        rewards = np.nan_to_num(rewards, nan=0)
    else:
        print("reward difference 矩阵中没有 NaN 值")

    # clip并归一化
    margin1 = rewards
    margin2 = rewards_real
    # 寻找上限x
    max_margin1 = np.max(margin1)
    x = int(max_margin1/2)
    while True:
        samples_above = np.sum((margin1 > x) & (margin1 <= max_margin1))
        if samples_above < (max_margin1 - x) or samples_above < 30 or x > max_margin1:
            break
        x += 1
    max_margin2 = np.max(margin2)
    y = int(max_margin2/1.5)
    while True:
        samples_above = np.sum((margin2 > y) & (margin2 <= max_margin2))
        if samples_above < (max_margin2 - y) or samples_above < 30 or y > max_margin2:
            break
        y += 1
    

    if strategy == 'add':
        margin = margin1 + margin2
    elif strategy == 'max':
        margin = np.maximum(margin1, margin2)
    elif strategy == 'min':
        margin = np.minimum(margin1, margin2)
    elif strategy == 'mul':
        # 将margin裁剪到[-2, x]范围内
        margin1 = np.clip(margin1, -2, x)
        margin2 = np.clip(margin2, -2, y)
        print(f"implicit reward margin clipped to range [-2, {x}]")
        print(f"skywork reward margin clipped to range [-2, {y}]")
        margin1 -= -2
        margin2 -= -2
        margin1 /= max(margin1)
        margin2 /= max(margin2)
        margin = margin1 * margin2 / ((margin1 * margin2) + (1 - margin1) * (1 - margin2))


    # 按照margin和分区选样本
    if part == 'top':
        indices = np.argsort(margin)[-Num:]
    if np.isnan(margin).any():
        nan_indices = np.where(np.isnan(margin))[0]
        print(f"NaN indices: {nan_indices}")
        

    return ds.select(indices)

if __name__ == '__main__':
    from datasets import load_from_disk
    ds = load_from_disk('dataset/llama3-ultrafeedback')['train']
    ds_tr = Sampling_ensemble('llama_uf', ds, 'mul', 'top', 100)
    print(ds_tr)