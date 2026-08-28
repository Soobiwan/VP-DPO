import json
# import matplotlib.pyplot as plt
import numpy as np


def Sampling(name, ds, strategy, part, Num):
    if strategy == 'random':
        if Num == -1:
            return ds.shuffle(seed=42)
        else:
            return ds.shuffle(seed=42).select(range(Num))
    if strategy == 'in_reward_margin':
        # 利用数据集本身的reward model给出的打分算margin
        score_list = np.zeros(len(ds))
        for i in range(len(ds)):
            scores = ds[i]['all_rm_scores']
            score_list[i] = max(scores) - min(scores)
        margin1 = score_list * 150

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

        margin2 = rewards_real
        margin = margin1 + margin2


    elif strategy == 'ex_reward_margin':
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

        margin = rewards_real
        #设置边缘值
        mid_edge = 1
        
    elif strategy == 'im_reward_margin':
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

        margin = rewards
        #设置边缘值
        mid_edge = 1
    elif strategy == 'ppl_margin':
        # 存储差值的列表
        differences = []

        idx = 1
        # 读取 jsonl 文件
        with open(f'./adpo/metric/sft_{name}_ppl.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                diff = data['ppl_chosen'][1]/data['ppl_chosen'][0] - data['ppl_rejected'][1]/data['ppl_rejected'][0]
                # 处理这些异常样本不要被采样到
                if diff < -2:
                    diff = -0.25
                elif diff > 2:
                    diff = 0.25
                differences.append(diff)

        differences = np.array(differences)
        # 处理nan
        if np.isnan(differences).any():
            differences = np.nan_to_num(differences, nan=0.3)
        else:
            print("ppl difference 矩阵中没有 NaN 值")

        margin = differences
        #设置边缘值
        mid_edge = 0.1

    # 按照margin和分区选样本
    if part == 'top':
        indices = np.argsort(margin)[-Num:]
    elif part == 'mid':
        middle_mask = (margin >= -mid_edge) & (margin <= mid_edge)
        indices = np.where(middle_mask)[0]

        # 如果[-mid_edge,mid_edge]区间内的样本超过Num个，随机采样Num个
        if len(indices) > Num:
            indices = np.random.choice(indices, size=Num, replace=False)
        elif len(indices) < Num:
            print(f"Warning: Only {len(indices)} samples found in middle range")
    elif part == 'bot':
        indices = np.argsort(margin)[:Num]
    else:
        print("Invalid part")
        return ds

    return ds.select(indices)
