import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

name = 'uf'

# 存储差值的列表
differences = []

idx = 1
# 读取 jsonl 文件
with open(f'./adpo/metric/dpo_2k_{name}_ppl.jsonl', 'r') as file:
    for line in file:
        data = json.loads(line)
        diff = data['ppl_chosen'][1]/data['ppl_chosen'][0] - data['ppl_rejected'][1]/data['ppl_rejected'][0]
        if diff < -2:
            diff = -2
        elif diff > 2:
            diff = 2
        differences.append(diff)

differences = np.array(differences)

# 存储差值的列表
rewards = []

f_dpo = open(f'./adpo/metric/dpo_2k_{name}_ppl.jsonl', 'r')
f_sft = open(f'./adpo/metric/sft_{name}_ppl.jsonl', 'r')

for dpo_line, sft_line in zip(f_dpo, f_sft):
    dpo_data = json.loads(dpo_line)
    sft_data = json.loads(sft_line)
    
    reward = dpo_data['reward'][0] - sft_data['reward'][0]
    # if reward < -10:
    #     reward = -10
    # elif reward > 10:
    #     reward = 10
    rewards.append(reward)

f_dpo.close()
f_sft.close()

rewards = np.array(rewards)

# 检查 differences 矩阵
if np.isnan(differences).any():
    print("differences 矩阵中存在 NaN 值")
    nan_indices = np.where(np.isnan(differences))
    print(f"differences 矩阵中 NaN 的索引: {nan_indices}")
    differences = np.nan_to_num(differences, nan=0.5)
else:
    print("differences 矩阵中没有 NaN 值")

# 检查 rewards 矩阵
if np.isnan(rewards).any():
    print("rewards 矩阵中存在 NaN 值")
    nan_indices = np.where(np.isnan(rewards))
    print(f"rewards 矩阵中 NaN 的索引: {nan_indices}")
    rewards = np.nan_to_num(rewards, nan=0)
else:
    print("rewards 矩阵中没有 NaN 值")

# Create 2D histogram plot
plt.figure(figsize=(10, 8))
plt.hist2d(differences, rewards, bins=100, cmap='viridis', norm='log')
plt.colorbar(label='Count (log scale)')

# Add labels and title
plt.xlabel('PPL Difference')
plt.ylabel('Reward Difference')
plt.title('2D Distribution of PPL Difference vs Reward Difference')

# 添加网格线
plt.grid(True, alpha=0.3)

# 保存图片
plt.savefig(f'./adpo/metric/{name}_ppl_reward_distribution.png')
plt.close()