import json
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk, concatenate_datasets

name = 'r1'

# 存储差值的列表
rewards = []

# ds = load_from_disk(f'dataset/online-uf/{name}')

if name == 'r0':
    ds0 = load_from_disk('dataset/online-uf-ins/r0/0')
    ds1 = load_from_disk('dataset/online-uf-ins/r0/1')
    ds2 = load_from_disk('dataset/online-uf-ins/r0/2')
    ds3 = load_from_disk('dataset/online-uf-ins/r0/3')
    ds = concatenate_datasets([ds0, ds1, ds2, ds3])
elif name == 'r1':
    ds0 = load_from_disk('dataset/online-uf-ins/r1/0')
    ds1 = load_from_disk('dataset/online-uf-ins/r1/1')
    ds2 = load_from_disk('dataset/online-uf-ins/r1/2')
    ds3 = load_from_disk('dataset/online-uf-ins/r1/3')
    ds = concatenate_datasets([ds0, ds1, ds2, ds3])
elif name == 'r2':
    ds0 = load_from_disk('dataset/online-uf-ins/r2/0')
    ds1 = load_from_disk('dataset/online-uf-ins/r2/1')
    ds2 = load_from_disk('dataset/online-uf-ins/r2/2')
    ds3 = load_from_disk('dataset/online-uf-ins/r2/3')
    ds = concatenate_datasets([ds0, ds1, ds2, ds3])


for i in range(len(ds)):
    reward_data = ds[i]['all_rm_scores']
    reward_diff = max(reward_data) - min(reward_data)
    rewards.append(reward_diff)

rewards = np.array(rewards)

# 创建直方图
plt.figure(figsize=(10, 6))
plt.hist(rewards, bins=100)
plt.title('Distribution of online rewards margin')
plt.xlabel('reward margin')
plt.ylabel('Frequency')

# 添加网格线
plt.grid(True, alpha=0.3)

# 保存图片
plt.savefig(f'./adpo/metric/ins_{name}_skywork_reward_distribution-.png')
plt.close()

print(f"Number of samples: {len(rewards)}")