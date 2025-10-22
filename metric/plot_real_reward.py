import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 设置plt样式
plt.style.use('classic')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

name = 'llama_uf'

# 存储差值的列表
rewards = []

f_reward = open(f'./adpo/metric/{name}_rm_scores.jsonl', 'r')


for line in f_reward:
    reward_data = json.loads(line)
    reward_diff = reward_data['reward_diff']
    rewards.append(reward_diff)

f_reward.close()

rewards = np.array(rewards)

# 将rewards中的nan替换为0
rewards = np.nan_to_num(rewards, nan=0.0)

# indexs = np.random.choice(len(rewards), 3000)
# rewards = rewards[indexs]

# 创建图形
fig, ax = plt.subplots(figsize=(5, 4))

# 绘制直方图
n, bins, patches = ax.hist(rewards, bins=100, density=True, alpha=0.7, 
                          color='#2E86C1', edgecolor='black', linewidth=0.3)

# 添加核密度估计曲线
kde = stats.gaussian_kde(rewards)
x_range = np.linspace(min(rewards), max(rewards), 200)
ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')

# 添加均值和中位数的垂直线
# mean_val = np.mean(rewards)
# median_val = np.median(rewards)
# ax.axvline(mean_val, color='green', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
# ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.3f}')

# 设置标题和标签
ax.set_title('Distribution of External Reward Margin', pad=15, fontsize=10, fontweight='bold')
ax.set_xlabel('External Reward Margin', labelpad=9)
ax.set_ylabel('Density', labelpad=9)

# 添加网格线
ax.grid(True, alpha=0.3, linestyle='--')

# 添加图例
ax.legend(frameon=True, fancybox=True, shadow=True)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig(f'./adpo/metric/figure/{name}_skywork_reward_distribution.pdf', 
            bbox_inches='tight')
plt.close()



# import json
# import matplotlib.pyplot as plt
# import numpy as np

# name = 'llama_uf'

# # 存储差值的列表
# rewards = []

# f_reward = open(f'./adpo/metric/{name}_rm_scores.jsonl', 'r')


# for line in f_reward:
#     reward_data = json.loads(line)
#     reward_diff = reward_data['reward_diff']
#     rewards.append(reward_diff)

# f_reward.close()

# rewards = np.array(rewards)

# # 创建直方图
# plt.figure(figsize=(10, 6))
# plt.hist(rewards, bins=100)
# plt.title('Distribution of reward margin')
# plt.xlabel('reward margin')
# plt.ylabel('Frequency')

# # 添加网格线
# plt.grid(True, alpha=0.3)

# # 保存图片
# plt.savefig(f'./adpo/metric/{name}_skywork_reward_distribution.png')
# plt.close()

# print(f"Number of samples: {len(rewards)}")