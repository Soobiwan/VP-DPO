import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datasets import load_from_disk
name = 'llama_uf_armo'

# 读取数据的函数
def load_rewards(filename):
    rewards = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            rewards.append(data['reward_diff'] if 'reward_diff' in data 
                         else data['reward'][0])
    return np.array(rewards)

# 加载数据
rewards_skywork = load_rewards(f'./adpo/metric/{name}_rm_scores.jsonl')

# 加载数据集里面自带的reward
ds = load_from_disk('dataset/llama3-ultrafeedback-armo')['train']
rewards = np.zeros(len(ds))
for i in range(len(ds)):
    scores = ds[i]['all_rm_scores']
    rewards[i] = max(scores) - min(scores)

# 处理NaN值
rewards_skywork = np.nan_to_num(rewards_skywork, nan=0.5)
rewards = np.nan_to_num(rewards, nan=0)

# 下采样, 减少点数，增强视觉效果
rewards_skywork = rewards_skywork[::10]
rewards = rewards[::10]

# 设置图表样式
plt.style.use('classic')
plt.figure(figsize=(6, 4))

# 创建散点图
scatter = plt.scatter(rewards_skywork, rewards, 
                     alpha=0.5,  # 透明度
                     s=20,      # 点的大小
                     c=np.abs(rewards_skywork - rewards),  # 颜色映射基于差异
                     cmap='viridis')

# 添加颜色条
plt.colorbar(scatter, label='Absolute Difference')

# 添加趋势线
z = np.polyfit(rewards_skywork, rewards, 1)
p = np.poly1d(z)
plt.plot(rewards_skywork, p(rewards_skywork), "r--", alpha=0.8, label=f'Trend line')

# 添加标签和标题
plt.xlabel('Skywork Reward Margin', fontsize=17)
plt.ylabel('Armorm Reward Margin', fontsize=17)
# plt.title('External vs DPO Implicit Reward Margin Distribution', fontsize=16)

# 添加网格线
plt.grid(True, alpha=0.3, linestyle='--')

# 添加图例
plt.legend()

# 设置坐标轴范围（可选）
plt.xlim(-40, 60)
plt.ylim(0, 0.2)

# 添加零线
plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)
plt.axvline(x=0, color='k', linestyle=':', alpha=0.3)

# 保存图片
plt.tight_layout()  # 自动调整布局
plt.savefig(f'./adpo/metric/figure/{name}_DM_distribution.pdf', bbox_inches='tight')
plt.close()
