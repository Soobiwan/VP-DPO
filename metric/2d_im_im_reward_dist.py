import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

name = 'hh'

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
s_dpo_rewards = load_rewards(f'./adpo/metric/1b-result/dpo_5k_{name}_ppl.jsonl')
s_sft_rewards = load_rewards(f'./adpo/metric/1b-result/sft_{name}_ppl.jsonl')
s_rewards = s_dpo_rewards - s_sft_rewards

# 加载DPO和SFT的差值
dpo_rewards = load_rewards(f'./adpo/metric/dpo_2k_{name}_ppl.jsonl')
sft_rewards = load_rewards(f'./adpo/metric/sft_{name}_ppl.jsonl')
rewards = dpo_rewards - sft_rewards

# 处理NaN值
s_rewards = np.nan_to_num(s_rewards, nan=0)
rewards = np.nan_to_num(rewards, nan=0)

# 下采样, 减少点数，增强视觉效果
s_rewards = s_rewards[::20]
rewards = rewards[::20]

# 设置图表样式
plt.style.use('classic')
plt.figure(figsize=(6, 4))

# 创建散点图
scatter = plt.scatter(s_rewards, rewards, 
                     alpha=0.5,  # 透明度
                     s=20,      # 点的大小
                     c=np.abs(s_rewards - rewards),  # 颜色映射基于差异
                     cmap='viridis')

# 添加颜色条
plt.colorbar(scatter, label='Absolute Difference')

# 添加趋势线
z = np.polyfit(s_rewards, rewards, 1)
p = np.poly1d(z)
plt.plot(s_rewards, p(s_rewards), "r--", alpha=0.8, label=f'Trend line')

# 添加标签和标题
plt.xlabel('1B Implicit Reward Margin', fontsize=17)
plt.ylabel('3B Implicit Reward Margin', fontsize=17)
# plt.title('1B vs 3B DPO Implicit Reward Margin Distribution', fontsize=16)

# 添加网格线
plt.grid(True, alpha=0.3, linestyle='--')

# 添加图例
plt.legend()

# 设置坐标轴范围（可选）
plt.xlim(-40, 60)
plt.ylim(-40, 60)

# 添加零线
plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)
plt.axvline(x=0, color='k', linestyle=':', alpha=0.3)

# 保存图片
plt.tight_layout()  # 自动调整布局
plt.savefig(f'./adpo/metric/figure/{name}_DIM_distribution.png', bbox_inches='tight')
plt.close()
