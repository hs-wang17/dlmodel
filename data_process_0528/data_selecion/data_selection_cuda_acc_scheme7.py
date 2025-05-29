import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
import cudf
import cupy as cp
import collections
import random
import gc

# 指定使用的GPU设备
DEVICE = 'cuda:1'
torch.cuda.set_device(DEVICE)

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 数据加载 (保持原有路径)
print('Loading data...')
total_factor = pd.read_pickle('/home/datamake117/data/haris/dataset_new/total_factor.pkl')
total_factor = total_factor[total_factor['date'] >= '2020-07-01']
total_factor['date'] = pd.to_datetime(total_factor['date'])
total_factor['Code'] = total_factor['Code'].astype('category')


def clear_gpu_memory():  # 释放GPU内存
    torch.cuda.empty_cache()                                # PyTorch缓存
    if 'cp' in globals():                                   # CuPy缓存
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    if 'cudf' in globals():                                 # cuDF缓存
        gc.collect()


class ReplayBuffer:  # 经验回放缓冲区
    def __init__(self, capacity=1000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward):
        self.buffer.append((state.cpu(), action.cpu(), reward))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # 将数据转移到指定GPU
        return (
            torch.stack([item[0] for item in batch]).to(DEVICE),
            torch.stack([item[1] for item in batch]).to(DEVICE),
            torch.tensor([item[2] for item in batch]).to(DEVICE)
        )

    def __len__(self):
        return len(self.buffer)


class FactorSelectionAgent(nn.Module):  # 因子选择智能体
    def __init__(self, input_dim=6, hidden_dim=4):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.to(DEVICE)

    def forward(self, x):
        encoded = self.feature_encoder(x)
        action_probs = torch.sigmoid(self.actor_head(encoded)).squeeze()
        state_value = self.value_head(encoded).squeeze()
        return action_probs, state_value


class RLFactorSelector:  # 强化学习因子选择器
    def __init__(self, device=DEVICE):
        self.device = device
        self.model = FactorSelectionAgent().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-5)
        self.gamma = 0.95           # 奖励折扣因子
        self.entropy_coef = 0.1     # 熵正则项

    def select_factors(self, feature_matrix, top_n=1800):
        """执行因子选择"""
        with torch.no_grad():
            self.model.eval()
            probs, _ = self.model(feature_matrix.to(self.device))
            selected = torch.topk(probs, min(top_n, len(probs))).indices
        return selected.cpu().numpy()

    def update_policy(self, states, actions, rewards):
        """策略优化核心"""
        self.model.train()
        states, actions, rewards = states.to(self.device), actions.to(self.device), rewards.to(self.device)
        action_probs, state_values = self.model(states)
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, device=self.device)
        advantages = returns - state_values.detach()
        probs = action_probs[actions]
        policy_loss = -(torch.log(probs + 1e-8) * advantages).mean()                                # Loss 1：策略梯度损失
        value_loss = nn.MSELoss()(state_values, returns)                                            # Loss 2：价值函数损失
        entropy_loss = self.entropy_coef * (action_probs * torch.log(action_probs + 1e-8)).mean()   # Loss 3：熵正则化
        total_loss = policy_loss + value_loss + entropy_loss                                        # 总损失
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return total_loss.item()


def rl_enhanced_scoring_v2(data, factor_cols, m_percent=0.1):  # 强化学习增强评分函数（状态）
    # GPU 数据预处理
    gpu_data = cudf.from_pandas(data).fillna(0)
    n_samples = len(gpu_data)
    # 原版核心特征计算
    factor_ranks = gpu_data[factor_cols].rank(method='average').values
    label_rank = gpu_data['label'].rank(method='average').values
    factors_normalized = (factor_ranks - factor_ranks.mean(axis=0)) / (factor_ranks.std(axis=0) + 1e-8)
    labels_normalized = (label_rank - label_rank.mean()) / (label_rank.std() + 1e-8)
    ic_values = cp.abs(cp.dot(factors_normalized.T, labels_normalized) / n_samples)
    m = max(5, int(n_samples * m_percent))
    factor_tensor = torch.as_tensor(factor_ranks, device=DEVICE)
    label_tensor = torch.as_tensor(label_rank, device=DEVICE)
    top_mask = factor_tensor >= (n_samples - m)
    long_ratio = (top_mask & (label_tensor.view(-1, 1) >= (n_samples - m))).sum(dim=0).float() / m
    long_ret = (label_tensor.view(-1, 1) * top_mask).sum(dim=0) / m
    bottom_mask = factor_tensor <= m
    short_ratio = (bottom_mask & (label_tensor.view(-1, 1) <= m)).sum(dim=0).float() / m
    short_ret = (label_tensor.view(-1, 1) * bottom_mask).sum(dim=0) / m
    shifted_ranks = cp.roll(factor_ranks, shift=1, axis=0)
    turnover = cp.abs(factor_ranks - shifted_ranks).mean(axis=0) / n_samples
    ret_diff = (long_ret - short_ret).cpu().numpy()
    window_size = min(20, n_samples)
    if window_size > 5:
        input_matrix = factors_normalized * labels_normalized.reshape(-1, 1)
        rolling_view = cp.lib.stride_tricks.sliding_window_view(
            input_matrix,
            window_shape=(window_size,),  # 单维度窗口
            axis=0  # 仅在时间维度滑动
        )
        # 计算每个窗口内的标准差（形状：n_stocks × n_factors × n_windows）
        window_std = rolling_view.std(axis=2)
        # 取所有窗口的标准差平均值（形状：n_factors × 1）
        ic_vol = window_std.mean(axis=0)
    else:
        ic_vol = cp.zeros(factors_normalized.shape[1], dtype=cp.float32)
    # 构建特征矩阵（每个因子的多维特征）
    feature_matrix = cp.zeros((len(factor_cols), 6), dtype=cp.float32)
    feature_matrix[:, 0] = ic_values.astype(cp.float32)                                 # 特征1: IC值
    feature_matrix[:, 1] = ic_vol.astype(cp.float32)                                    # 特征2: IC波动率
    feature_matrix[:, 2] = cp.asarray(long_ratio.cpu().numpy(), dtype=cp.float32)       # 特征3: 多头捕获率
    feature_matrix[:, 3] = cp.asarray(short_ratio.cpu().numpy(), dtype=cp.float32)      # 特征4: 空头捕获率
    feature_matrix[:, 4] = turnover.astype(cp.float32)                                  # 特征5: 换手率
    feature_matrix[:, 5] = cp.asarray(ret_diff, dtype=cp.float32)                       # 特征6: 多空差异
    return torch.tensor(cp.asnumpy(feature_matrix), dtype=torch.float32)


def rl_gpu_factor_selection(total_factor, train_periods, top_n=1800):  # 强化学习版主优化流程
    factor_cols = [str(i) for i in range(total_factor.shape[1] - 8)]
    selector = RLFactorSelector()
    scheme_results = {}
    for period_idx, (start, end) in enumerate(tqdm(train_periods, desc='Total Progress'), 1):
        buffer = ReplayBuffer(capacity=1000)
        mask = (total_factor['date'] >= start) & (total_factor['date'] <= end)
        train_data = total_factor.loc[mask]
        weekly_groups = list(train_data.groupby(pd.Grouper(key='date', freq='W')))  # 周滚动训练窗口
        for week_idx, (_, weekly_data) in tqdm(enumerate(weekly_groups), total=len(weekly_groups), desc='Weekly Training'):
            if len(weekly_data) < 5:
                continue
            for _, daily_data in weekly_data.groupby('date'):
                states = rl_enhanced_scoring_v2(daily_data, factor_cols)    # 生成状态特征
                selected = selector.select_factors(states, top_n=top_n)     # 因子选择
                future_ret = daily_data.groupby('Code', observed=True)['label'].shift(-5).fillna(0).values
                sharpe = np.mean(future_ret[selected]) / (np.std(future_ret[selected]) + 1e-8)  # 计算即时奖励（未来5日夏普比率）
                buffer.push(states, torch.tensor(selected), sharpe.item())  # 存储经验
            if len(buffer) > 500:  # 经验回放更新
                batch_states, batch_actions, batch_rewards = zip(*buffer.sample(64))
                selector.update_policy(batch_states, batch_actions, batch_rewards)
                del batch_states, batch_actions, batch_rewards  # 内存清理
                torch.cuda.empty_cache()
                cp.get_default_memory_pool().free_all_blocks()
        # full_states = rl_enhanced_scoring_v2(train_data, factor_cols)
        batch_size = 1024  # 根据显存调整
        gpu_data = cudf.from_pandas(train_data)
        chunks = [gpu_data.iloc[i:i+batch_size] for i in range(0, len(train_data), batch_size)]
        # 使用cupy加速矩阵运算
        state_list = []
        for chunk in tqdm(chunks):
            states = rl_enhanced_scoring_v2(chunk.to_pandas(), factor_cols)  # 暂时保持原有函数
            state_list.append(cp.asarray(states))
        # 使用cupy的concatenate替代torch.stack
        full_states = torch.from_numpy(cp.mean(cp.stack(state_list), axis=0).get())
        final_selected = selector.select_factors(full_states, top_n=top_n)
        scheme_results[f'Round{period_idx}'] = [factor_cols[i] for i in final_selected]  # 最终因子选择
        del full_states, train_data  # 释放周期数据
        clear_gpu_memory()
    return pd.DataFrame(scheme_results)


# 训练周期设置
train_periods = [
    ('2020-07-01', '2022-07-01'),  # Round 1
    ('2021-01-01', '2023-01-01'),  # Round 2
    ('2021-04-01', '2023-04-01'),  # Round 3
    ('2021-07-01', '2023-07-01'),  # Round 4
    ('2021-10-01', '2023-10-01'),  # Round 5
    ('2022-01-01', '2024-01-01'),  # Round 6
    ('2022-04-01', '2024-04-01'),  # Round 7
    ('2022-07-01', '2024-07-01'),  # Round 8
    ('2022-10-01', '2024-10-01')   # Round 9
]

# 执行优化
print('Running RL-enhanced factor selection...')
result_df = rl_gpu_factor_selection(total_factor, train_periods)

# 保存结果
print('Saving results...')
result_df.to_feather('/home/datamake117/data/haris/dataset_new/scheme7_selected_factors.fea')
