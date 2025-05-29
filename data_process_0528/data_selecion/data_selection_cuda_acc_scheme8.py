import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import cudf
import cupy as cp

print('Loading data...')
total_factor = pd.read_pickle('/home/datamake117/data/haris/dataset_new/total_factor.pkl')
total_factor = total_factor[total_factor['date'] >= '2020-07-01']
total_factor['date'] = pd.to_datetime(total_factor['date'])
total_factor['Code'] = total_factor['Code'].astype('category')

def enhanced_scoring(data, factor_cols, m_percent=0.1):
    """增强版多维度因子评分函数"""
    # 转换到GPU
    gpu_data = cudf.from_pandas(data).fillna(0)
    n_samples = len(gpu_data)
    # 基础指标计算
    factor_ranks = gpu_data[factor_cols].rank(method='average').values
    label_rank = gpu_data['label'].rank(method='average').values
    # 1. 改进的IC计算（考虑波动率）
    factors_normalized = (factor_ranks - factor_ranks.mean(axis=0)) / (factor_ranks.std(axis=0) + 1e-8)
    labels_normalized = (label_rank - label_rank.mean()) / (label_rank.std() + 1e-8)
    ic_values = cp.abs(cp.dot(factors_normalized.T, labels_normalized) / n_samples)
    # 2. 动态多空头计算
    m = max(5, int(n_samples * m_percent))
    factor_tensor = torch.as_tensor(factor_ranks, device='cuda')
    label_tensor = torch.as_tensor(label_rank, device='cuda')
    # 多头增强指标
    top_mask = factor_tensor >= (n_samples - m)
    long_ratio = (top_mask & (label_tensor.view(-1, 1) >= (n_samples - m))).sum(dim=0).float() / m
    long_ret = (label_tensor.view(-1, 1) * top_mask).sum(dim=0) / m
    # 空头增强指标
    bottom_mask = factor_tensor <= m
    short_ratio = (bottom_mask & (label_tensor.view(-1, 1) <= m)).sum(dim=0).float() / m
    short_ret = (label_tensor.view(-1, 1) * bottom_mask).sum(dim=0) / m
    # 3. 换手率指标（GPU加速）
    shifted_ranks = cp.roll(factor_ranks, shift=1, axis=0)
    turnover = cp.abs(factor_ranks - shifted_ranks).mean(axis=0) / n_samples
    # 4. 收益差异指标
    ret_diff = (long_ret - short_ret).cpu().numpy()
    # 5. IC稳定性指标（滑动窗口波动率）
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
    # 6. 因子衰减率（使用未来5日收益）
    future_ret_5d = cp.asarray(gpu_data.groupby('Code')['label'].shift(-5).fillna(0).values)
    ic_5d = cp.abs(cp.dot(factors_normalized.T, future_ret_5d) / n_samples)
    decay_rate = (ic_values - ic_5d) / (ic_values + 1e-8)
    # 7. 极端值鲁棒性（MAD方法）
    median = cp.median(factor_ranks, axis=0)
    mad = cp.median(cp.abs(factor_ranks - median), axis=0)
    outlier_robust = 1 / (mad + 1e-8)
    # 8. 动量持续性（5日排名变化）
    if n_samples > 10:  # 需要有足够数据
        rank_change = cp.mean(factor_ranks[5:] > factor_ranks[:-5], axis=0)
    else:
        rank_change = cp.zeros(len(factor_cols))
    # 9. 空头端区分度（后20%股票中的区分能力）
    bottom_20 = label_rank <= cp.quantile(label_rank, 0.2)
    short_discrim = cp.abs(cp.mean(factors_normalized[bottom_20], axis=0) - cp.mean(factors_normalized[~bottom_20], axis=0))
    # 10. 信息半衰期（IC自相关）
    if window_size > 10:
        ic_series = cp.dot(factors_normalized[-window_size:].T, labels_normalized[-window_size:])
        autocorr = cp.corrcoef(ic_series[:-5], ic_series[5:])[0, 1]
        info_half_life = 5 / (autocorr + 1e-8)
    else:
        info_half_life = cp.ones(len(factor_cols)) * 5
    # 权重分配
    weights = {
        'ic': 0.25, 
        'ic_vol': -0.15,
        'long_ratio': 0.1,
        'short_ratio': 0.1,
        'turnover': -0.08,
        'ret_diff': 0.2,
        'decay_rate': -0.12,    # 衰减越快惩罚越大
        'outlier_robust': 0.15, # 抗极端值能力
        'rank_change': 0.1,     # 动量持续性
        'short_discrim': 0.15,  # 空头区分能力
        'info_half_life': 0.05  # 信息持续性
    }
    
    # 标准化各指标
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    total_scores = (
        weights['ic'] * normalize(cp.asnumpy(ic_values)) +
        weights['ic_vol'] * normalize(cp.asnumpy(ic_vol)) + 
        weights['long_ratio'] * normalize(cp.asnumpy(long_ratio)) +
        weights['short_ratio'] * normalize(cp.asnumpy(short_ratio)) +
        weights['turnover'] * normalize(cp.asnumpy(turnover)) +
        weights['ret_diff'] * normalize(cp.asnumpy(ret_diff)) +
        weights['decay_rate'] * normalize(cp.asnumpy(decay_rate)) +
        weights['outlier_robust'] * normalize(cp.asnumpy(outlier_robust)) +
        weights['rank_change'] * normalize(cp.asnumpy(rank_change)) +
        weights['short_discrim'] * normalize(cp.asnumpy(short_discrim)) +
        weights['info_half_life'] * normalize(cp.asnumpy(info_half_life))
    )
    return total_scores

def gpu_optimized_factor_selection(total_factor, train_periods, top_n, m_percent=0.2):
    """改进版因子筛选主函数"""
    scheme_results = {}
    factor_cols = [str(i) for i in range(total_factor.shape[1] - 8)]
    for i, (start, end) in enumerate(tqdm(train_periods), 1):
        mask = (total_factor['date'] >= start) & (total_factor['date'] <= end)
        train_data = total_factor[mask]
        daily_scores = []
        for _, daily_group in tqdm(train_data.groupby('date'), desc=f'Processing Round {i}'):
            if len(daily_group) < 10:
                continue
            scores = enhanced_scoring(daily_group, factor_cols, m_percent)
            daily_scores.append(scores)
        total_scores = np.mean(daily_scores, axis=0)
        sorted_indices = np.argsort(total_scores)[::-1]
        selected = [int(factor_cols[idx]) for idx in sorted_indices[:top_n]]
        scheme_results[f'Round{i}'] = selected
    return pd.DataFrame(scheme_results)

# 完整训练周期定义
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

print('Running optimization algorithm...')
# 运行优化算法
result_df = gpu_optimized_factor_selection(total_factor, train_periods, top_n=1800, m_percent=0.2)

print('Saving results...')
# 保存结果
result_df.to_feather('/home/datamake117/data/haris/dataset_new/scheme8_selected_factors.fea')
