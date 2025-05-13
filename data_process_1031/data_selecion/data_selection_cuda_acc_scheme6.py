import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import cudf
import cupy as cp
from cuml.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

print('Loading data...')
total_factor = pd.read_pickle('/home/datamake117/data/haris/dataset_new/total_factor.pkl')
total_factor = total_factor[total_factor['date'] >= '2020-07-01']
total_factor['date'] = pd.to_datetime(total_factor['date'])
total_factor['Code'] = total_factor['Code'].astype('category')

def enhanced_scoring_v2(data, factor_cols, m_percent=0.1):
    """集成学习增强版评分函数"""
    # GPU数据准备
    gpu_data = cudf.from_pandas(data).fillna(0)
    n_samples = len(gpu_data)
    # 基础指标计算（保持原有逻辑）
    factor_ranks = gpu_data[factor_cols].rank(method='average').values
    label_rank = gpu_data['label'].rank(method='average').values
    factors_normalized = (factor_ranks - factor_ranks.mean(axis=0)) / (factor_ranks.std(axis=0) + 1e-8)
    labels_normalized = (label_rank - label_rank.mean()) / (label_rank.std() + 1e-8)
    ic_values = cp.abs(cp.dot(factors_normalized.T, labels_normalized) / n_samples)
    m = max(5, int(n_samples * m_percent))
    factor_tensor = torch.as_tensor(factor_ranks, device='cuda')
    label_tensor = torch.as_tensor(label_rank, device='cuda')
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
    # 目标变量：未来收益率（滚动5日）
    future_window = 5
    future_ret = gpu_data.groupby('Code')['label'].shift(-future_window).fillna(0)
    future_rank = future_ret.rank(method='average').values
    y = (factor_ranks * future_rank.reshape(-1, 1)).mean(axis=0)
    # 时间序列交叉验证训练
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cp.zeros(len(factor_cols))
    for train_idx, test_idx in tscv.split(feature_matrix):
        X_train = cp.asnumpy(feature_matrix[train_idx])
        y_train = cp.asnumpy(y[train_idx])
        X_test = cp.asnumpy(feature_matrix[test_idx])
        model = RandomForestRegressor(n_estimators=100, max_depth=5)
        model.fit(X_train, y_train)
        scores[test_idx] += cp.asarray(model.predict(X_test))
    return scores / tscv.n_splits  # 返回平均得分

def gpu_optimized_factor_selection_enhanced(total_factor, train_periods, top_n=1800, m_percent=0.2):
    """集成学习版主函数"""
    scheme_results = {}
    factor_cols = [str(i) for i in range(total_factor.shape[1] - 8)]
    for i, (start, end) in enumerate(tqdm(train_periods), 1):
        mask = (total_factor['date'] >= start) & (total_factor['date'] <= end)
        train_data = total_factor[mask]
        # 按周滚动训练（平衡数据量和时效性）
        weekly_groups = train_data.groupby(pd.Grouper(key='date', freq='W'))
        all_scores = []
        for week_idx, (week_start, weekly_data) in enumerate(tqdm(weekly_groups)):
            if len(weekly_data) < 5:  # 至少5天数据
                continue
            # 周内每日特征累积
            daily_features = []
            for _, daily_data in weekly_data.groupby('date'):
                scores = enhanced_scoring_v2(daily_data, factor_cols, m_percent)
                daily_features.append(cp.asnumpy(scores))
            # 周维度集成
            weekly_scores = np.mean(daily_features, axis=0)
            all_scores.append(weekly_scores)
        # 周期总评分
        total_scores = np.mean(all_scores, axis=0)
        # 稳定性筛选
        stable_mask = calculate_stability(all_scores)
        sorted_indices = np.argsort(total_scores * stable_mask)[::-1]
        selected = [int(factor_cols[idx]) for idx in sorted_indices[:top_n]]
        scheme_results[f'Round{i}'] = selected
    return pd.DataFrame(scheme_results)

def calculate_stability(score_matrix, threshold=0.99):
    """因子得分稳定性计算"""
    # 转换为二进制特征：是否进入当周前1%
    rank_matrix = np.argsort(np.argsort(score_matrix, axis=1), axis=1)
    stability = (rank_matrix >= (rank_matrix.shape[1] * (1 - threshold))).mean(axis=0)
    return stability

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

print('Running enhanced optimization...')
result_df = gpu_optimized_factor_selection_enhanced(total_factor, train_periods, top_n=1800, m_percent=0.2)

# 结果保存与验证
print('Saving results...')
result_df.to_feather('/home/datamake117/data/haris/dataset_new/scheme6_selected_factors.fea')
