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

def spearman_corr(factors: cp.ndarray, labels: cp.ndarray) -> cp.ndarray:
    """
    用 CuPy 手动计算 Spearman 秩相关系数的绝对值
    输入:
        factors: (n_samples, n_factors) 的因子排名矩阵
        labels: (n_samples,) 的标签排名向量
    返回:
        (n_factors,) 的绝对值相关系数数组
    """
    factors = (factors - factors.mean(axis=0)) / (factors.std(axis=0) + 1e-8)
    labels = (labels - labels.mean()) / (labels.std() + 1e-8)
    corr = cp.dot(factors.T, labels) / len(labels)
    return cp.abs(corr).astype(cp.float32)  # 关键修改：添加 cp.abs() 函数

def gpu_calculate_factor_scores(data, factor_cols, m_percent=0.1):
    """GPU加速的因子计算"""
    # 转换到GPU
    gpu_data = cudf.from_pandas(data)
    # 预计算所有因子的排名矩阵
    gpu_data = gpu_data.fillna(0)
    factor_ranks = gpu_data[factor_cols].rank(method='average').values
    label_rank = gpu_data['label'].rank(method='average').values
    # 批量计算Spearman相关系数
    ic_scores = spearman_corr(factor_ranks, label_rank)
    # 计算多空头占比
    m = int(len(gpu_data) * m_percent)
    # 转换到PyTorch Tensor以利用GPU加速
    factor_tensor = torch.as_tensor(factor_ranks, device='cuda')
    label_tensor = torch.as_tensor(label_rank, device='cuda')
    # 多头占比计算
    label_tensor_expanded = label_tensor.view(-1, 1).expand_as(factor_tensor)
    long_ratio = torch.sum((factor_tensor >= (len(gpu_data) - m)) & (label_tensor_expanded >= (len(gpu_data) - m)), dim=0).float() / m
    # 空头占比计算
    short_ratio = torch.sum((factor_tensor <= m) & (label_tensor_expanded <= m), dim=0).float() / m
    # 综合得分
    total_scores = (cp.asnumpy(ic_scores) + long_ratio.cpu().numpy() + short_ratio.cpu().numpy()) / 3
    return total_scores

def gpu_optimized_factor_selection(total_factor, train_periods, top_n, m_percent=0.2):
    """GPU加速的因子筛选主函数"""
    # 结果存储
    scheme_results = {}
    # 因子列表
    factor_cols = [str(i) for i in range(total_factor.shape[1] - 8)]
    for i, (start, end) in enumerate(tqdm(train_periods), 1):
        # 1. 数据准备
        mask = (total_factor['date'] >= start) & (total_factor['date'] <= end)
        train_data = total_factor[mask]
        # 2. 按日期分批处理
        daily_scores = []
        for _, daily_group in tqdm(train_data.groupby('date')):
            if len(daily_group) < 10:
                continue
            # 批量计算当日所有因子得分
            scores = gpu_calculate_factor_scores(daily_group, factor_cols, m_percent)
            daily_scores.append(scores)
        # 3. 汇总得分
        total_scores = np.mean(daily_scores, axis=0)
        # 4. 选取Top N
        top_indices = np.argsort(total_scores)[::-1][:top_n]
        selected = [int(factor_cols[i]) for i in top_indices]
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
result_df.to_feather('/home/datamake117/data/haris/dataset_new/scheme2_selected_factors.fea')
