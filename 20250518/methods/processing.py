import numpy as np


# 根据流动性调整收益率前7%-10%附近的训练标签，给的默认参数应该就是之前测试的效果比较好的
def adjust_daily_returns(returns, liquidity, threshold=0.01, lower_percentile=0.86, upper_percentile=0.93):
    """
    思路：收益率接近的一组股票按照流动性从高到低重新分配调整后的收益率。
    根据流动性调整收益率，但仅针对真实收益率处于指定分位数范围内的部分。
    :param returns: 单天的原始收益率标签 (1D NumPy 数组)，可能含 NaN。
    :param liquidity: 单天的流动性指标 (1D NumPy 数组)，可能含 NaN。
    :param threshold: 收益率分组的阈值 (同组内收益率差值最大为 threshold)。
    :param lower_percentile: 分位数的下界 (默认 90%)。
    :param upper_percentile: 分位数的上界 (默认 93%)。
    :return: 调整后的收益率标签 (1D NumPy 数组)，NaN 保留在原位。
    """
    # Step 0: 筛选非 NaN 数据
    valid_mask = ~np.isnan(returns) & ~np.isnan(liquidity)  # 同时非 NaN
    valid_returns = returns[valid_mask]
    valid_liquidity = liquidity[valid_mask]
    if len(valid_returns) == 0:  # 如果有效数据不足，直接返回原数组（保持原始 NaN 结构）
        return returns

    # Step 1: 计算分位数范围
    lower_bound = np.percentile(valid_returns, lower_percentile * 100)
    upper_bound = np.percentile(valid_returns, upper_percentile * 100)

    # Step 2: 筛选处于分位数范围内的收益率
    in_range_mask = (valid_returns >= lower_bound) & (valid_returns <= upper_bound)
    if not np.any(in_range_mask):  # 如果分位数范围内没有数据，则直接返回原数组
        return returns
    in_range_indices = np.where(in_range_mask)[0]
    in_range_returns = valid_returns[in_range_mask]
    in_range_liquidity = valid_liquidity[in_range_mask]

    # Step 3: 按收益率排序并分组
    sorted_indices = np.argsort(in_range_returns)
    sorted_returns = in_range_returns[sorted_indices]
    sorted_liquidity = in_range_liquidity[sorted_indices]
    diff = np.diff(sorted_returns)
    group_indices = np.where(diff > threshold)[0] + 1  # 找到分组边界
    groups = np.split(np.arange(len(sorted_returns)), group_indices)  # 分组索引

    # Step 4: 调整每组的收益率（每组内在[r_min, r_max]区间内线性插值，按照流动性大小排序）
    adjusted_returns = np.zeros_like(in_range_returns)
    for group in groups:
        group_returns = sorted_returns[group]
        group_liquidity = sorted_liquidity[group]
        mu = group_returns.mean()
        sigma = group_returns.std()
        if sigma == 0:  # 避免除0
            sigma = 1e-6
        liquidity_sorted_indices = np.argsort(-group_liquidity)  # 按流动性降序排序
        liquidity_scores = group_liquidity[liquidity_sorted_indices]  # 构造 softmax 权重
        weights = np.exp(liquidity_scores)
        weights /= np.sum(weights)
        # 将 softmax 权重映射到 z 分数（标准正态排序值）
        from scipy.stats import norm
        ranks = np.cumsum(weights) - 0.5 * weights  # 类似 rank 的值
        z_scores = norm.ppf(ranks)  # 转换为标准正态分布值
        z_scores = (z_scores - np.min(z_scores)) / (np.max(z_scores) - np.min(z_scores))  # 归一化到 [0,1]
        adjusted_group_returns = mu + sigma * z_scores  # 非线性调整后的收益率
        adjusted_returns[group[liquidity_sorted_indices]] = adjusted_group_returns

    # Step 5: 将调整后的部分还原到原始索引中
    final_adjusted_returns = returns.copy()
    final_adjusted_returns[valid_mask] = valid_returns  # 初始化为原值 
    final_adjusted_returns[valid_mask][in_range_indices] = adjusted_returns  # 覆盖调整部分

    return final_adjusted_returns
