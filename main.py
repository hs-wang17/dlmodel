import os
import random
import gc
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from tqdm.notebook import tqdm, trange
import torch
import torch.multiprocessing as mp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from methods.model import *
from methods.logger import *
from methods.processing import *
from methods.train import *


def main(
    round_num, dt1, dt2, dt3, dt4, dt5,
    correlation_df, grouped, grouped_label, grouped_liquidity,
    total_date_list, main_folder_name, 
    pid_num=5243, factor_num=2221, corr_thres=0.9, seed_num=5, model_mode=False, multi_model=6
    ):
    '''
    TODO: 参数含义
    para round_num: 轮数（周期序号）
    para dt1: 训练集开始时间
    para dt2: 验证集开始时间
    para dt3: 验证集结束时间
    para dt4: 测试集开始时间
    para dt5: 测试集结束时间
    
    dt1 ------训练集------ dt2 ------验证集------ dt3/dt4 ------测试集------ dt5
    
    para correlation_df: 因子筛选辅助数据
    para grouped: 按日期分组的因子数据
    para grouped_label: 按日期分组的标签数据
    para grouped_liquidity: 按日期分组的流动性数据
    para total_date_list: 全部日期
    para main_folder_name: 主文件夹名称
    para pid_num: 股票数量
    para factor_num: 因子数量
    para corr_thres: 因子筛选相关系数阈值
    para seed_num: 每个模型的种子数
    para model_mode: 是否继续训练
    para multi_model: 模型数量
    '''
    seed_list = []
    for i in range(seed_num):
        random.seed(i)
        seed_list.append(list(random.sample(range(100), multi_model)))
    total_train_num = len(seed_list)  # seed_num * multi_model
    total_test_output = []
    total_test_name = 'test_output_' + str(round_num) + '.pt'
    total_date_pid_name = 'test_date_pid_' + str(round_num) + '.pt'
    save_path = "/home/datamake117/data/haris/DL/" + main_folder_name
    
    # 根据给定的时间范围 dt1 到 dt3，选出训练集的日期列表。之后，有一个特别的日期范围处理（过滤掉指定日期段的训练数据）。
    date_list_train = total_date_list[np.where((total_date_list >= dt1) & (total_date_list < dt3))[0]]
    # 若20240223在训练周期或测试周期内，训练周期或测试周期去除20240201-20240223这一时间段
    if 20240223 >= dt1 and 20240223 <= dt3:
        date_list_train = np.array([date_train for date_train in date_list_train if date_train < 20240201 or date_train > 20240223])
    total_ts_train_val1 = np.zeros((len(date_list_train), pid_num, factor_num)) # 因子数据 shape: (len(date_list_train), pid_num, factor_num)
    total_label_train_val = np.zeros((len(date_list_train), pid_num, 5))        # 标签数据 shape: (len(date_list_train), pid_num, 5)
    total_group_train_val = np.zeros((len(date_list_train), pid_num, 1))        # 流动性数据 shape: (len(date_list_train), pid_num, 1)
    for i in trange(len(date_list_train), desc='train_val_data'):
        date = date_list_train[i]
        total_ts_train_val1[i, :, :] = grouped.loc[date].iloc[:pid_num, :]          # 因子
        total_label_train_val[i, :, :] = grouped_label.loc[date].iloc[:pid_num, :]  # 标签
        # 根据流动性调整收益率前7%-10%附近的训练标签：label(returns)
        total_label_train_val[i, :, 0] = adjust_daily_returns(total_label_train_val[i, :, 0], total_label_train_val[i, :, 4])
        total_group_train_val[i, :, :] = np.array(grouped_liquidity.loc[date])[:pid_num].reshape(-1, 1)  # 流动性
    
    # 类似地，date_list_test 被定义为测试集的日期范围，时间从 dt4 到 dt5。
    date_list_test = total_date_list[np.where((total_date_list >= dt4) & (total_date_list < dt5))[0]]
    total_ts_test1 = np.zeros((len(date_list_test), pid_num, factor_num))
    total_label_test = np.zeros((len(date_list_test), pid_num, 5))
    total_group_test = np.zeros((len(date_list_test), pid_num, 1))
    for i in trange(len(date_list_test), desc='test_data'):
        date = date_list_test[i]
        total_ts_test1[i, :, :] = grouped.loc[date].iloc[:pid_num, :]
        total_label_test[i, :, :] = grouped_label.loc[date].iloc[:pid_num, :]
        total_label_test[i, :, 0] = adjust_daily_returns(total_label_test[i, :, 0], total_label_test[i, :, 4])
        total_group_test[i, :, :] = np.array(grouped_liquidity.loc[date])[:pid_num].reshape(-1, 1)
    
    # 流动性数据归一化
    def min_max_standard(column):
        return (column - column.min()) / (column.max() - column.min())
    print('Min-max scaling.')
    total_group_train_val, total_group_test = min_max_standard(total_group_train_val), min_max_standard(total_group_test)
    
    # 因子数据标准化
    print('Standard scaling.')
    scaler = StandardScaler()
    total_ts_train_val1 = np.apply_along_axis(
        lambda x: np.clip(x, np.percentile(x, 0.5), np.percentile(x, 99.5)), axis=0, arr=total_ts_train_val1.reshape(-1, factor_num)
        )  # 去极值，保留5%-95%数据
    total_ts_train_val1 = total_ts_train_val1.reshape(len(date_list_train), pid_num, factor_num)
    total_ts_train_val1 = np.nan_to_num(scaler.fit_transform(total_ts_train_val1.reshape(-1, factor_num)).reshape(len(date_list_train), pid_num, factor_num), nan=0)
    total_ts_test1 = np.apply_along_axis(
        lambda x: np.clip(x, np.percentile(x, 0.5), np.percentile(x, 99.5)), axis=0, arr=total_ts_test1.reshape(-1, factor_num)
        )
    total_ts_test1 = total_ts_test1.reshape(len(date_list_test), pid_num, factor_num)
    total_ts_test1 = np.nan_to_num(scaler.transform(total_ts_test1.reshape(-1, factor_num)).reshape(len(date_list_test), pid_num, factor_num), nan=0)
    
    # KFold 交叉验证（并行训练）  TODO: 看到这里
    print('KFold training.')
    kf = KFold(n_splits=total_train_num, shuffle=False)
    processes = []
    for train_num, index_tuple in enumerate(kf.split(total_ts_train_val1)):
        p = mp.Process(
            target=train_one_Fold, 
            args=(
                round_num, train_num, index_tuple, main_folder_name,
                total_ts_train_val1, total_label_train_val, total_group_train_val, date_list_train,
                total_ts_test1, total_label_test, total_group_test, date_list_test,
                correlation_df, seed_list, dt1, dt2, dt3, dt4, dt5,
                factor_num, corr_thres, save_path, model_mode, multi_model
                )
            )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # 保存测试数据
    print('Save test data.')
    total_test_output = []
    for train_num in range(total_train_num):
        test_name = 'test_output_ic' + str(round_num) + str(train_num) + '.pt'
        test_path = os.path.join(save_path, test_name)
        total_test_output.append(torch.load(test_path))
        
    total_test_path = os.path.join(save_path, total_test_name)
    total_date_pid_path = os.path.join(save_path, total_date_pid_name)
    
    total_test_output = torch.stack(total_test_output)
    weight_tensor = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3]).view(-1, *([1] * (total_test_output.dim() - 1)))
    total_test_output = (total_test_output * weight_tensor).sum(dim=0)
    torch.save(total_test_output, total_test_path)
    
    stocks = np.array(grouped_label.loc[20200102].index)
    repeated_stocks = np.tile(stocks, len(date_list_test))
    repeated_dates = np.repeat(date_list_test, len(stocks))
    date_pid_test = np.column_stack((repeated_dates, repeated_stocks))
    torch.save(date_pid_test, total_date_pid_path)
    
    del total_ts_train_val1
    del total_ts_test1
    del total_label_train_val
    del total_label_test
    del total_group_train_val
    del total_group_test
    
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main_device_name = 0
    print('Read Factor.')
    factor = pd.read_pickle('/home/datamake117/data/haris/dataset/total_date.pkl')                      # 日期+股票代码
    grouped = pd.read_pickle('/home/datamake117/data/haris/dataset/grouped_adj.pkl').fillna(0)          # 特征
    grouped_label = pd.read_pickle('/home/datamake117/data/haris/dataset/grouped_label_adj.pkl')        # 标签
    grouped_liquidity = pd.read_pickle('/home/datamake117/data/haris/dataset/grouped_liquidity.pkl')    # 流动性指标
    grouped_liquidity.index = grouped_liquidity.index.strftime('%Y%m%d').astype(int)
    correlation_df = pd.read_pickle('/home/datamake117/data/haris/dataset/corr_byday_abs.pkl')          # 因子筛选辅助数据
    correlation_df.index = correlation_df.index.strftime('%Y%m%d').astype(int)
    total_date_list = np.array(factor['date'].drop_duplicates().tolist())                               # 日期列表

folder_path = "/home/datamake117/data/haris/DL/" + main_folder_name
os.makedirs(folder_path, exist_ok=True)

# 第1轮
print('Round 1.')
round_num = 1
dt1 = int(pd.to_datetime("2020-07-01").strftime('%Y%m%d'))  # 训练集开始时间
dt2 = int(pd.to_datetime("2022-07-01").strftime('%Y%m%d'))  # 验证集开始时间
dt3 = int(pd.to_datetime("2022-12-30").strftime('%Y%m%d'))  # 验证集结束时间
dt4 = int(pd.to_datetime("2023-01-01").strftime('%Y%m%d'))  # 测试集开始时间
dt5 = int(pd.to_datetime("2023-07-01").strftime('%Y%m%d'))  # 测试集结束时间
main(
    round_num, dt1, dt2, dt3, dt4, dt5,
    correlation_df, grouped, grouped_label, grouped_liquidity,
    total_date_list, main_folder_name, corr_thres=0.9
    )
torch.cuda.empty_cache()
gc.collect()

test_output1 = torch.load("/home/datamake117/data/haris/DL/" + main_folder_name + "/test_output_1.pt")
test_output = torch.cat([test_output1])
test_output = test_output.cpu()
date_pid1 = torch.load("/home/datamake117/data/haris/DL/" + main_folder_name + "/test_date_pid_1.pt")
total_date_pid = np.concatenate([date_pid1], axis=0)
total_date_pid_test = total_date_pid
grading_factor = pd.DataFrame(index=np.unique(total_date_pid_test[:, 0]), columns=np.unique(total_date_pid_test[:, 1]))
test_output_list = test_output.tolist()
for i in range(len(total_date_pid_test)):
    grading_factor.loc[total_date_pid_test[i][0], total_date_pid_test[i][1]] = test_output_list[i]
grading_factor.to_pickle("/home/datamake117/data/haris/DL/" + main_folder_name + "/单次_KFold_2023.pkl")

gc.collect()

# 第2-8轮
total_date_list = np.array(factor['date'].drop_duplicates().tolist())
rolling_step = 3    # 3个月滚动训练
window_size = 24    # 训练集大小
val_size = 3        # 验证集大小
corr_thres = 0.9
for round_num in range(2, 9):
    print('Round %i.' % round_num)
    start_date = pd.to_datetime('2021-01-01')
    dt1 = start_date + relativedelta(months=rolling_step * (round_num - 2)) # 训练集开始时间
    dt2 = dt1 + relativedelta(months=window_size)                           # 验证集开始时间
    dt3 = dt2 + relativedelta(months=val_size)                              # 验证集结束时间
    dt4 = dt3                                                               # 测试集开始时间
    dt5 = dt3 + relativedelta(months=rolling_step)                          # 测试集结束时间
    dt3 = total_date_list[total_date_list < int(dt3.strftime('%Y%m%d'))][-1]
    dt1, dt2, dt3, dt4, dt5 = int(dt1.strftime('%Y%m%d')), int(dt2.strftime('%Y%m%d')), int(dt3), int(dt4.strftime('%Y%m%d')), int(dt5.strftime('%Y%m%d'))
    main(
        round_num, dt1, dt2, dt3, dt4, dt5,
        correlation_df, grouped, grouped_label, grouped_liquidity,
        total_date_list, main_folder_name, corr_thres=0.9, seed_num=5, model_mode=False
        )
    torch.cuda.empty_cache()
    gc.collect()

test_output_list = []
for round_num in range(2, 9):
    test_output = torch.load("/home/datamake117/data/haris/DL/" + main_folder_name + "/test_output_" + str(round_num) + ".pt")
    test_output_list.append(test_output)
test_output = torch.cat(test_output_list)
test_output = test_output.cpu()
date_pid_list = []
for round_num in range(2, 9):
    date_pid = torch.load("/home/datamake117/data/haris/DL/" + main_folder_name + "/test_date_pid_" + str(round_num) + ".pt")
    date_pid_list.append(date_pid)
total_date_pid = np.concatenate(date_pid_list, axis=0)
total_date_pid_test = total_date_pid
grading_factor = pd.DataFrame(index=np.unique(total_date_pid_test[:, 0]), columns=np.unique(total_date_pid_test[:, 1]))
test_output_list = test_output.tolist()
for i in range(len(total_date_pid_test)):
    grading_factor.loc[total_date_pid_test[i][0], total_date_pid_test[i][1]] = test_output_list[i]
grading_factor2023 = pd.read_pickle("/home/datamake117/data/haris/DL/" + main_folder_name + "/单次_KFold_2023.pkl")
grading_factor2023 = grading_factor2023[grading_factor2023.index < 20230401]
grading_factor = pd.concat([grading_factor2023, grading_factor], axis=0)
grading_factor.to_pickle("/home/datamake117/data/haris/DL/" + main_folder_name + "/单次_KFold_0.pkl")
