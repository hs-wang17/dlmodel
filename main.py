import os
import random
import gc
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import torch
import torch.multiprocessing as mp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from utils.model import *
from utils.logger import *
from utils.processing import *
from utils.train import *


def main(
    round_num, dt1, dt2, dt3, dt4, dt5,
    correlation_df, grouped, grouped_label, grouped_liquidity,
    total_date_list, main_folder_name, 
    pid_num=5243, factor_num=2790, corr_thres=0.9, seed_num=5, model_mode=False, multi_model=6
    ):
    '''
    TODO: 参数含义
    para round_num:
    para dt1: 训练集开始时间
    para dt2: 验证集开始时间
    para dt3: 验证集结束时间
    para dt4: 测试集开始时间
    para dt5: 测试集结束时间
    para correlation_df:
    para grouped: 按日期分组的因子数据
    para grouped_label: 按日期分组的标签数据
    para grouped_liquidity: 按日期分组的流动性数据
    para total_date_list:
    para main_folder_name:
    para pid_num: 股票数量
    para factor_num: 因子数量
    para corr_thres:
    para seed_num:
    para model_mode:
    para multi_model:
    '''
    seed_list = []
    for i in range(seed_num):
        random.seed(i)
        seed_list.append(list(random.sample(range(100), multi_model)))
    total_train_num = len(seed_list)  # seed_num * multi_model
    total_test_output = []
    total_test_name = 'test_output_' + str(round_num) + '.pt'
    total_date_pid_name = 'test_date_pid_' + str(round_num) + '.pt'
    save_path = "/home/datamake117/Documents/haris/DL/" + main_folder_name
    
    # 根据给定的时间范围 dt1 到 dt3，选出训练集的日期列表。之后，有一个特别的日期范围处理（过滤掉指定日期段的训练数据）。
    # 删除空值（去除预测目标中为空值的样本） TODO: 没看懂这个逻辑
    date_list_train = total_date_list[np.where((total_date_list >= dt1) & (total_date_list < dt3))[0]]
    if pd.to_datetime("2024-02-23") >= dt1 and pd.to_datetime("2024-02-23") <= dt3:
        date_list_train = np.array([date_train for date_train in date_list_train if date_train < pd.to_datetime("2024-02-01") or date_train > pd.to_datetime("2024-02-23")])
    total_ts_train_val1 = np.zeros((len(date_list_train), pid_num, factor_num)) # shape: (len(date_list_train), pid_num, factor_num)
    total_label_train_val = np.zeros((len(date_list_train), pid_num, 5))        # shape: (len(date_list_train), pid_num, 5) 
    total_group_train_val = np.zeros((len(date_list_train), pid_num, 1))        # shape: (len(date_list_train), pid_num, 1)
    for i, date in enumerate(date_list_train):
        total_ts_train_val1[i, :, :] = grouped.loc[date].iloc[:pid_num, :]
        total_label_train_val[i, :, :] = grouped_label.loc[date].iloc[:pid_num, :]
        total_label_train_val[i, :, 0] = adjust_daily_returns(total_label_train_val[i, :, 0], total_label_train_val[i, :, 4])  # returns 和 liquidity
        total_group_train_val[i, :, :] = np.array(grouped_liquidity.loc[date])[:pid_num].reshape(-1, 1)
    
    # 类似地，date_list_test 被定义为测试集的日期范围，时间从 dt4 到 dt5。
    date_list_test = total_date_list[np.where((total_date_list >= dt4) & (total_date_list < dt5))[0]]
    total_ts_test1 = np.zeros((len(date_list_test), pid_num, factor_num))
    total_label_test = np.zeros((len(date_list_test), pid_num, 5))
    total_group_test = np.zeros((len(date_list_test), pid_num, 1))
    for i, date in enumerate(date_list_test):
        total_ts_test1[i, :, :] = grouped.loc[date].iloc[:pid_num, :]
        total_label_test[i, :, :] = grouped_label.loc[date].iloc[:pid_num, :]
        total_label_test[i, :, 0] = adjust_daily_returns(total_label_test[i, :, 0], total_label_test[i, :, 4])
        total_group_test[i, :, :] = np.array(grouped_liquidity.loc[date])[:pid_num].reshape(-1, 1)
    
    def min_max_standard(column):
        return (column - column.min()) / (column.max() - column.min())
    
    # 数据归一化
    total_group_train_val = min_max_standard(total_group_train_val)
    total_group_test = min_max_standard(total_group_test)
    scaler = StandardScaler()
    total_ts_train_val1 = np.apply_along_axis(
        lambda x: np.clip(x, np.percentile(x, 0.5), np.percentile(x, 99.5)), axis=0, arr=total_ts_train_val1.reshape(-1, factor_num)
        )
    total_ts_train_val1 = total_ts_train_val1.reshape(len(date_list_train), pid_num, factor_num)
    total_ts_train_val1 = np.nan_to_num(scaler.fit_transform(total_ts_train_val1.reshape(-1, factor_num)).reshape(len(date_list_train), pid_num, factor_num), nan=0)
    total_ts_test1 = np.apply_along_axis(
        lambda x: np.clip(x, np.percentile(x, 0.5), np.percentile(x, 99.5)), axis=0, arr=total_ts_test1.reshape(-1, factor_num)
        )
    total_ts_test1 = total_ts_test1.reshape(len(date_list_test), pid_num, factor_num)
    total_ts_test1 = np.nan_to_num(scaler.transform(total_ts_test1.reshape(-1, factor_num)).reshape(len(date_list_test), pid_num, factor_num), nan=0)
    
    # KFold 交叉验证
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
    
    stocks = np.array(grouped_label.loc['2020-01-02'].index)
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
    factor = pd.read_pickle('/home/datamake117/test1101/Data/all_feature/total_date.pkl')
    grouped = pd.read_pickle('/home/datamake117/test1101/Data/all_feature/grouped_adj3.pkl').fillna(0)
    grouped_label = pd.read_pickle('/home/datamake117/test1101/Data/all_feature/grouped_label_adj3.pkl')
    grouped_liquidity = pd.read_pickle('/home/datamake117/test1101/Data/all_feature/grouped_liquidity.pkl')
    correlation_df = pd.read_pickle('/home/datamake117/test1101/Data/辅助数据/corr_byday_abs3.pkl')
    total_date_list = np.array(factor['date'].drop_duplicates().tolist())

    # 第1轮
    round_num = 1
    dt1 = pd.to_datetime("2020-07-01")  # 训练集开始时间
    dt2 = pd.to_datetime("2022-07-01")  # 验证集开始时间
    dt3 = pd.to_datetime("2022-12-30")  # 验证集结束时间
    dt4 = pd.to_datetime("2023-01-01")  # 测试集开始时间
    dt5 = pd.to_datetime("2023-07-01")  # 测试集结束时间
    main(
        round_num, dt1, dt2, dt3, dt4, dt5,
        correlation_df, grouped, grouped_label, grouped_liquidity,
        total_date_list, main_folder_name, corr_thres=0.9
        )

    torch.cuda.empty_cache()
    gc.collect()

    test_output1 = torch.load("/home/datamake117/Documents/haris/DL/" + main_folder_name + "/test_output_1.pt")
    test_output = torch.cat([test_output1])
    test_output = test_output.cpu()
    date_pid1 = torch.load("/home/datamake117/Documents/haris/DL/" + main_folder_name + "/test_date_pid_1.pt")
    total_date_pid = np.concatenate([date_pid1], axis=0)
    total_date_pid_test = total_date_pid
    grading_factor = pd.DataFrame(index=np.unique(total_date_pid_test[:, 0]), columns=np.unique(total_date_pid_test[:, 1]))
    test_output_list = test_output.tolist()
    for i in range(len(total_date_pid_test)):
        grading_factor.loc[total_date_pid_test[i][0], total_date_pid_test[i][1]] = test_output_list[i]
    grading_factor.to_pickle("/home/datamake117/Documents/haris/DL/" + main_folder_name + "/单次_KFold_2023.pkl")

    gc.collect()

    # 第2-8轮
    total_date_list = np.array(factor['date'].drop_duplicates().tolist())
    rolling_step = 3    # 3个月滚动训练
    window_size = 24    # 训练集大小
    val_size = 3        # 验证集大小
    model_max_num = 20
    model_num = 4
    BATCH_SIZE = 2
    BATCH_SIZE_VAL = 1
    ic_threshold = 0.93
    corr_thres = 0.9
    for round_num in range(2, 9):
        start_date = pd.to_datetime('2021-01-01')
        dt1 = start_date + relativedelta(months=rolling_step * (round_num - 2)) # 训练集开始时间
        dt2 = dt1 + relativedelta(months=window_size)                           # 验证集开始时间
        dt3 = dt2 + relativedelta(months=val_size)                              # 验证集结束时间
        dt4 = dt3                                                               # 测试集开始时间
        dt5 = dt3 + relativedelta(months=rolling_step)                          # 测试集结束时间
        dt3 = total_date_list[total_date_list < dt3][-1]
        main(
            round_num, dt1, dt2, dt3, dt4, dt5,
            correlation_df, grouped, grouped_label, grouped_liquidity,
            total_date_list, main_folder_name, corr_thres=0.9, seed_num=5, model_mode=False
            )
        
        torch.cuda.empty_cache()
        gc.collect()

    test_output_list = []
    for round_num in range(2, 9):
        test_output = torch.load("/home/datamake117/Documents/haris/DL/" + main_folder_name + "/test_output_" + str(round_num) + ".pt")
        test_output_list.append(test_output)
    test_output = torch.cat(test_output_list)
    test_output = test_output.cpu()
    date_pid_list = []
    for round_num in range(2, 9):
        date_pid = torch.load("/home/datamake117/Documents/haris/DL/" + main_folder_name + "/test_date_pid_" + str(round_num) + ".pt")
        date_pid_list.append(date_pid)
    total_date_pid = np.concatenate(date_pid_list, axis=0)
    total_date_pid_test = total_date_pid
    grading_factor = pd.DataFrame(index=np.unique(total_date_pid_test[:, 0]), columns=np.unique(total_date_pid_test[:, 1]))
    test_output_list = test_output.tolist()
    for i in range(len(total_date_pid_test)):
        grading_factor.loc[total_date_pid_test[i][0], total_date_pid_test[i][1]] = test_output_list[i]
    grading_factor2023 = pd.read_pickle("/home/datamake117/Documents/haris/DL/" + main_folder_name + "/单次_KFold_2023.pkl")
    grading_factor2023 = grading_factor2023[grading_factor2023.index < pd.to_datetime('2023-04-01')]
    grading_factor = pd.concat([grading_factor2023, grading_factor], axis=0)
    grading_factor.to_pickle("/home/datamake117/Documents/haris/DL/" + main_folder_name + "/单次_KFold_0.pkl")
