import os
import time
import gc
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.cluster import KMeans

from methods.model import *
from methods.logger import *
from methods.processing import *
from methods.hyper import *


wpcc = PartialCosLoss()

# K折交叉验证训练中的一折
def train_one_Fold(
    round_num, train_num, index_tuple, main_folder_name,
    total_ts_train_val1, total_label_train_val, total_group_train_val, date_list_train,
    total_ts_test1, total_label_test, total_group_test, date_list_test,
    correlation_df, Seed_list, dt1, dt2, dt3, dt4, dt5,
    factor_num, corr_thres, save_path, model_mode=False, multi_model=6
    ):
    '''
    用于训练深度学习模型的一折交叉验证函数，支持多模型并行训练，并包含特征选择、早停、日志记录及资源管理等功能
    ** 数据相关：
    para total_ts_train_val1: 训练+验证集的因子数据
    para total_label_train_val: 训练+验证集的标签数据
    para total_group_train_val: 训练+验证集的流动性数据
    para total_ts_test1: 测试集的因子数据
    para total_label_test: 测试集的标签数据
    para total_group_test: 测试集的流动性数据
    para date_list_train: 训练集的日期列表
    para correlation_df: 因子与标签的相关性数据，用于特征筛选
    TODO: 未使用的参数：total_group_train_val和total_group_test。
    ** 训练配置：
    para round_num: 当前交叉验证的轮数（周期序号）
    para train_num: K折交叉验证的的第几折
    para Seed_list: 随机种子列表，用于模型初始化
    para corr_thres: 相关性阈值，用于过滤高相关因子
    para model_mode: 是否加载已有模型（用于继续训练）
    para multi_model: 模型数量
    ** 其他：
    para dt1: 训练集开始时间
    para dt2: 验证集开始时间
    para dt3: 验证集结束时间
    para dt4: 测试集开始时间
    para dt5: 测试集结束时间
    dt1 ------训练集------ dt2 ------验证集------ dt3/dt4 ------测试集------ dt5
    '''
    
    # 设置路径
    path_name = 'best_network_ic_' + str(round_num) + str(train_num)
    test_name = 'test_output_ic' + str(round_num) + str(train_num) + '.pt'
    logger_path = save_path + '/logger.log'
    
    # 划分训练集、验证集和测试集
    train_index, val_index = index_tuple
    total_ts_train1 = total_ts_train_val1[train_index, :, :]
    total_ts_val1 = total_ts_train_val1[val_index[1:], :, :]  # [1:] 是为了跳过第一日
    total_label_train = total_label_train_val[train_index, :]
    total_label_val = total_label_train_val[val_index[1:], :]
    total_group_train = total_group_train_val[train_index, :]
    total_group_val = total_group_train_val[val_index[1:], :]

    x_train1 = torch.from_numpy(total_ts_train1).float()
    y_train = torch.from_numpy(total_label_train).float()
    group_train = torch.from_numpy(total_group_train).float()
    torch_dataset_train = Data.TensorDataset(x_train1, group_train, y_train)
    loader_train = Data.DataLoader(dataset=torch_dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)   # 使用GPU训练：pin_memory=True
    
    x_val1 = torch.from_numpy(total_ts_val1).float()
    y_val = torch.from_numpy(total_label_val).float()
    group_val = torch.from_numpy(total_group_val).float()
    torch_dataset_val = Data.TensorDataset(x_val1, group_val, y_val)
    loader_val = Data.DataLoader(dataset=torch_dataset_val, batch_size=BATCH_SIZE_VAL, shuffle=False, num_workers=0, pin_memory=True)
    
    x_test1 = torch.from_numpy(total_ts_test1).float()
    y_test = torch.from_numpy(total_label_test).float()
    group_test = torch.from_numpy(total_group_test).float()
    torch_dataset_test = Data.TensorDataset(x_test1, group_test, y_test)
    loader_test = Data.DataLoader(dataset=torch_dataset_test, batch_size=5, shuffle=False, num_workers=0, pin_memory=True)
    
    # correlation_df这个就是用每天的因子和当天要预测的标签计算出的包括IC、IC平方和IC立方最大值的dataframe
    # 用correlation_df聚类决定辅助因子分组
    data_scaled = correlation_df.loc[date_list_train].iloc[train_index, 680:]
    data_scaled = data_scaled.fillna(0)
    kmeans = BalancedKMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data_scaled.T)
    # 将feature0中的680个主要因子与feature1中辅助因子的每个聚类结果组合，生成3组特征索引（factor_list），用于后续多模型训练
    factor_list = []
    factor_list.append(torch.from_numpy(np.array(list(range(0, 680)) + [list(range(680, 2221))[i] for i in np.where(clusters==0)[0]])))
    factor_list.append(torch.from_numpy(np.array(list(range(0, 680)) + [list(range(680, 2221))[i] for i in np.where(clusters==1)[0]])))
    factor_list.append(torch.from_numpy(np.array(list(range(0, 680)) + [list(range(680, 2221))[i] for i in np.where(clusters==2)[0]])))

    # 根据相关性阈值筛选掉高相关性的因子（保留的逻辑目前是按顺序保留第一个）
    mask = generate_mask(x_train1.reshape(-1, x_train1.size(2)), corr_thres=corr_thres).cpu()

    # 创建模型列表，主要可能一些复杂的地方就在这里
    model_list = [base_model(
        mask=mask.to(torch.device("cuda:" + str(i % GPU_COUNT))),
        factor_list=factor_list[(i % 3)].to(torch.device("cuda:" + str((i % GPU_COUNT)))),
        seed=Seed_list[train_num][i]
        ).to(torch.device("cuda:" + str((i % GPU_COUNT)))) for i in range(multi_model)]
    
    # 是否加载已有模型（用于继续训练）
    if model_mode:
        temp_path_name = 'best_network_ic_' + str(round_num) + str(train_num)
        for n in range(multi_model):
            path_name_model = temp_path_name + '_' + str(n) + '.pth'
            model_path = os.path.join(save_path, path_name_model)
            model_list[n].load_state_dict(torch.load(model_path))
    
    # 训练时每个模型是异步训练的，分别创建独立的优化器
    optimizer_list = []
    for n in range(multi_model):
        base_optimizer = torch.optim.Adam
        optimizer_list.append(SAM(model_list[n].parameters(), base_optimizer, lr=learning_rate))

    logger = get_logger(logger_path)
    logger.info(f"Period{round_num}, Train{train_num}, Train Period:{dt1}-{dt2}, Val Period:{dt2}-{dt3}, Test Period:{dt4}-{dt5}")
    logger.info(f"Train1 Shape: {x_train1.shape}, Val1 Shape: {x_val1.shape}, Test1 Shape: {x_test1.shape}")
    logger.info("Start Training")
    early_stopping = EarlyStopping(save_path, logger, experts_num=multi_model)

    # 训练模型
    for epoch in range(num_epochs):
        # 训练集
        start = time.time()
        train_loss_list = []
        for n in range(multi_model):
            model_list[n].train()
        for step, (batch_x1, batch_group, batch_y) in enumerate(loader_train):
            batch_x1 = batch_x1.reshape(-1, batch_x1.size(2)).to(torch.device("cuda:0"))
            batch_y = batch_y.reshape(-1, batch_y.size(2)).to(torch.device("cuda:0"))
            batch_group = batch_group.reshape(-1, batch_group.size(2)).to(torch.device("cuda:0"))
            nan_index = torch.isnan(batch_y[:, 0])
            batch_y = batch_y[~nan_index]
            batch_x1 = batch_x1[~nan_index]
            batch_group = batch_group[~nan_index]
            # 使用异步计算（torch.jit.fork）来并行训练多个模型
            futures_list = []
            for n in range(multi_model):
                futures_list.append(torch.jit.fork(
                    backward_and_step, n, optimizer_list[n], batch_x1, batch_group, batch_y,
                    model_list[n], early_stopping.early_stop, early_stopping.val_loss_min, wpcc
                    ))
            losses = [torch.jit.wait(future) for future in futures_list]
            train_loss_list.append(torch.mean(torch.stack(losses)).item())
        train_loss = sum(train_loss_list) / len(train_loss_list)
        
        # 验证集
        for n in range(multi_model):
            model_list[n].eval()
        with torch.no_grad():
            val_loss_list = []
            for step, (batch_x1, batch_group, batch_y) in enumerate(loader_val):
                batch_x1 = batch_x1.reshape(-1, batch_x1.size(2)).to(torch.device("cuda:0"))
                batch_y = batch_y.reshape(-1, batch_y.size(2)).to(torch.device("cuda:0"))
                batch_group = batch_group.reshape(-1, batch_group.size(2)).to(torch.device("cuda:0"))
                nan_index = torch.isnan(batch_y[:, 0])
                batch_y = batch_y[~nan_index]
                batch_x1 = batch_x1[~nan_index]
                batch_group = batch_group[~nan_index]
                val_outputs_list = [model_list[n](batch_x1) for n in range(multi_model)]
                val_loss_list_model = []
                for n in range(multi_model):
                    val_loss_list_model.append(wpcc(val_outputs_list[n], batch_y))
                val_loss_list.append(val_loss_list_model)
            val_loss = [torch.median(torch.tensor([val_loss_list[j][n] for j in range(step + 1)])).cpu() for n in range(multi_model)]
            
        end = time.time()
        running_time = end - start
        template = ",".join([f"{val_loss[j]}" for j in range(len(val_loss))])
        logger.info(f"Epoch[{epoch + 1}/{num_epochs}], Time:{running_time:.2f}sec, Train Loss: {train_loss:.6f}, Val Loss: {template}")

        # 早停
        early_stopping(val_loss, model_list, path_name)
        if np.sum(early_stopping.early_stop) == early_stopping.experts_num:
            logger.info("Early stopping")
            break

    val_loss_min = early_stopping.val_loss_min
    template = ",".join([f"{val_loss_min[i]}" for i in range(len(val_loss_min))])
    logger.info(f"Val Loss: {template}")

    # 测试集（从六个模型里选择topk个模型） -> online learning
    model_path = os.path.join(save_path, path_name)
    test_path = os.path.join(save_path, test_name)
    
    for n in range(multi_model):
        model_list[n].eval()
    with torch.no_grad():
        test_outputs_list = []
        test_loss_list = []
        topk_last_week = [0, 1, 2, 3, 4, 5]
        # 这一块可能需要一些特殊处理，是在预测时候是分批的
        # 这里放到模拟盘可能作为inference阶段可能需要调整逻辑
        for step, (batch_x1, batch_group, batch_y) in enumerate(loader_test):
            batch_y_list = []
            outputs_list = []
            # 对每个批次，取前 5 个样本并进行处理（一周数据）
            for i in range(min(5, batch_y.size(0)) - 1):
                batch_y_list.append(batch_y[i].to(torch.device("cuda:0")))
                temp_output = [model_list[n](batch_x1[i]) for n in range(multi_model)]
                outputs_list.append(temp_output)
            loss_list_model = []
            for n in range(multi_model):
                loss_list_day = []
                for i in range(min(5, batch_y.size(0)) - 1):
                    outputs = outputs_list[i][n]
                    loss_list_day.append(simu_trade(outputs, batch_y_list[i]).to(torch.device("cuda:0")))
                loss_week = torch.mean(torch.tensor(loss_list_day))
                loss_list_model.append(loss_week)
                
            batch_x1 = batch_x1.reshape(-1, batch_x1.size(2))
            batch_group = batch_group.reshape(-1, batch_group.size(2))
            batch_y = batch_y.reshape(-1, batch_y.size(2))
            nan_index = torch.isnan(batch_y[:, 0])
            test_outputs = [model_list[n](batch_x1) for n in range(multi_model)]
            test_outputs = torch.stack([test_outputs[j].to(torch.device("cuda:0")) for j in topk_last_week]).mean(dim=0)
            _, topk_last_week = torch.topk(torch.tensor(loss_list_model), 6, largest=True)  # 修改数值6，可只保留最好的k个模型

            out_mean = test_outputs.mean(axis=0)
            out_std = test_outputs.std(axis=0)
            test_outputs = (test_outputs - out_mean) / out_std
            test_loss = simu_trade(test_outputs, batch_y)
            test_outputs_list.append(test_outputs)
            test_loss_list.append(test_loss)

        test_outputs = torch.cat(test_outputs_list).cpu()
        test_loss = torch.tensor(test_loss_list)
        torch.save(test_outputs, test_path)

    logger.info(f"Test Loss: {test_loss.mean().cpu():.6f}")
    
    del total_ts_train_val1
    del total_label_train_val
    del total_group_train_val
    del date_list_train
    del total_ts_train1
    del total_label_train
    del total_group_train
    del total_ts_val1
    del total_label_val
    del total_group_val
    del x_train1
    del y_train
    del x_val1
    del y_val
    del x_test1
    del y_test
    del group_train
    del group_val
    del group_test
    for model in model_list:
        del model
    del model_list
    del mask
    del batch_x1
    del batch_group
    del batch_y
    del torch_dataset_train
    del torch_dataset_val
    del torch_dataset_test
    del loader_train
    del loader_val
    del loader_test
    
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info('Finish ' + str(train_num) + ' Fold.')
