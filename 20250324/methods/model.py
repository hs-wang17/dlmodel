import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import mutual_info_score


# SAM（Sharpness-Aware Minimization）优化器
# 损失函数的锐利度：在高维参数空间中，损失函数往往具有复杂的形状。
# SAM的假设：局部最小值通常会形成较尖锐的区域，而平缓的区域则通常对应着较好的泛化性能。
# SAM通过在梯度的方向上引入扰动，试图使得参数更新到平坦区域，以减少过拟合。
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        '''
        params: 传入的参数集合，通常是模型的权重参数。
        base_optimizer: 传入基础优化器（如 Adam 或 SGD），SAM 会在此基础上进行修改。
        rho: 控制扰动大小的超参数，通常较小。
        adaptive: 如果为 True，则在更新时每个参数会使用不同的步长；如果为 False，所有参数使用相同的步长。
        '''
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        '''
        first_step: 这是SAM的第一步更新，在这一步中，优化器将参数扰动到损失函数的局部最大值附近。
        grad_norm: 计算所有参数的梯度范数（具体见 _grad_norm 方法）。这将决定每个参数更新的步长。
        scale: 计算每个参数的更新比例，rho 控制了更新的大小，grad_norm 是当前梯度的范数，避免除以0，因此加上了一个很小的常数（1e-12）。
        e_w: 计算每个参数的扰动量。如果 adaptive=True，则每个参数的扰动量会乘以 torch.pow(p, 2)（即自适应扰动）。如果 adaptive=False，则所有参数的扰动量大小相同。
        p.add_(e_w): 更新参数，这一步是让参数沿梯度方向“爬升”，即移动到损失函数的局部最大值。
        zero_grad: 如果 zero_grad=True，则在这一步结束时将所有参数的梯度清零，防止在后续步骤中影响梯度计算。
        '''
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        '''
        second_step: 在这一步中，优化器将参数恢复到第一步更新之前的位置，之后使用基础优化器执行常规更新。
        p.data = self.state[p]["old_p"]: 将参数恢复到原来的位置，即“从 w + e(w) 回退到 w”。
        self.base_optimizer.step(): 执行实际的优化步骤，基础优化器（如 Adam 或 SGD）在这个步骤中进行标准的更新，更新规则根据基础优化器的定义（通常是 p = p - η * grad）。
        zero_grad: 如果 zero_grad=True，则将所有参数的梯度清零，防止影响后续计算。
        '''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: 
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        '''
        step: 这是 SAM 的核心步骤，首先调用 first_step 计算扰动，然后通过 closure 执行一次前向和反向传播，最后调用 second_step 执行实际的优化更新。
        closure: closure 是一个回调函数，用来计算损失值并进行反向传播。为了保证 SAM 进行完整的前向和反向传播，我们需要使用 closure 来获得损失值和梯度。
        first_step: 计算参数的扰动，并更新参数。
        closure(): 执行前向和反向传播，计算新的梯度。
        second_step: 执行第二步更新，恢复到原位置并应用基础优化器进行标准更新。
        '''
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"] if p.grad is not None
                    ]), p=2
                )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
        
# 到下一个注释前是一些定义的模型
# 这个是用来mask部分因子的，高相关性的因子的处理也还是用mask来处理的
class DynamicMaskLayer(nn.Module):
    def __init__(self, input_dim):
        super(DynamicMaskLayer, self).__init__()
        self.input_dim = input_dim
    
    def forward(self, x, mask=None):
        if mask == None:
            mask = generate_mask(x)
        x = x.to(mask.device)
        return x * mask

# 相关系数超过阈值时，保留前一个因子
def generate_mask(x, corr_thres=0.9):
    mean = torch.mean(x, dim=0)
    std = torch.std(x, dim=0)
    normalized_tensor = (x - mean) / std
    corr_matrix = torch.corrcoef(normalized_tensor.T)
    corr_matrix = torch.abs(corr_matrix)
    mask = torch.ones(x.size(1)).cuda()
    for i in range(corr_matrix.size(0)):
        if mask[i] == 0:
            continue
        correlated_features = corr_matrix[i] > corr_thres
        mask[correlated_features] = 0
        mask[i] = 1
    return mask

# 同时训练的多个模型的子模型，后面其实也可以进一步改成支持时间序列input的LSTM或stockMixer，这个的尝试可能得到下周一了
# 这里面用来决定因子分组（即每个子模型接收哪些因子）的是一个tensor形式的索引factor_list，可能需要单独存储
# 用于筛除高相关因子的mask可能也需要单独存储
def construct_graph(features, threshold=0.8, absolute=True, chunk_size=1000):
    """
    基于皮尔逊相关系数构建特征图结构（特征间相关系数）
    Args:
        features: 输入特征张量，形状为 (num_samples, num_features)
        threshold: 相关系数阈值，绝对值超过此值时保留边
        absolute: 是否使用相关系数的绝对值
        chunk_size: 分块大小，控制显存占用
    Returns:
        edge_index: 边的索引 [2, num_edges] (特征索引)
        edge_weight: 边的权重（相关系数值）
    """
    feature_matrix = features.float()
    num_features = feature_matrix.shape[0]
    device = feature_matrix.device
    edge_index = []
    edge_weight = []
    # 分块遍历特征对
    for i in range(0, num_features, chunk_size):
        for j in range(i + 1, num_features, chunk_size):            # 仅计算上三角部分
            chunk_i = feature_matrix[i : i + chunk_size]            # 获取当前特征块
            chunk_j = feature_matrix[j : j + chunk_size]
            # 计算块间相关系数矩阵 [chunk_i_size, chunk_j_size]
            corr_chunk = torch.corrcoef(
                torch.cat([chunk_i, chunk_j], dim=0)
            )[:chunk_i.shape[0], chunk_i.shape[0]:]
            # 生成坐标网格
            rows, cols = torch.meshgrid(
                torch.arange(chunk_i.shape[0], device=device) + i,
                torch.arange(chunk_j.shape[0], device=device) + j,
                indexing='ij'
            )
            # 筛选有效边
            mask = (rows != cols)
            if absolute:
                mask &= (torch.abs(corr_chunk) > threshold)
            else:
                mask &= (corr_chunk > threshold)
            # 添加有效边
            valid_rows = rows[mask]
            valid_cols = cols[mask]
            chunk_edges = torch.stack([valid_rows, valid_cols], dim=0)
            edge_index.append(chunk_edges)
            edge_weight.append(corr_chunk[mask])
    # 合并结果并去重
    edge_index = torch.cat(edge_index, dim=1) if edge_index else torch.empty((2, 0), dtype=torch.long, device=device)
    edge_weight = torch.cat(edge_weight, dim=0) if edge_weight else torch.empty((0,), dtype=torch.float, device=device)
    edge_index, unique_idx = torch.unique(edge_index, dim=1, return_inverse=True)
    edge_weight = edge_weight[unique_idx]
    # edge_weight = torch.nan_to_num(edge_weight, nan=0.0, posinf=0.0, neginf=0.0)
    return edge_index, edge_weight

class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.bn2(self.linear2(x))
        x += residual
        return F.relu(x)

class MergedGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(MergedGCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.residual_fc = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

    def forward(self, x, edge_index, edge_weight=None):
        residual = x
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.leaky_relu(x, 0.1)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i == 0 and self.residual_fc is not None:
                residual = self.residual_fc(residual)
            x += residual
        return F.layer_norm(x, x.shape[1:])

class base_model(nn.Module):
    def __init__(self, output_size=1, drop_out=0.2, mask=None, factor_list=None, seed=1):
        super(base_model, self).__init__()
        ic_mask = mask.to(dtype=torch.bool)
        factor_mask = torch.zeros(ic_mask.shape, dtype=torch.bool).to(ic_mask.device)
        factor_mask[factor_list] = True
        final_mask = ic_mask & factor_mask
        self.selected_factors = torch.nonzero(final_mask).squeeze()
        self.main_features = factor_list[:1250]
        # 主要特征处理分支
        self.main_processor = nn.Sequential(
            nn.Linear(len(self.main_features), 256),
            nn.BatchNorm1d(256),
            ResBlock(256, 256),
            ResBlock(256, 128),
            ResBlock(128, 64),
            nn.Dropout(drop_out),
            nn.Linear(64, 16)
        )
        # 组合特征处理分支
        self.combined_processor = nn.Sequential(
            nn.Linear(len(self.selected_factors), 256),
            nn.BatchNorm1d(256),
            ResBlock(256, 256),
            ResBlock(256, 128),
            ResBlock(128, 64),
            nn.Dropout(drop_out),
            nn.Linear(64, 16)
        )
        # 合并后的GCN处理
        self.merged_gcn = MergedGCN(
            input_dim=32,  # 16+16 from two branches
            hidden_dim=64,
            num_layers=1,
            dropout=drop_out
        )
        self.final_output = nn.Linear(64, output_size)
        self._initialize_weights(seed)

    def _initialize_weights(self, seed):
        # 保持原有的初始化逻辑
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, GCNConv):
                nn.init.kaiming_normal_(m.lin.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.to(self.selected_factors.device)                              # 将输入移动到正确的设备
        main_out = self.main_processor(x[:, self.main_features])            # 处理主要特征
        combined_out = self.combined_processor(x[:, self.selected_factors]) # 处理组合特征
        merged = torch.cat([main_out, combined_out], dim=1)                 # 合并特征
        edge_index, edge_weight = construct_graph(merged, threshold=0.8)    # 构建合并特征的图结构
        gcn_out = self.merged_gcn(merged, edge_index, edge_weight)          # GCN处理
        return self.final_output(gcn_out).squeeze(-1)                       # 最终输出


# 负责早停的类
class EarlyStopping:
    def __init__(self, save_path, logger, patience=10, verbose=True, delta=0, experts_num=6):
        '''
        save_path: 存储模型的路径，当早停时保存最优模型。
        logger: 用于记录日志，通常是训练过程中的信息输出。
        patience: 容忍度，即如果验证集损失在 patience 个训练周期内没有改善，则触发早停。
        verbose: 如果为 True，则在早停触发时会打印日志。
        delta: 验证损失下降的最小阈值。如果当前损失变化小于 delta，则认为没有改善。
        experts_num: 模型的个数。在这里，每个模型都有自己的早停机制（因此支持多个模型并行训练）。
        counter: 用于记录每个模型验证损失没有改善的训练周期数。
        best_score: 用来记录每个模型在验证集上的最佳损失。初始化为 None。
        early_stop: 每个模型是否已触发早停的标志列表。初始值为 False。
        val_loss_min: 每个模型在验证集上的最小损失值，初始化为正无穷。
        '''
        self.save_path = save_path
        self.logger = logger
        self.experts_num = experts_num
        self.patience = patience
        self.verbose = verbose
        self.counter = [0] * experts_num
        self.best_score = None
        self.early_stop = [False] * experts_num
        self.val_loss_min = [np.inf] * experts_num
        self.delta = delta

    def __call__(self, val_loss, model_list, path_name):
        '''
        在每次调用时检查验证集损失的变化，并决定是否触发早停：
        score = [-val_loss[n] for n in range(self.experts_num)]：计算每个模型的损失的负值，因为我们通常希望最小化损失，
        但为了便于比较，我们将其转换为负值（大的损失值对应小的负值）。score 存储了每个模型当前的损失值。
        if self.best_score is None：在第一次调用时，best_score 为 None，这时将当前的验证损失设置为最优损失，并保存模型。
        else：对于后续的调用，检查每个模型的损失变化：
        如果当前损失小于 best_score + delta，则表示损失没有足够大的改进，增加 counter，并输出当前的 counter 值。
        如果 counter 达到 patience，则认为训练停止，设置 early_stop[n] 为 True。
        如果损失为 NaN，也将增加 counter 并判断是否早停。
        如果损失有改善（即当前损失小于 best_score），则更新 best_score，保存模型，并重置 counter。
        '''
        score = [-val_loss[n] for n in range(self.experts_num)]
        if self.best_score is None:
            for n in range(self.experts_num):
                self.best_score = score
                self.save_checkpoint(val_loss, model_list, path_name, n)
        else:
            for n in range(self.experts_num):
                if (score[n] < self.best_score[n] + self.delta):
                    self.counter[n] += 1
                    self.logger.info(f'EarlyStopping counter: {self.counter[n]} out of {self.patience}')
                    if self.counter[n] >= self.patience:
                        self.early_stop[n] = True
                elif np.isnan(val_loss[n].cpu()):
                    self.counter[n] += 1
                    self.logger.info(f'EarlyStopping counter: {self.counter[n]} out of {self.patience}')
                    if self.counter[n] >= self.patience:
                        self.early_stop[n] = True
                elif self.early_stop[n] == False:
                    self.best_score[n] = score[n]
                    self.save_checkpoint(val_loss, model_list, path_name,n)
                    self.val_loss_min[n] = val_loss[n]
                    self.counter[n] = 0

    def save_checkpoint(self, val_loss, model_list, path_name,model_id):
        if self.verbose:
            self.logger.info(f'Validation loss decreased ({self.val_loss_min[model_id]:.6f} --> {val_loss[model_id]:.6f}).  Saving model {model_id:.1f}...')
        path_name_model = path_name + '_' + str(model_id) + '.pth'
        path = os.path.join(self.save_path, path_name_model)
        model_state_dict = model_list[model_id].state_dict()
        torch.save(model_state_dict, path)	
        self.val_loss_min[model_id] = val_loss[model_id]


# 加权皮尔逊相关系数损失函数
class PartialCosLoss(nn.Module):
    def __init__(self):
        super(PartialCosLoss, self).__init__()

    # 加权皮尔逊相关系数（Weighted Pearson Correlation Coefficient, WPCC）
    def wpcc_org(self, preds, y, amount):
        preds = preds.unsqueeze(-1)
        y = y.unsqueeze(-1)
        nan_index = torch.isnan(y[:, 0])
        y, preds, amount = y[~nan_index], preds[~nan_index], amount[~nan_index]  # 去除空值
        _, argsort = torch.sort(amount, descending=True, dim=0)
        argsort = argsort.squeeze()
        # 排序靠前的样本会拥有较大的权重，排序靠后的样本会拥有较小的权重
        weight = torch.zeros(preds.shape, device=preds.device)
        weight_new = torch.tensor([0.5 ** ((i - 1) / (preds.shape[0] - 1)) for i in range(1, preds.shape[0] + 1)],
                                device=preds.device).unsqueeze(dim=1)
        weight[argsort, :] = weight_new
        weight = weight.to(y.device)
        # WPCC = wcov / (pred_std * y_std)
        wcov = (preds * y * weight).sum(dim=0) / weight.sum(dim=0) \
            - ((preds * weight).sum(dim=0) / weight.sum(dim=0)) * ((y * weight).sum(dim=0) / weight.sum(dim=0))
        pred_std = torch.sqrt(((preds - preds.mean(dim=0)) ** 2 * weight).sum(dim=0) / weight.sum(dim=0))
        y_std = torch.sqrt(((y - y.mean(dim=0)) ** 2 * weight).sum(dim=0) / weight.sum(dim=0))
        return 1 - (wcov / (pred_std * y_std)).mean()

    def forward(self, output, target):
        amount = target[:, 4].float().to(output.device)
        target = target[:, 0].float().to(output.device)
        # print(self.wpcc_org(output, target, amount))
        return self.wpcc_org(output, target, amount)  # amount

# 既可以调整成损失又可以当验证集等指标的模拟交易收益，这个目前是在测试集每周选一次模型中用到
def simu_trade(output, target):
    capital = 5e8                                                           # 总资金
    target = target.to(output.device)
    buyable_amount = target[:, 4].unsqueeze(-1).float().to(output.device)   # 每只股票最大可买入的资金数额
    true_yields = target[:, 0].unsqueeze(-1).float().to(output.device)      # 每只股票的真实收益率
    predicted_yields = output.unsqueeze(-1).float()                         # 模型预测的收益率
    valid_mask = ~torch.isnan(buyable_amount) & ~torch.isnan(true_yields)   # 过滤掉缺失值对应的数据
    buyable_amount = buyable_amount[valid_mask]
    true_yields = true_yields[valid_mask]
    predicted_yields = predicted_yields[valid_mask]
    top500_values, _ = torch.topk(predicted_yields, 500, largest=True, sorted=True)
    value_500 = top500_values[-1]
    buy_amount = buyable_amount[predicted_yields >= value_500]
    true_yields = true_yields[predicted_yields >= value_500]
    total_profit = torch.sum(buy_amount * true_yields) / capital            # 计算总收益率：股票收益总和 / 总资金
    return total_profit

# 训练函数
def backward_and_step(n, optimizer, batch_x, batch_group, batch_y, model, early_stop, val_loss_min, wpcc):
    '''
    训练一个step的函数，带L1正则化，这个正则化系数lambda是比较早调优的结果
    对于不同因子个数情况下以及可能被mask成0的因子比例可能需要用不同大小的lambda
    para n: 当前训练的步数或迭代索引。
    para optimizer: 用于优化模型参数的优化器（例如 Adam）。
    para batch_x: 当前批次的输入数据。
    TODO: 增加使用para batch_group数据。
    para batch_y: 当前批次的目标标签。
    para model: 训练的模型（例如神经网络）。
    para early_stop: 早期停止的标志，用于在训练达到一定标准后提前停止训练。
    para val_loss_min: 存储每个训练阶段的最小验证损失。
    '''
    lambda_ = 0.1
    if early_stop[n] == False:
        # SAM优化器：第一步
        outputs = model(batch_x)
        loss = wpcc(outputs, batch_y)
        return_loss = loss.clone()
        loss = loss + lambda_ * sum(torch.abs(param).sum() for param in model.parameters()) / sum(p.numel() for p in model.parameters())
        loss_grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        for param, grad in zip(model.parameters(), loss_grad):
            param.grad = grad
        optimizer.first_step(zero_grad=True)
        # SAM优化器：第二步
        outputs_second = model(batch_x)
        loss_second = wpcc(outputs_second, batch_y)
        return_loss = loss_second.clone()
        loss_second = loss_second + lambda_ * sum(torch.abs(param).sum() for param in model.parameters()) / sum(p.numel() for p in model.parameters())
        loss_grad_second = torch.autograd.grad(loss_second, model.parameters(), retain_graph=True)
        for param, grad_second in zip(model.parameters(), loss_grad_second):
            param.grad = grad_second
        optimizer.second_step(zero_grad=True)
    else:
        return_loss = val_loss_min[n]
    return return_loss.to(torch.device("cuda:0"))

# 辅助因子平衡K均值聚类
class BalancedKMeans:
    def __init__(self, n_clusters, random_state=42, max_iter=200):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        
    def fit_predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        max_size = (n_samples + self.n_clusters - 1) // self.n_clusters
        np.random.seed(self.random_state)
        centers = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            dists = np.linalg.norm(X[:, None] - centers, axis=2)
            clusters = -np.ones(n_samples, dtype=int)
            for i in np.argsort(np.min(dists, axis=1)):
                closest = np.argmin(dists[i])
                if np.sum(clusters == closest) < max_size:
                    clusters[i] = closest
                else:
                    mask = np.arange(self.n_clusters) != closest
                    valid_centers = np.where(mask)[0]
                    next_best = valid_centers[np.argmin(dists[i, mask])]
                    clusters[i] = next_best
            new_centers = np.array([X[clusters == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        return clusters

# 基于多种距离度量的辅助因子平衡K均值聚类
class BalancedKMeansMetric:
    def __init__(self, n_clusters, metric='euclidean', random_state=42, max_iter=200, covariance_matrix=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state
        self.max_iter = max_iter
        self.covariance_matrix = covariance_matrix
        
    def fit_predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        max_size = (n_samples + self.n_clusters - 1) // self.n_clusters
        np.random.seed(self.random_state)
        centers = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        # 选择不同的度量方法
        if self.metric == 'euclidean':
            dist_func = lambda x, c: np.linalg.norm(x - c, axis=1)
        elif self.metric == 'manhattan':
            dist_func = lambda x, c: np.sum(np.abs(x - c), axis=1)
        elif self.metric == 'cosine':
            dist_func = lambda x, c: 1 - np.dot(x, c) / (np.linalg.norm(x) * np.linalg.norm(c))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        for _ in range(self.max_iter):
            # 计算度量矩阵
            similarities = np.array([dist_func(X, center) for center in centers]).T
            clusters = -np.ones(n_samples, dtype=int)
            for i in np.argsort(np.min(similarities, axis=1)):
                closest = np.argmin(similarities[i])
                if np.sum(clusters == closest) < max_size:
                    clusters[i] = closest
                else:
                    mask = np.arange(self.n_clusters) != closest
                    valid_centers = np.where(mask)[0]
                    next_best = valid_centers[np.argmin(similarities[i, mask])]
                    clusters[i] = next_best
            # 更新聚类中心
            new_centers = np.array([X[clusters == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        return clusters
    