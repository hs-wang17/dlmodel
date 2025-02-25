import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # 如果输入输出维度不同，需要projection
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
        x = F.relu(x)
        return x

class base_model(nn.Module):
    def __init__(self, output_size=1, drop_out=0.2, mask=None, factor_list=None, seed=1):
        super(base_model, self).__init__()
        ic_mask = mask.to(dtype=torch.bool)
        factor_mask = torch.zeros(ic_mask.shape, dtype=torch.bool).to(ic_mask.device)
        factor_mask[factor_list] = True
        final_mask = ic_mask & factor_mask                          # final_mask是ic_mask和factor_mask的交集
        self.selected_factor = torch.nonzero(final_mask).squeeze()  # 获取final_mask的索引
        # 主要特征和次要特征的分离
        self.main_features = factor_list[:1251]                      # 主要特征是factor_list中的前1250个
        self.secondary_features = factor_list[1251:]                 # 次要特征是factor_list中的1250以后的部分
        # 主要特征处理
        self.input_bn_main = nn.BatchNorm1d(self.main_features.shape[0])
        self.input_linear_main = nn.Linear(self.main_features.shape[0], 512)
        self.res_blocks_main = nn.Sequential(
            ResBlock(512, 512),
            ResBlock(512, 256),
            ResBlock(256, 128),
            ResBlock(128, 64)
        )
        # 主要特征的输出
        self.output_main = nn.Sequential(
            nn.Dropout(drop_out),
            nn.Linear(64, 16)
        )
        # 主要特征和次要特征结合的处理
        self.input_bn_combined = nn.BatchNorm1d(self.selected_factor.shape[0])
        self.input_linear_combined = nn.Linear(self.selected_factor.shape[0], 512)
        self.res_blocks_combined = nn.Sequential(
            ResBlock(512, 512),
            ResBlock(512, 256),
            ResBlock(256, 128),
            ResBlock(128, 64)
        )
        # 合并后的输出
        self.output_combined = nn.Sequential(
            nn.Dropout(drop_out),
            nn.Linear(64, 16)
        )
        # 新增的一层神经网络，用来合并output_main和output_combined
        self.merge_layer = nn.Linear(32, output_size)   # 输入是两个输出，合并后得到最终输出
        self._initialize_weights(seed)

    def seed_everything(self, seed):
        random.seed(seed)                               # 设置随机种子
        np.random.seed(seed)                            # 设置NumPy的随机种子
        torch.manual_seed(seed)                         # 设置PyTorch的随机种子
        torch.cuda.manual_seed(seed)                    # 设置CUDA的随机种子
        torch.cuda.manual_seed_all(seed)                # 设置所有CUDA设备的随机种子
        torch.backends.cudnn.deterministic = True       # 确保CUDA的行为是确定的
        torch.backends.cudnn.benchmark = False          # 禁用CUDA的自动优化

    def _initialize_weights(self, seed):
        self.seed_everything(seed)                      # 用固定种子初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)       # 对全连接层进行Kaiming Normal初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)        # 初始化偏置为0
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)          # 批量归一化的权重初始化为1
                nn.init.constant_(m.bias, 0)            # 批量归一化的偏置初始化为0

    def forward(self, x):
        x = x.to(self.selected_factor.device)           # 将输入移动到正确的设备
        x_main = x[:, self.main_features]               # 提取主要特征（factor_list中的前1250个）
        x_combined = x[:, self.selected_factor]
        # 主要特征的处理
        x_main = self.input_bn_main(x_main)
        x_main = F.relu(self.input_linear_main(x_main))
        x_main = self.res_blocks_main(x_main)
        output_main = self.output_main(x_main).squeeze(-1)
        # 主要特征和次要特征的结合处理
        x_combined = self.input_bn_combined(x_combined)
        x_combined = F.relu(self.input_linear_combined(x_combined))
        x_combined = self.res_blocks_combined(x_combined)
        output_combined = self.output_combined(x_combined).squeeze(-1)
        # 使用merge_layer合并两个输出
        output = torch.cat((output_main, output_combined), dim=1)
        final_output = self.merge_layer(output).squeeze(-1)
        return final_output


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
        return self.wpcc_org(output, target, output)  # amount

# 既可以调整成损失又可以当验证集等指标的模拟交易收益，这个目前是在测试集每周选一次模型中用到
def simu_trade(output, target):
    capital = 5e8  # 总资金
    target = target.to(output.device)
    buyable_amount = target[:, 4].unsqueeze(-1).float().to(output.device)  # 每只股票最大可买入的资金数额
    true_yields = target[:, 0].unsqueeze(-1).float().to(output.device)  # 每只股票的真实收益率
    predicted_yields = output.unsqueeze(-1).float()  # 模型预测的收益率
    # 过滤掉缺失值对应的数据
    valid_mask = ~torch.isnan(buyable_amount) & ~torch.isnan(true_yields)
    buyable_amount = buyable_amount[valid_mask]
    true_yields = true_yields[valid_mask]
    predicted_yields = predicted_yields[valid_mask]
    top500_values, _ = torch.topk(predicted_yields, 500, largest=True, sorted=True)
    value_500 = top500_values[-1]
    buy_amount = buyable_amount[predicted_yields >= value_500]
    true_yields = true_yields[predicted_yields >= value_500]
    total_profit = torch.sum(buy_amount * true_yields) / capital  # 计算总收益率：股票收益总和 / 总资金
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
    