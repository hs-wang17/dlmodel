**主要更新部分：**

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 使用WPCC和simu_trade归一化损失加权平均进行训练
