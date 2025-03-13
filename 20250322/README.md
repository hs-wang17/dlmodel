**主要更新部分：**

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 交替使用WPCC、simu_trade和excess_loss进行训练（3:1:1）
