**主要更新部分：**

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 修改主要因子和辅助因子的模型结构（主要因子单独使用ResNet+主要&辅助因子合并使用ResNet+特征重组网络）

