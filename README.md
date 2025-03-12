**主要更新部分：**

20250219:

1. 辅助因子的平衡K均值聚类方法的实现

---

20250220:

1. 辅助因子的平衡K均值聚类方法的实现

2. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

---

20250221:

1. 辅助因子的平衡K均值聚类方法的实现

2. 聚类度量为曼哈顿距离

---

20250222:

1. 辅助因子的平衡K均值聚类方法的实现

2. 聚类度量为相关系数

---

20250224:

1. 辅助因子的平衡K均值聚类方法的实现

2. 聚类度量为相关系数

3. 修改主要因子和辅助因子的模型结构（主要因子单独使用ResNet+主要&辅助因子合并使用ResNet）

---

20250225:

1. 辅助因子的平衡K均值聚类方法的实现

2. 修改聚类方式为均衡聚类，即将距离近的特征分散在不同组

---

20250227:

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

---

20250228:

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 按照讨论思路将全部因子进行聚类训练多个模型

---

20250302:

1. 辅助因子的平衡K均值聚类方法的实现

2. 按照讨论思路将全部因子进行聚类训练多个模型

---

20250303:

1. 按照讨论思路将全部因子进行聚类训练多个模型

---

20250304:

1. 修改主要因子和辅助因子的模型结构（主要因子单独使用ResNet+主要&辅助因子合并使用ResNet）

---

20250305:

1. 保留ic最大的前500个主要因子，将其余750个主要因子和辅助因子一起聚类

---

20250306:

1. 修改simu_trade为可反向传播的损失函数

---

20250307:

1. 辅助因子的平衡K均值聚类方法的实现

2. 修改simu_trade为可反向传播的损失函数

---

20250308(with1209):

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 修改主要因子和辅助因子的模型结构（主要因子单独使用ResNet+主要&辅助因子合并使用ResNet）

---

20250309:

1. 辅助因子的平衡K均值聚类方法的实现

2. 修改主要因子和辅助因子的模型结构（主要因子单独使用ResNet+主要&辅助因子合并使用ResNet）

---

20250310:

1. 修改simu_trade为可前1000个股票的损失函数

---

20250311:

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 交替使用WPCC和simu_trade进行训练

---

20250312:

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 使用WPCC和simu_trade归一化损失加权平均进行训练

---

20250313:

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 使用WPCC和simu_trade归一化损失梯度均衡进行训练

---

20250314:

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 修改主要因子和辅助因子的模型结构（主要因子单独使用ResNet+主要&辅助因子合并使用ResNet+对称交叉注意力机制）

---

20250315:

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 修改主要因子和辅助因子的模型结构（主要因子单独使用ResNet+主要&辅助因子合并使用ResNet+非对称交叉注意力机制+动态融合）

---

20250316:

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 修改主要因子和辅助因子的模型结构（主要因子+主要&辅助因子层级金字塔模型）

---

20250317:

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 修改主要因子和辅助因子的模型结构（主要因子单独使用ResNet+主要&辅助因子合并使用ResNet+动态门控融合网络）

---

20250318:

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 修改主要因子和辅助因子的模型结构（主要因子单独使用ResNet+主要&辅助因子合并使用ResNet+特征重组网络）

---

20250319:

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 修改主要因子和辅助因子的模型结构（主要因子单独使用ResNet+主要&辅助因子合并使用ResNet）

3. 在模型输出端增加因子的图结构

---

20250320:

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 修改主要因子和辅助因子的模型结构（主要因子单独使用ResNet+主要&辅助因子合并使用ResNet）

3. 在模型输入端增加因子的图结构

---

20250321:

1. 修改WPCC损失函数为成交量加权：self.wpcc_org(output, target, amount)

2. 交替使用WPCC、simu_trade和excess_loss进行训练（3:1:1）

---

