# GCN 节点分类热身赛（Cora 数据集）

基于图卷积网络（GCN）对 Cora 论文引用数据集进行节点分类的入门示例。  
本项目使用 Jittor 深度学习框架与 jittor_geometric 图神经网络库实现。

## 任务描述

- 数据集：Cora（2708 个节点，5429 条边，每个节点为 1433 维词袋特征，共 7 个类别）
- 目标：利用图结构信息，预测测试集中节点的类别
- 评价指标：分类准确率（Accuracy）

## 环境要求

- Python 3.7+
- Jittor >= 1.3.0
- jittor_geometric（需安装，提供 GCNConv, cootocsc, cootocsr 等）
- numpy

推荐使用 GPU 运行（代码中已开启 jt.flags.use_cuda = 1）。

## 数据准备

将预处理好的 cora.pkl 文件放置在 data/ 目录下。  
该文件应包含以下键值（均为 numpy 数组）：

- x: 特征矩阵，shape (N, 1433)
- y: 标签，shape (N,)
- edge_index: 边索引，shape (2, E)
- train_mask: 训练集掩码，shape (N,)
- val_mask: 验证集掩码，shape (N,)
- test_mask: 测试集掩码，shape (N,)
- num_features: 特征维度
- num_classes: 类别数

若未提供，可自行从原始 Cora 数据集生成。

## 使用方法

1. 将代码保存为 gcn.py
2. 在终端中执行：python gcn.py

训练过程会在控制台输出每 20 个 epoch 的训练集准确率与验证集最佳准确率，例如：

Epoch: 020, Train Acc: 0.9857, Val Acc: 0.7420
...
最终结果: Val Acc: 0.7640

3. 运行结束后，会生成 result.json 文件，内容为测试集中每个节点 id 到预测类别的映射。

## 模型结构

- 两层图卷积层（GCNConv）
- 隐藏层维度：256
- 激活函数：ReLU
- Dropout：0.8（仅训练时启用）
- 输出层：7 维 logits

### 超参数

| 参数 | 值 |
| ---- | -- |
| 学习率 | 0.01 |
| 权重衰减 | 5e-4 |
| 优化器 | Adam |
| 训练轮数 | 200 |
| 自环 + 归一化 | 启用 |

## 代码结构说明

- 数据加载：从 cora.pkl 读取并转为 Jittor 张量，对特征做行归一化。
- 边归一化：调用 gcn_norm 添加自环并进行对称归一化，再将边索引转换为 CSC/CSR 格式，用于高效稀疏矩阵乘法。
- 模型定义：GCNNet 类继承 nn.Module，在 execute 中依次进行两次卷积 + ReLU + Dropout。
- 训练/测试：train() 执行单次前向传播并计算交叉熵损失；test() 返回训练集和验证集的准确率。
- 主循环：迭代 200 个 epoch，记录最佳验证准确率。
- 结果保存：对测试集节点进行预测，输出 result.json。

## 注意事项

- 代码假定 GPU 可用，若无 GPU 可修改 jt.flags.use_cuda = 0。
- Dropout 概率较高（0.8），可根据验证集表现调整。
- 如需复现结果，已设置随机种子 jt.misc.set_global_seed(42)。
- 生成的 result.json 格式示例：{"0": 3, "1": 5, ...}

## 参考

- Jittor: https://github.com/Jittor/jittor
- Jittor Geometric: https://github.com/Jittor/jittor-geometric
- Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
