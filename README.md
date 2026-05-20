# README

## 1. 环境安装

- Python >= 3.7
- 安装依赖：
  pip install jittor jittor_geometric numpy
- 或使用 requirements.txt (内容: jittor>=1.3.0, jittor_geometric, numpy)

## 2. 数据准备

- 下载 cora.pkl 文件（比赛页面获取）
- 创建 data/ 文件夹，将 cora.pkl 放入其中：
  mkdir data
  cp /path/to/cora.pkl data/
- 代码默认读取 data/cora.pkl，无需额外配置

## 3. 训练

运行命令：
python gcn.py

超参数已内置：epochs=200, lr=0.01, hidden_dim=256, dropout=0.8

## 4. 评测/推理

推理与训练合并执行，训练结束后自动生成 result.json：
python gcn.py

如需单独加载模型进行推理，可修改代码添加 checkpoint 保存与加载逻辑（当前版本未实现）

## 5. 结果说明

- 指标：分类准确率（Accuracy）= 正确预测节点数 / 总节点数
- 计算方式：(pred == data.y[mask]).float().mean()
- 线上提交成绩可能与本地验证集存在差异，可能原因：
  - 本地验证集划分与线上测试集不同
  - 随机种子或环境差异导致训练波动
  - 未保存最佳模型，使用最终 epoch 输出
