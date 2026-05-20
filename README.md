# README

## 1. 环境安装

- Python >= 3.10
- 安装依赖：
- python -m pip install git+https://github.com/Jittor/jittor.git
- pip install requirements.txt
- git clone https://github.com/AlgRUC/JittorGeometric.git
- cd JittorGeometric
- pip install .

## 2. 数据准备

- 下载 cora.pkl 文件（比赛页面获取）
- 创建 data/ 文件夹，将 cora.pkl 放入其中：
  mkdir data
  cp /path/to/cora.pkl data/
- 代码默认读取 data/cora.pkl，无需额外配置

## 3. 训练

运行命令：
python gcn.py

超参数已内置：epochs=200, lr=0.01, hidden_dim=256, dropout=0.8，已经设置jt.misc.set_global_seed(42)
由于是热身赛，无需修改超参数，省略了config文件和seed设置


## 4. 评测/推理

推理与训练合并执行，训练结束后自动生成 result.json：
python gcn.py


## 5. 结果说明

- 指标：分类准确率（Accuracy）= 正确预测节点数 / 总节点数 =0.808
- 计算方式：(pred == data.y[mask]).float().mean()

