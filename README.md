# NLP-Beginner Task-1（基于机器学习的文本分类）

已按要求在 `NLP-Beginner` 下构建可直接用 `uv` 管理的项目，并预留 `data/` 目录。

## 1. 项目结构

- `data/`：放置你稍后上传的数据文件（`new_train.tsv`、`new_test.tsv`）
- `src/textcls/data.py`：读取 TSV 并划分训练/验证集
- `src/textcls/features.py`：手写 BoW/N-gram 向量化
- `src/textcls/model.py`：手写线性分类器 + 手写损失与梯度（不使用 `torch.nn`）
- `src/textcls/train.py`：mini-batch 训练与评估
- `src/textcls/experiments.py`：BoW vs N-gram、loss/lr 对比实验与绘图
- `outputs/`：训练输出与实验图表（默认忽略提交）

## 2. 使用 uv 初始化环境

在项目根目录执行：

1. `uv sync`
2. `uv run python -m textcls.train --help`

> 如需 Notebook：`uv run jupyter notebook`

## 3. 数据格式要求

放入 `data/` 后，训练脚本默认读取：

- `data/new_train.tsv`
- `data/new_test.tsv`

文件需包含：

- 文本列：`text` / `sentence` / `review` 之一
- 标签列：`label` / `sentiment` / `score` 之一

标签应为整数（你的修订数据为 0~4）。

## 4. 训练示例

### 4.1 Bag of Words + Cross Entropy

`uv run python -m textcls.train --ngram 1 --loss ce --lr 0.1 --epochs 12`

### 4.2 N-gram(1+2) + Cross Entropy

`uv run python -m textcls.train --ngram 2 --loss ce --lr 0.1 --epochs 12`

### 4.3 MSE 损失对比

`uv run python -m textcls.train --ngram 1 --loss mse --lr 0.1 --epochs 12`

## 5. 一键实验

`uv run python -m textcls.experiments`

输出：

- `outputs/experiment_summary.csv`
- `outputs/experiment_test_acc.png`

## 6. 实现说明（符合任务约束）

- 使用 `torch` 仅做张量和矩阵运算
- 不调用 `torch.nn` 中封装好的网络函数
- 使用 mini-batch 矩阵化训练
- 训练中可观察 loss 与验证集准确率
- 训练后在测试集评估
