# NLP-Beginner Tasks（Task-1 / Task-2 / Task-3）

本仓库包含三个阶段任务：

- Task-1：传统机器学习文本分类（手写线性分类器）
- Task-2：深度学习文本分类（CNN / RNN / TransformerEncoder）
- Task-3：Transformer 基础结构与实验（手写多头注意力，支持 RMSNorm / RoPE）

默认数据文件：

- data/new_train.tsv
- data/new_test.tsv

TSV 兼容格式：

1) 有表头：`text/sentence/review` + `label/sentiment/score`
2) 无表头：第 1 列文本，第 2 列标签

---

## 1. 目录与代码文件说明

### 1.1 Task-1（task1/textcls）

- task1/textcls/data.py：读取数据、划分 train/val/test
- task1/textcls/features.py：`CountVectorizer`，支持 BoW / N-gram
- task1/textcls/model.py：手写 `LinearClassifier`，含前向、损失、梯度与更新
- task1/textcls/train.py：训练主流程，输出 JSON 结果
- task1/textcls/experiments.py：批量实验与柱状图绘制

### 1.2 Task-2（task2/textdl）

- task2/textdl/data.py：分词、词表、padding、DataLoader
- task2/textdl/models.py：`TextCNN` / `TextRNN` / `TextTransformer`
- task2/textdl/train.py：训练入口，支持多损失、多优化器、GloVe、GPU
- task2/textdl/experiments.py：系统化实验与图表输出

### 1.3 Task-3（task3/transformer_basics）

- task3/transformer_basics/data_math.py：多位数加法数据构造与 DataLoader
- task3/transformer_basics/data_lm.py：合成语料、Tokenizer、LM 数据切分
- task3/transformer_basics/models.py：
	- 手写 `MultiHeadAttention`
	- `RMSNorm`
	- `RotaryEmbedding`（RoPE）
	- `Seq2SeqTransformer` / `DecoderOnlyTransformer`
- task3/transformer_basics/train_math.py：数学加法任务训练与 exact-match 评测
- task3/transformer_basics/train_lm.py：语言模型训练与 perplexity 评测
- task3/transformer_basics/experiments.py：Task-3 各模块对比实验（结构、Norm、位置编码、Tokenizer、词表）

---

## 2. 环境安装与准备

1. 安装依赖

	 `uv sync`

2. 查看可用脚本

	 - `uv run task1-train --help`
	 - `uv run task2-train --help`

> Task-3 当前使用模块方式运行（`python -m ...`）。

---

## 3. 如何跑通各任务

### 3.1 Task-1 跑通

单次训练：

`uv run task1-train --epochs 2 --ngram 1 --loss ce --lr 0.1`

批量实验：

`uv run task1-exp`

输出目录：task1/outputs

---

### 3.2 Task-2 跑通

CNN：

`uv run task2-train --model cnn --epochs 2 --device auto`

RNN：

`uv run task2-train --model rnn --rnn-type lstm --epochs 2 --device auto`

Transformer：

`uv run task2-train --model transformer --embedding-dim 192 --nhead 6 --epochs 2 --device auto`

批量实验：

`uv run task2-exp`

输出目录：task2/outputs

---

### 3.3 Task-3 跑通

数学任务（Seq2Seq，默认配置）：

`uv run python -m task3.transformer_basics.train_math`

语言模型任务（Decoder-only，默认配置）：

`uv run python -m task3.transformer_basics.train_lm`

Task-3 全部实验：

`uv run python -m task3.transformer_basics.experiments`

输出目录：task3/outputs/math 与 task3/outputs/lm

---

## 4. 如何测试各代码（建议流程）

### 4.1 快速功能测试（Smoke Test）

目标：确认每个任务都可训练 1~2 个 epoch 并正常产出文件。

- Task-1：运行 `task1-train`，检查 task1/outputs 下是否生成 `result_*.json`
- Task-2：运行 `task2-train`，检查 task2/outputs 下是否生成 `*.pt` 与 `*.json`
- Task-3：运行 `train_math` / `train_lm`，检查 task3/outputs 下是否生成 `*.pt` 与 `*.json`

### 4.2 实验结果测试

- Task-1：运行 `task1-exp`，检查：
	- task1/outputs/experiment_summary.csv
	- task1/outputs/experiment_test_acc.png

- Task-2：运行 `task2-exp`，检查：
	- task2/outputs/experiment_summary.csv
	- task2/outputs/all_test_acc.png
	- task2/outputs/model_compare.png

- Task-3：运行 `python -m task3.transformer_basics.experiments`，检查：
	- task3/outputs/math/experiment_summary.csv
	- task3/outputs/math/test_exact.png
	- task3/outputs/lm/experiment_summary.csv
	- task3/outputs/lm/test_ppl.png

### 4.3 指标解读

- 分类任务（Task-1/2）：主要看 `test_acc`
- 数学生成任务（Task-3 math）：主要看 `test_exact`
- 语言模型任务（Task-3 lm）：主要看 `test_ppl`（越低越好）

---

## 5. 各任务实验说明（建议复现实验）

### 5.1 Task-1（传统方法）

建议比较：

- N-gram：1 vs 2
- 损失：CE vs MSE
- 学习率：0.1 vs 0.05

目标：观察特征工程与优化配置对准确率影响。

### 5.2 Task-2（深度学习分类）

建议比较：

- 模型：CNN vs RNN vs Transformer
- 损失：CE / Focal / MSE
- 优化器：Adam / AdamW / SGD
- 卷积核组合、GloVe 初始化与是否冻结

目标：观察架构与训练策略对分类效果影响。

### 5.3 Task-3（Transformer 基础结构）

建议比较：

- 结构：`seq2seq` vs `decoder_only`
- 归一化：`layernorm` vs `rmsnorm`
- 位置编码：`sinusoidal` vs `rope`（或 `none`）
- 数学任务数据划分：同分布测试 vs OOD（如训练 3+3/3+4/4+3/4+4，测试 3+5/5+3）
- LM 任务：`char` vs `word` tokenizer，词表大小变化

目标：验证模型泛化性与模块设计对训练稳定性、性能的影响。

---

## 6. 常见问题

- 如果编辑器报 “无法解析导入 torch/pandas/matplotlib”，通常是 Python 解释器未切到当前环境。
- 先执行 `uv sync`，再在编辑器中选择项目环境解释器。
- 如果使用 GPU，训练命令中建议设置 `--device auto`。
