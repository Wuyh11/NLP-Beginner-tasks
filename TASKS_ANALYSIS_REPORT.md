# NLP-Beginner 三个任务综合分析报告

> 项目：Task-1 / Task-2 / Task-3  
> 日期：2026-03-18

---

## 1. 总览

本仓库按“从传统方法到深度学习再到 Transformer 机制细化”的学习路径设计：

- **Task-1**：手写线性分类器 + BoW/N-gram（传统 NLP 基线）
- **Task-2**：CNN/RNN/TransformerEncoder（深度学习文本分类）
- **Task-3**：手写 Transformer 关键模块（MHA、RMSNorm、RoPE）并做模块对比实验

从教学价值看，这三个任务分别对应：

1. **可解释基线**（特征工程 + 线性模型）
2. **神经网络建模能力**（序列建模/局部模式/全局注意力）
3. **架构级理解与实验能力**（模块替换、归一化、位置编码、OOD 泛化）

---

## 2. Task-1：传统文本分类（BoW / N-gram + 手写线性分类器）

### 2.1 原理

Task-1 的核心思想是把文本变成高维稀疏向量，然后做线性分类：

1. **特征表示**：
   - unigram 或 1~n gram 计数特征（BoW / N-gram）
   - 每个维度表示某个词/短语出现次数
2. **分类器**：
   - 线性打分：$z = XW + b$
   - 用 softmax + CE（或 MSE）优化
3. **优化方式**：
   - 手写梯度 + SGD 更新

这是一条非常经典且强可解释的文本分类路线。

### 2.2 流程

1. 读入 TSV 数据并统一列名（兼容有表头/无表头）
2. 切分 train/val/test
3. 在 train 上构建词表并向量化
4. mini-batch 训练线性模型
5. 定期在 val 上评估，最后在 test 上汇报准确率
6. 输出 JSON 结果

### 2.3 如何实现（代码视角）

- 数据读取与切分：task1/textcls/data.py
- 向量化：task1/textcls/features.py（`CountVectorizer`）
- 模型：task1/textcls/model.py（`LinearClassifier`，手写 CE/MSE 梯度）
- 训练：task1/textcls/train.py
- 批量实验：task1/textcls/experiments.py

关键实现点：

- 通过 `Counter` 统计 token 频率并按 `min_freq/max_features` 截断词表
- batch 向量化时构建稀疏计数张量
- CE 与 MSE 两种损失都提供了显式梯度计算

### 2.4 实现效果怎么样

- **优势**：
  - 训练快、可解释强、调试难度低
  - 小数据集上往往能得到稳定 baseline
- **局限**：
  - 无上下文语义，无法建模长距离依赖
  - OOV 与同义表达泛化能力弱

在教学场景中，Task-1 的价值主要是建立“特征-模型-优化”闭环认知，而非追求 SOTA。

### 2.5 对比效果如何（Task-1 内部）

典型对比趋势（经验规律）：

- **N-gram 2 往往优于纯 unigram**（可捕获短语信息），但词表更大
- **CE 通常优于 MSE**（分类任务更匹配概率建模）
- 学习率较大时收敛快但不稳定，较小更稳但训练慢

### 2.6 可以如何优化

1. 特征层：TF-IDF、子词特征、字词混合特征
2. 训练层：学习率衰减、早停、类别不平衡重加权
3. 数据层：文本清洗、噪声样本过滤、数据增强（同义替换等）

### 2.7 面试可能会问

1. CE 和 MSE 用于分类的本质区别？为什么 CE 更常用？
2. BoW 为什么丢失词序信息？N-gram 如何补偿？
3. 稀疏高维特征会导致什么问题（过拟合、存储）？
4. 手写梯度时如何验证梯度正确性（数值梯度检查）？

---

## 3. Task-2：深度学习文本分类（CNN / RNN / Transformer）

### 3.1 原理

Task-2 进入神经网络范式，核心变化是“**自动学习表示**”：

1. 文本 → token id → embedding
2. 通过神经网络编码语义：
   - CNN：局部 n-gram 模式
   - RNN：顺序依赖
   - TransformerEncoder：全局自注意力
3. 分类头输出 logits，使用 CE/Focal/MSE 优化

### 3.2 流程

1. 构建词表（`<pad>/<unk>`）
2. 文本编码并动态 padding
3. 前向计算 logits
4. 计算损失并反向传播
5. 验证集选最优权重，最终测试
6. 保存 `.pt` 与 `.json`

### 3.3 如何实现（代码视角）

- 数据与 DataLoader：task2/textdl/data.py
- 模型定义：task2/textdl/models.py
  - `TextCNN`
  - `TextRNN`（LSTM/GRU + pack）
  - `TextTransformer`（Encoder + mean pooling）
- 训练入口：task2/textdl/train.py
  - 可选优化器：Adam / AdamW / SGD
  - 可选损失：CE / Focal / MSE
  - 可选 GloVe 初始化 + 是否冻结 embedding
- 实验脚本：task2/textdl/experiments.py

### 3.4 实现效果怎么样

通常可预期：

- **总体性能**：Task-2 明显优于 Task-1（特别是语义表达复杂时）
- **模型差异**：
  - CNN 训练快，效果常作为强基线
  - RNN 对序列依赖友好，但并行性较差
  - Transformer 在中长文本通常更有优势，但更依赖数据量和超参

### 3.5 对比效果如何（Task-2 内部）

1. **损失函数**：
   - CE 是默认强基线
   - Focal 在类别不平衡时可能提升 tail 类
   - MSE 一般不如 CE 稳定
2. **优化器**：
   - Adam/AdamW 收敛快、鲁棒性强
   - SGD 在精心调参后可有竞争力，但早期收敛慢
3. **模型结构**：
   - 短文本：CNN 常有性价比优势
   - 依赖顺序信息较强：RNN 可能更稳
   - 长程依赖：Transformer 更具潜力
4. **GloVe**：
   - 小数据时常能加速收敛并提升泛化
   - 是否冻结 embedding 需要按数据规模与领域差异决定

### 3.6 可以如何优化

1. 训练策略：
   - 学习率调度（cosine/warmup）
   - 梯度裁剪
   - early stopping + model checkpoint
2. 模型策略：
   - CNN 加多尺度卷积 + 残差
   - RNN 加 attention pooling
   - Transformer 加 CLS token + 更合理 pooling
3. 数据策略：
   - 子词切分（BPE/WordPiece）降低 OOV
   - 类别重采样与 hard example mining
4. 工程策略：
   - mixed precision
   - 更完善日志（TensorBoard/W&B）

### 3.7 面试可能会问

1. 为什么要用 `pack_padded_sequence`？
2. Focal Loss 适合什么场景？数学形式是什么？
3. Adam 和 AdamW 的关键差异？
4. CNN/RNN/Transformer 在文本分类中的优缺点对比？
5. 什么时候应该冻结预训练词向量？

---

## 4. Task-3：Transformer 基础结构与模块实验

### 4.1 原理

Task-3 重点从“会用模型”升级到“理解模型机制”。

涉及核心模块：

1. **Multi-Head Attention**：并行子空间注意力
2. **Norm**：LayerNorm vs RMSNorm
3. **位置编码**：Sinusoidal vs RoPE vs None
4. **架构对比**：Seq2Seq vs Decoder-only
5. **指标体系**：
   - 数学任务：Exact Match
   - 语言模型：Perplexity

### 4.2 流程

#### 4.2.1 数学任务（train_math）

1. 按位数模板生成加法数据
2. 训练/验证/测试划分（含 OOD 模板）
3. 选择 `seq2seq` 或 `decoder_only`
4. 训练后按字符级生成答案
5. 计算 exact match（预测值是否完全等于真值）

#### 4.2.2 语言模型任务（train_lm）

1. 生成合成语料
2. char/word tokenizer 构词表
3. 固定窗口 next-token 训练
4. 验证/测试计算 perplexity
5. 给定 prompt 做贪心生成

### 4.3 如何实现（代码视角）

- 数据：
  - task3/transformer_basics/data_math.py
  - task3/transformer_basics/data_lm.py
- 模型：task3/transformer_basics/models.py
  - 手写 `MultiHeadAttention`
  - `RMSNorm`
  - `RotaryEmbedding`
  - `Seq2SeqTransformer` / `DecoderOnlyTransformer`
- 训练：
  - task3/transformer_basics/train_math.py
  - task3/transformer_basics/train_lm.py
- 实验编排：task3/transformer_basics/experiments.py

### 4.4 实现效果怎么样

教学与研究价值很高：

- **可解释性**：可以明确看到 mask、padding mask、残差、norm 的作用位置
- **可实验性**：模块可替换，容易做 controlled experiment
- **泛化观察**：数学任务的 OOD 模板能直观暴露“记忆式拟合”问题

### 4.5 对比效果如何（Task-3 内部）

经验趋势（常见但非绝对）：

1. **数学任务**：
   - in-distribution 明显高于 OOD
   - seq2seq 在结构化映射任务中常更稳
   - RoPE/RMSNorm 在某些配置下可改善长度泛化与训练稳定性
2. **LM 任务**：
   - word tokenizer 一般更利于语义层预测，char 更细粒度但序列更长
   - 合理词表大小有助于平衡 OOV 与稀疏性
   - perplexity 对训练稳定性敏感（优化器、norm、位置编码）

### 4.6 可以如何优化

1. 模型层：
   - Pre-Norm / Post-Norm 结构对比
   - FFN 换 GELU / SwiGLU
   - 增加 dropout 策略
2. 训练层：
   - warmup + cosine decay
   - label smoothing
   - 梯度裁剪与 EMA
3. 数据层：
   - 数学任务增加进位模式覆盖
   - LM 引入更真实语料与验证集清洗
4. 评估层：
   - 数学任务增加按位准确率、进位错误类型分析
   - LM 增加长度分段 PPL 与生成质量人工评测

### 4.7 面试可能会问

1. 为什么 attention 里要做缩放 $1/\sqrt{d_k}$？
2. causal mask 与 key padding mask 有什么区别？
3. LayerNorm 与 RMSNorm 的差异与适用场景？
4. RoPE 的直觉与优势是什么？
5. Seq2Seq 和 Decoder-only 在任务建模上的差异？
6. perplexity 如何计算，和交叉熵关系是什么？

---

## 5. 三个任务横向结论

### 5.1 能力进阶关系

- **Task-1**：建立最小可用分类系统（可解释、快速）
- **Task-2**：引入神经网络表达能力（性能提升主力）
- **Task-3**：深入架构细节与科研型对比实验（理解机制）

### 5.2 效果与成本权衡

| 维度 | Task-1 | Task-2 | Task-3 |
|---|---|---|---|
| 建模能力 | 低-中 | 中-高 | 高（可扩展） |
| 训练成本 | 低 | 中 | 中-高 |
| 可解释性 | 高 | 中 | 中-高（机制级） |
| 工程复杂度 | 低 | 中 | 高 |
| 面试展示力 | 中 | 高 | 很高 |

### 5.3 推荐答辩/面试叙事方式

可按以下主线陈述：

1. 我先用 Task-1 建立可解释 baseline，验证数据和评估流程正确
2. 再用 Task-2 做架构对比，得到更强性能并分析损失函数/优化器影响
3. 最后在 Task-3 手写核心模块，做结构化实验（Norm、Pos、架构、OOD）
4. 结论不仅是“哪个指标高”，而是“为什么高、代价是什么、如何继续优化”

---

## 6. 当前仓库“效果”说明与复现实验建议

当前仓库默认会把实验结果写入：

- `task1/outputs`
- `task2/outputs`
- `task3/outputs/math`
- `task3/outputs/lm`

如果你要得到可直接写进报告的真实对比数值，建议执行：

1. 跑各任务实验脚本产出 `experiment_summary.csv`
2. 把关键表格（best val / test）贴到报告附录
3. 给出“最优配置 + 次优配置 + 失败案例”的分析

---

## 7. 可直接复用的面试问答清单（精选）

### 7.1 基础

1. 为什么分类任务通常用 CE 而不是 MSE？
2. 什么是过拟合？你在这个项目中如何判断过拟合？
3. 为什么要划分验证集，不直接看测试集？

### 7.2 工程

4. 你如何保证实验可复现？
5. 你是如何做模型保存与最优权重选择的？
6. 遇到类别不平衡你会怎么处理？

### 7.3 架构

7. CNN、RNN、Transformer 的归纳偏置差异是什么？
8. Transformer 为什么需要位置编码？
9. RoPE 相比绝对位置编码的优势是什么？

### 7.4 进阶

10. OOD 泛化差时，你会从数据、模型、训练哪三层排查？
11. perplexity 降低但生成质量一般，可能原因是什么？
12. 如果上线服务，你如何在准确率与延迟之间做取舍？

---

## 8. 一句话总结

这个项目的最大价值不只是“跑出一个指标”，而是完整覆盖了 **传统基线 → 深度模型 → Transformer 机制实验** 的能力链条，既能用于课程作业，也能作为面试中“懂原理、会实现、会对比、会优化”的系统化项目案例。
