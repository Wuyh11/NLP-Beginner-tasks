"""Task-3: Transformer 基础结构与实验。

包含两个子任务：
- 数学加法（`train_math`）：对比 seq2seq 与 decoder-only 结构；
- 语言模型（`train_lm`）：评估 tokenizer、归一化、位置编码等模块。

并提供：
- `models`: 手写 Transformer 核心部件；
- `experiments`: 一键批量实验与统计图绘制。
"""

__all__ = [
    "models",
    "data_math",
    "data_lm",
    "train_math",
    "train_lm",
    "experiments",
]
