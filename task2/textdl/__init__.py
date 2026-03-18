"""Task-2 基于深度学习的文本分类。

子模块说明：
- `data`: 分词、词表构建、Dataset 与 DataLoader。
- `models`: TextCNN / TextRNN / TextTransformer。
- `train`: 训练主流程与评估逻辑。
- `experiments`: 多组实验配置、汇总与绘图。
"""

__all__ = ["data", "models", "train", "experiments"]
