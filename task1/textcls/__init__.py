"""Task-1 文本分类项目包。

子模块说明：
- `data`: 数据加载与训练/验证/测试切分。
- `features`: BoW/N-gram 特征抽取。
- `model`: 手写线性分类器与梯度更新。
- `train`: 单次训练入口。
- `experiments`: 批量实验与可视化。
"""

__all__ = [
    "data",
    "features",
    "model",
    "train",
    "experiments",
]
