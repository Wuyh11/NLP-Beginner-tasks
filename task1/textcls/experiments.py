"""Task-1 对比实验模块。

用于批量运行多组超参数（n-gram、损失函数、学习率）并：
- 汇总结果为表格；
- 导出 CSV；
- 绘制测试集准确率柱状图。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .train import TrainConfig, run_training


def run_default_experiments(output_dir: str = "task1/outputs") -> pd.DataFrame:
    """运行 Task-1 的默认对比实验并绘图。"""

    settings = [
        {"name": "bow_ce_lr0.1", "ngram": 1, "loss": "ce", "lr": 0.1},
        {"name": "ngram2_ce_lr0.1", "ngram": 2, "loss": "ce", "lr": 0.1},
        {"name": "bow_ce_lr0.05", "ngram": 1, "loss": "ce", "lr": 0.05},
        {"name": "bow_mse_lr0.1", "ngram": 1, "loss": "mse", "lr": 0.1},
    ]

    rows: list[dict] = []
    for cfg_dict in settings:
        cfg = TrainConfig(output_dir=output_dir, **{k: v for k, v in cfg_dict.items() if k != "name"})
        result = run_training(cfg)
        rows.append(
            {
                "name": cfg_dict["name"],
                "ngram": cfg.ngram,
                "loss": cfg.loss,
                "lr": cfg.lr,
                "val_acc": result["final_val_acc"],
                "test_acc": result["final_test_acc"],
                "vocab_size": result["vocab_size"],
            }
        )

    df = pd.DataFrame(rows)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "experiment_summary.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    plt.bar(df["name"], df["test_acc"], color="#4C72B0")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Test Accuracy")
    plt.title("Task-1 实验对比")
    plt.tight_layout()
    plt.savefig(out_dir / "experiment_test_acc.png", dpi=160)

    return df


if __name__ == "__main__":
    table = run_default_experiments(output_dir="task1/outputs")
    print(table)
