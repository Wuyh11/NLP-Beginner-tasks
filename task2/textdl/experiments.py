from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .train import TrainConfig, run_training


def _plot_group(df: pd.DataFrame, x_col: str, y_col: str, title: str, save_path: Path) -> None:
    """绘制柱状图并保存。"""

    plt.figure(figsize=(9, 4.8))
    plt.bar(df[x_col], df[y_col], color="#4C72B0")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def run_all_experiments(output_dir: str = "task2/outputs", glove_path: str = "") -> pd.DataFrame:
    """运行 Task-2 系列实验并汇总结果。"""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs: list[TrainConfig] = []

    # 1) 损失函数 + 学习率
    configs.extend(
        [
            TrainConfig(run_name="cnn_ce_lr1e3", model_type="cnn", loss="ce", lr=1e-3, output_dir=output_dir),
            TrainConfig(run_name="cnn_focal_lr1e3", model_type="cnn", loss="focal", lr=1e-3, output_dir=output_dir),
            TrainConfig(run_name="cnn_mse_lr1e3", model_type="cnn", loss="mse", lr=1e-3, output_dir=output_dir),
            TrainConfig(run_name="cnn_ce_lr5e4", model_type="cnn", loss="ce", lr=5e-4, output_dir=output_dir),
        ]
    )

    # 2) 卷积核 + 优化器
    configs.extend(
        [
            TrainConfig(
                run_name="cnn_k345_adam",
                model_type="cnn",
                kernel_sizes="3,4,5",
                num_filters=100,
                optimizer="adam",
                output_dir=output_dir,
            ),
            TrainConfig(
                run_name="cnn_k23_adamw",
                model_type="cnn",
                kernel_sizes="2,3",
                num_filters=128,
                optimizer="adamw",
                output_dir=output_dir,
            ),
            TrainConfig(
                run_name="cnn_k34567_sgd",
                model_type="cnn",
                kernel_sizes="3,4,5,6,7",
                num_filters=64,
                optimizer="sgd",
                lr=5e-3,
                output_dir=output_dir,
            ),
        ]
    )

    # 3) GloVe 初始化
    if glove_path:
        configs.extend(
            [
                TrainConfig(
                    run_name="cnn_glove_ft",
                    model_type="cnn",
                    glove_path=glove_path,
                    freeze_embedding=False,
                    output_dir=output_dir,
                ),
                TrainConfig(
                    run_name="cnn_glove_freeze",
                    model_type="cnn",
                    glove_path=glove_path,
                    freeze_embedding=True,
                    output_dir=output_dir,
                ),
            ]
        )

    # 4) CNN vs RNN vs Transformer
    configs.extend(
        [
            TrainConfig(run_name="cnn_compare", model_type="cnn", output_dir=output_dir),
            TrainConfig(run_name="rnn_compare", model_type="rnn", output_dir=output_dir),
            TrainConfig(
                run_name="transformer_compare",
                model_type="transformer",
                embedding_dim=192,
                nhead=6,
                output_dir=output_dir,
            ),
        ]
    )

    rows: list[dict] = []
    for cfg in configs:
        result = run_training(cfg)
        rows.append(
            {
                "run_name": result["run_name"],
                "model": cfg.model_type,
                "loss": cfg.loss,
                "lr": cfg.lr,
                "optimizer": cfg.optimizer,
                "kernel_sizes": cfg.kernel_sizes,
                "glove": bool(cfg.glove_path),
                "glove_coverage": result["glove_coverage"],
                "best_val_acc": result["best_val_acc"],
                "test_acc": result["test_acc"],
                "device": result["device"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "experiment_summary.csv", index=False)

    _plot_group(df, "run_name", "test_acc", "Task-2: 全部实验 Test Acc", out_dir / "all_test_acc.png")

    model_df = df[df["run_name"].isin(["cnn_compare", "rnn_compare", "transformer_compare"])]
    if not model_df.empty:
        _plot_group(model_df, "model", "test_acc", "Task-2: 模型对比", out_dir / "model_compare.png")

    return df


if __name__ == "__main__":
    table = run_all_experiments(output_dir="task2/outputs")
    print(table)
