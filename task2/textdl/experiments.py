"""Task-2 批量实验模块。

本模块将多组配置自动串行训练，输出：
- `experiment_summary.csv` 结果表；
- 多张柱状图（全部实验、模型对比）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .train import TrainConfig, run_training


def _plot_group(df: pd.DataFrame, x_col: str, y_col: str, title: str, save_path: Path) -> None:
    """绘制柱状图并保存。

    参数:
        df: 输入结果表。
        x_col: x 轴字段名。
        y_col: y 轴字段名。
        title: 图标题。
        save_path: 图片输出路径。
    """

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


def run_experments(
    glove_path: str = "",
    output_dir: str = "task2/outputs/exp8_a10",
    lr: float = 8e-4,
    epochs: int = 20,
    batch_size: int = 128,
    device: str = "cuda",
) -> pd.DataFrame:
    """按要求运行 8 组实验（A10 设定）。

    约束：
    1) 共 8 组；
    2) 优化器统一 AdamW，学习率统一；
    3) 包含 CNN 三种损失函数；
    4) 包含 CNN 使用/不使用 GloVe 初始化对比；
    5) 包含 CNN/LSTM/Transformer 在 GloVe+FocalLoss 下模型对比。

    参数:
        glove_path: GloVe 向量文件路径（为空时自动使用 task2/resources/glove/glove.6B.200d.txt）。
        output_dir: 新实验目录。
        lr: 全部实验统一学习率。
    """

    if not glove_path:
        glove_path = str((Path(__file__).resolve().parents[1] / "resources" / "glove" / "wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"))
    if not Path(glove_path).exists():
        raise FileNotFoundError(
            f"未找到 GloVe 文件: {glove_path}。"
            "请先下载 glove.6B.200d.txt 到 task2/resources/glove/ 或通过 --glove-path 指定。"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A10(24GB) 友好配置：较大 batch + 统一 epoch，兼顾吞吐与稳定性
    common = dict(
        output_dir=output_dir,
        device=device,
        optimizer="adamw",
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        max_len=128,
        weight_decay=1e-2,
        dropout=0.4,
        embedding_dim=200,
    )

    configs: list[TrainConfig] = [
        # 1) CNN + 三种损失函数
        TrainConfig(run_name="exp8_cnn_ce", model_type="cnn", loss="ce", glove_path=glove_path, **common),
        TrainConfig(run_name="exp8_cnn_mse", model_type="cnn", loss="mse", glove_path=glove_path, **common),
        TrainConfig(run_name="exp8_cnn_focal", model_type="cnn", loss="focal", glove_path=glove_path, **common),
        # 2) CNN + GloVe 初始化对比（固定 focal，便于与最终模型对比口径一致）
        TrainConfig(run_name="exp8_cnn_focal_glove", model_type="cnn", loss="focal", glove_path=glove_path, **common),
        TrainConfig(run_name="exp8_cnn_focal_no_glove", model_type="cnn", loss="focal", glove_path="", **common),
        # 3) GloVe + FocalLoss 下模型对比：CNN / LSTM / Transformer
        TrainConfig(run_name="exp8_model_cnn", model_type="cnn", loss="focal", glove_path=glove_path, **common),
        TrainConfig(
            run_name="exp8_model_lstm",
            model_type="rnn",
            rnn_type="lstm",
            loss="focal",
            glove_path=glove_path,
            **common,
        ),
        TrainConfig(
            run_name="exp8_model_transformer",
            model_type="transformer",
            loss="focal",
            glove_path=glove_path,
            embedding_dim=192,
            nhead=6,
            ff_dim=768,
            transformer_layers=2,
            **{k: v for k, v in common.items() if k != "embedding_dim"},
        ),
    ]

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
                "glove": bool(cfg.glove_path),
                "glove_coverage": result["glove_coverage"],
                "best_val_acc": result["best_val_acc"],
                "test_acc": result["test_acc"],
                "test_loss": result["test_loss"],
                "device": result["device"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "experiment_summary.csv", index=False)

    # 全部实验总览图 + 图表数据
    _plot_group(df, "run_name", "test_acc", "Task-2(A10): 8组实验 Test Acc", out_dir / "all_test_acc.png")
    df[["run_name", "test_acc"]].to_csv(out_dir / "all_test_acc_plot_data.csv", index=False)

    # CNN 三损失对比
    loss_df = df[df["run_name"].isin(["exp8_cnn_ce", "exp8_cnn_mse", "exp8_cnn_focal"])].copy()
    if not loss_df.empty:
        loss_df["loss_name"] = loss_df["run_name"].str.replace("exp8_cnn_", "", regex=False)
        _plot_group(loss_df, "loss_name", "test_acc", "Task-2(A10): CNN 不同损失函数对比", out_dir / "cnn_loss_compare.png")
        loss_df[["loss_name", "test_acc"]].to_csv(out_dir / "cnn_loss_compare_plot_data.csv", index=False)

    # CNN GloVe 初始化对比
    glove_df = df[df["run_name"].isin(["exp8_cnn_focal_glove", "exp8_cnn_focal_no_glove"])].copy()
    if not glove_df.empty:
        glove_df["init"] = glove_df["glove"].map({True: "glove", False: "no_glove"})
        _plot_group(glove_df, "init", "test_acc", "Task-2(A10): CNN GloVe 初始化对比", out_dir / "cnn_glove_compare.png")
        glove_df[["init", "test_acc"]].to_csv(out_dir / "cnn_glove_compare_plot_data.csv", index=False)

    # GloVe + FocalLoss 模型对比
    model_df = df[df["run_name"].isin(["exp8_model_cnn", "exp8_model_lstm", "exp8_model_transformer"])].copy()
    if not model_df.empty:
        model_df["model_name"] = model_df["run_name"].map(
            {
                "exp8_model_cnn": "CNN",
                "exp8_model_lstm": "LSTM",
                "exp8_model_transformer": "Transformer",
            }
        )
        _plot_group(
            model_df,
            "model_name",
            "test_acc",
            "Task-2(A10): GloVe + FocalLoss 模型对比",
            out_dir / "model_compare_glove_focal.png",
        )
        model_df[["model_name", "test_acc"]].to_csv(out_dir / "model_compare_glove_focal_plot_data.csv", index=False)

    return df


def build_parser() -> argparse.ArgumentParser:
    """构建批量实验 CLI。"""

    p = argparse.ArgumentParser(description="Task-2 批量实验入口")
    p.add_argument("--suite", choices=["all", "exp8"], default="exp8", help="运行实验集合")
    p.add_argument("--output-dir", type=str, default="task2/outputs/exp8_a10", help="结果输出目录")
    p.add_argument("--glove-path", type=str, default="", help="GloVe txt 文件路径")
    p.add_argument("--lr", type=float, default=8e-4, help="统一学习率")
    p.add_argument("--epochs", type=int, default=20, help="训练轮数（A10 默认 20）")
    p.add_argument("--batch-size", type=int, default=128, help="批大小")
    p.add_argument("--device", type=str, default="cuda", help="训练设备，例如 cuda / cpu / auto")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.suite == "all":
        table = run_all_experiments(output_dir=args.output_dir, glove_path=args.glove_path)
    else:
        table = run_experments(
            glove_path=args.glove_path,
            output_dir=args.output_dir,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
    print(table)
