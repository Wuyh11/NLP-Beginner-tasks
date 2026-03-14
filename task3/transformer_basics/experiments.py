from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .train_lm import LMTrainConfig, run_training as run_lm_training
from .train_math import MathTrainConfig, run_training as run_math_training


def _plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, save_path: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    plt.bar(df[x_col], df[y_col], color="#4C72B0")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def run_math_module_experiments(output_dir: str = "task3/outputs/math") -> pd.DataFrame:
    """数学加法子任务实验：测试模型结构、归一化、位置编码与泛化拆分。"""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        # 基线：seq2seq + LayerNorm + Sinusoidal
        MathTrainConfig(
            run_name="math_s2s_ln_sin_in_dist",
            model_variant="seq2seq",
            norm_type="layernorm",
            pos_encoding="sinusoidal",
            train_templates=((3, 3), (3, 4), (4, 3), (4, 4)),
            test_templates=((3, 3), (3, 4), (4, 3), (4, 4)),
            output_dir=output_dir,
        ),
        # OOD 泛化：测试 3+5 / 5+3
        MathTrainConfig(
            run_name="math_s2s_ln_sin_ood",
            model_variant="seq2seq",
            norm_type="layernorm",
            pos_encoding="sinusoidal",
            train_templates=((3, 3), (3, 4), (4, 3), (4, 4)),
            test_templates=((3, 5), (5, 3)),
            output_dir=output_dir,
        ),
        # RMSNorm 对比
        MathTrainConfig(
            run_name="math_s2s_rms_sin_ood",
            model_variant="seq2seq",
            norm_type="rmsnorm",
            pos_encoding="sinusoidal",
            train_templates=((3, 3), (3, 4), (4, 3), (4, 4)),
            test_templates=((3, 5), (5, 3)),
            output_dir=output_dir,
        ),
        # RoPE 对比
        MathTrainConfig(
            run_name="math_s2s_rms_rope_ood",
            model_variant="seq2seq",
            norm_type="rmsnorm",
            pos_encoding="rope",
            train_templates=((3, 3), (3, 4), (4, 3), (4, 4)),
            test_templates=((3, 5), (5, 3)),
            output_dir=output_dir,
        ),
        # Decoder-only 变种
        MathTrainConfig(
            run_name="math_dec_ln_sin_ood",
            model_variant="decoder_only",
            norm_type="layernorm",
            pos_encoding="sinusoidal",
            train_templates=((3, 3), (3, 4), (4, 3), (4, 4)),
            test_templates=((3, 5), (5, 3)),
            output_dir=output_dir,
        ),
        MathTrainConfig(
            run_name="math_dec_rms_rope_ood",
            model_variant="decoder_only",
            norm_type="rmsnorm",
            pos_encoding="rope",
            train_templates=((3, 3), (3, 4), (4, 3), (4, 4)),
            test_templates=((3, 5), (5, 3)),
            output_dir=output_dir,
        ),
    ]

    rows: list[dict] = []
    for cfg in configs:
        result = run_math_training(cfg)
        rows.append(
            {
                "run_name": result["run_name"],
                "variant": cfg.model_variant,
                "norm": cfg.norm_type,
                "pos": cfg.pos_encoding,
                "best_val_exact": result["best_val_exact"],
                "test_exact": result["test_exact"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "experiment_summary.csv", index=False)
    _plot_bar(df, "run_name", "test_exact", "Task-3 Math: 模块对比 (Exact Match)", out_dir / "test_exact.png")
    return df


def run_lm_module_experiments(output_dir: str = "task3/outputs/lm") -> pd.DataFrame:
    """语言模型子任务实验：测试 tokenizer、词表大小、归一化与位置编码。"""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        LMTrainConfig(
            run_name="lm_word_ln_sin_v500",
            tokenizer_mode="word",
            vocab_size=500,
            norm_type="layernorm",
            pos_encoding="sinusoidal",
            output_dir=output_dir,
        ),
        LMTrainConfig(
            run_name="lm_word_rms_rope_v500",
            tokenizer_mode="word",
            vocab_size=500,
            norm_type="rmsnorm",
            pos_encoding="rope",
            output_dir=output_dir,
        ),
        LMTrainConfig(
            run_name="lm_word_ln_sin_v200",
            tokenizer_mode="word",
            vocab_size=200,
            norm_type="layernorm",
            pos_encoding="sinusoidal",
            output_dir=output_dir,
        ),
        LMTrainConfig(
            run_name="lm_char_ln_sin_v200",
            tokenizer_mode="char",
            vocab_size=200,
            norm_type="layernorm",
            pos_encoding="sinusoidal",
            output_dir=output_dir,
        ),
        LMTrainConfig(
            run_name="lm_char_rms_rope_v200",
            tokenizer_mode="char",
            vocab_size=200,
            norm_type="rmsnorm",
            pos_encoding="rope",
            output_dir=output_dir,
        ),
    ]

    rows: list[dict] = []
    for cfg in configs:
        result = run_lm_training(cfg)
        rows.append(
            {
                "run_name": result["run_name"],
                "tokenizer": cfg.tokenizer_mode,
                "vocab_size": cfg.vocab_size,
                "norm": cfg.norm_type,
                "pos": cfg.pos_encoding,
                "best_val_ppl": result["best_val_ppl"],
                "test_ppl": result["test_ppl"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "experiment_summary.csv", index=False)
    _plot_bar(df, "run_name", "test_ppl", "Task-3 LM: 模块对比 (Perplexity)", out_dir / "test_ppl.png")
    return df


def run_all_experiments(output_root: str = "task3/outputs") -> dict[str, pd.DataFrame]:
    """一键运行 Task-3 两个子任务的实验。"""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    math_df = run_math_module_experiments(output_dir=str(root / "math"))
    lm_df = run_lm_module_experiments(output_dir=str(root / "lm"))
    return {"math": math_df, "lm": lm_df}


if __name__ == "__main__":
    tables = run_all_experiments(output_root="task3/outputs")
    print("[Math]\n", tables["math"])
    print("[LM]\n", tables["lm"])
