"""Task-1 数据读取与切分模块。

该模块职责：
1) 从 `tsv` 文件读取文本分类数据；
2) 兼容“有表头 / 无表头”两种数据格式；
3) 输出统一结构的 `Corpus`，供特征工程与训练代码复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(slots=True)
class DatasetSplit:
    """单个数据切分（文本与标签）。"""

    texts: list[str]
    labels: list[int]


@dataclass(slots=True)
class Corpus:
    """训练/验证/测试三份数据集合。"""

    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit


def _load_tsv(tsv_path: Path) -> pd.DataFrame:
    """读取 TSV 并统一为 `text,label` 两列。"""

    if not tsv_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {tsv_path}")

    # 兼容两种格式：
    # 1) 含表头：text/sentence/review + label/sentiment/score
    # 2) 无表头：第一列文本、第二列标签
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    lowered = {c.lower(): c for c in df.columns}

    text_col = lowered.get("text") or lowered.get("sentence") or lowered.get("review")
    label_col = lowered.get("label") or lowered.get("sentiment") or lowered.get("score")

    if text_col is not None and label_col is not None:
        sub = df[[text_col, label_col]].copy()
        sub.columns = ["text", "label"]
    else:
        raw = pd.read_csv(tsv_path, sep="\t", header=None, names=["text", "label"], usecols=[0, 1], dtype=str)
        sub = raw.copy()

    sub = sub.dropna().reset_index(drop=True)
    sub["text"] = sub["text"].astype(str)
    sub["label"] = sub["label"].astype(int)
    return sub


def load_corpus(
    data_dir: str | Path,
    train_file: str = "new_train.tsv",
    test_file: str = "new_test.tsv",
    val_size: float = 0.1,
    random_state: int = 42,
) -> Corpus:
    """读取训练/测试集并按比例切分验证集。

    参数:
        data_dir: 数据目录。
        train_file: 训练集文件名。
        test_file: 测试集文件名。
        val_size: 从训练集中切分验证集的比例。
        random_state: 随机种子，确保切分可复现。

    返回:
        `Corpus` 对象，包含 `train/val/test` 三个 `DatasetSplit`。
    """
    data_dir = Path(data_dir)

    train_df = _load_tsv(data_dir / train_file)
    test_df = _load_tsv(data_dir / test_file)

    tr_text, va_text, tr_label, va_label = train_test_split(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        test_size=val_size,
        random_state=random_state,
        stratify=train_df["label"].tolist(),
    )

    return Corpus(
        train=DatasetSplit(texts=tr_text, labels=tr_label),
        val=DatasetSplit(texts=va_text, labels=va_label),
        test=DatasetSplit(
            texts=test_df["text"].tolist(),
            labels=test_df["label"].tolist(),
        ),
    )
