"""Task-2 数据处理模块。

主要功能：
1) TSV 数据读取与标准化；
2) 文本分词与词表（`Vocab`）构建；
3) `Dataset` + `DataLoader` 组织，并在 `collate_fn` 内动态 padding。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[\u4e00-\u9fff]")


@dataclass(slots=True)
class DatasetSplit:
    """单个数据切分（文本与标签）。"""

    texts: list[str]
    labels: list[int]


@dataclass(slots=True)
class Corpus:
    """训练/验证/测试三份数据。"""

    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit


def tokenize(text: str) -> list[str]:
    """统一英文/数字/中文字符级分词规则。"""

    return TOKEN_PATTERN.findall(text.lower())


class Vocab:
    """词表对象：维护 token-id 映射及编码逻辑。"""

    def __init__(self, min_freq: int = 1, max_size: int = 50000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.itos: list[str] = [self.pad_token, self.unk_token]
        self.stoi: dict[str, int] = {self.pad_token: 0, self.unk_token: 1}

    @property
    def pad_idx(self) -> int:
        return 0

    @property
    def unk_idx(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return len(self.itos)

    def build(self, texts: list[str]) -> None:
        """在训练文本上统计词频并建立 token-id 映射。"""

        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))

        items = [(tok, freq) for tok, freq in counter.items() if freq >= self.min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        items = items[: max(0, self.max_size - 2)]

        for tok, _ in items:
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)

    def encode(self, text: str, max_len: int) -> list[int]:
        """将单条文本转为定长上限的 token id 序列。"""

        token_ids = [self.stoi.get(tok, self.unk_idx) for tok in tokenize(text)]
        if not token_ids:
            token_ids = [self.unk_idx]
        return token_ids[:max_len]


def _load_tsv(tsv_path: Path) -> pd.DataFrame:
    """读取 TSV 并兼容有/无表头两种格式。"""

    if not tsv_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    lowered = {c.lower(): c for c in df.columns}

    text_col = lowered.get("text") or lowered.get("sentence") or lowered.get("review")
    label_col = lowered.get("label") or lowered.get("sentiment") or lowered.get("score")

    if text_col is not None and label_col is not None:
        out = df[[text_col, label_col]].copy()
        out.columns = ["text", "label"]
    else:
        out = pd.read_csv(
            tsv_path,
            sep="\t",
            header=None,
            names=["text", "label"],
            usecols=[0, 1],
            dtype=str,
        )

    out = out.dropna().reset_index(drop=True)
    out["text"] = out["text"].astype(str)
    out["label"] = out["label"].astype(int)
    return out


def load_corpus(
    data_dir: str | Path,
    train_file: str = "new_train.tsv",
    test_file: str = "new_test.tsv",
    val_size: float = 0.1,
    random_state: int = 42,
) -> Corpus:
    """读取并切分数据集。"""

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
        test=DatasetSplit(texts=test_df["text"].tolist(), labels=test_df["label"].tolist()),
    )


class TextDataset(Dataset):
    """将原始文本按需编码成 token id 序列。"""

    def __init__(self, texts: list[str], labels: list[int], vocab: Vocab, max_len: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        token_ids = self.vocab.encode(self.texts[idx], self.max_len)
        return token_ids, self.labels[idx]


def build_dataloader(
    texts: list[str],
    labels: list[int],
    vocab: Vocab,
    max_len: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """构建 DataLoader，并在 collate 时完成动态 padding。"""

    ds = TextDataset(texts, labels, vocab, max_len)

    def collate_fn(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将不同长度样本拼成 batch，并返回长度信息供 RNN 使用。"""

        seqs, ys = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        max_batch_len = int(lengths.max().item())
        x = torch.full((len(seqs), max_batch_len), vocab.pad_idx, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        return x, lengths, y

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
