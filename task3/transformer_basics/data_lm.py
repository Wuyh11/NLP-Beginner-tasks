"""Task-3 语言模型数据模块。

职责：
1) 构造可控的合成语料；
2) 提供 char/word 两种 tokenizer；
3) 将 token 序列切成固定窗口的 next-token 训练样本。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
import re

import torch
from torch.utils.data import DataLoader, Dataset


def build_synthetic_corpus(n_sentences: int = 6000, seed: int = 42) -> list[str]:
    """自造语料：简单叙述句，便于 language model 训练。"""
    rng = random.Random(seed)

    subjects = ["the student", "the model", "the teacher", "the system", "the team"]
    verbs = ["learns", "tests", "builds", "analyzes", "improves"]
    objects = ["transformer", "attention", "tokenizer", "dataset", "result"]
    tails = ["carefully", "quickly", "in practice", "for research", "with patience"]

    sents: list[str] = []
    for _ in range(n_sentences):
        s = f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} {rng.choice(tails)}."
        sents.append(s)
    return sents


class LMTokenizer:
    """支持 char / word 两种分词方式的简易 tokenizer。"""

    def __init__(self, mode: str = "char", vocab_size: int = 500):
        if mode not in {"char", "word"}:
            raise ValueError("mode 只支持 char 或 word")
        self.mode = mode
        self.max_vocab = vocab_size
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.itos: list[str] = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    @property
    def pad_idx(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def bos_idx(self) -> int:
        return self.stoi[self.bos_token]

    @property
    def eos_idx(self) -> int:
        return self.stoi[self.eos_token]

    @property
    def unk_idx(self) -> int:
        return self.stoi[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def _split(self, text: str) -> list[str]:
        if self.mode == "char":
            return list(text)
        return re.findall(r"[A-Za-z]+|\d+|[^\w\s]", text.lower())

    def fit(self, texts: list[str]) -> None:
        cnt: Counter[str] = Counter()
        for t in texts:
            cnt.update(self._split(t))

        keep = [tok for tok, _ in cnt.most_common(max(0, self.max_vocab - len(self.itos)))]
        for tok in keep:
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_idx)
        ids.extend(self.stoi.get(tok, self.unk_idx) for tok in self._split(text))
        if add_eos:
            ids.append(self.eos_idx)
        return ids

    def decode(self, ids: list[int]) -> str:
        toks: list[str] = []
        for idx in ids:
            tok = self.itos[idx]
            if tok in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            toks.append(tok)
        if self.mode == "char":
            return "".join(toks)
        return " ".join(toks)


@dataclass(slots=True)
class LMDataSplit:
    """语言模型数据切分（按 token 时间顺序拆分）。"""

    train_ids: list[int]
    val_ids: list[int]
    test_ids: list[int]


def build_lm_split(
    corpus: list[str],
    tokenizer: LMTokenizer,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> LMDataSplit:
    """将语料编码后按时间顺序划分 train/val/test。"""
    all_ids: list[int] = []
    for sent in corpus:
        all_ids.extend(tokenizer.encode(sent, add_bos=True, add_eos=True))

    n = len(all_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return LMDataSplit(
        train_ids=all_ids[:n_train],
        val_ids=all_ids[n_train : n_train + n_val],
        test_ids=all_ids[n_train + n_val :],
    )


class LMDataset(Dataset):
    """固定窗口 next-token 训练样本。"""

    def __init__(self, token_ids: list[int], block_size: int):
        self.ids = token_ids
        self.block = block_size

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.block - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回一个样本 `(x, y)`，其中 `y` 为 `x` 右移一位。"""

        x = torch.tensor(self.ids[idx : idx + self.block], dtype=torch.long)
        y = torch.tensor(self.ids[idx + 1 : idx + self.block + 1], dtype=torch.long)
        return x, y


def make_lm_loader(ds: LMDataset, batch_size: int, shuffle: bool) -> DataLoader:
    """构建 LM 训练 DataLoader。"""

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
