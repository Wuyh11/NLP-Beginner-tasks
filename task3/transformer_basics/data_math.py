"""Task-3 数学加法数据模块。

该模块生成可控位数模板的加法样本，并提供：
- seq2seq 训练数据（输入 `a+b`，输出 `sum`）；
- decoder-only 训练数据（完整串 `a+b=sum`）。
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch.utils.data import DataLoader, Dataset


class CharTokenizer:
    """字符级 tokenizer（含 `<pad>/<bos>/<eos>`）。"""

    def __init__(self):
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.itos: list[str] = [self.pad_token, self.bos_token, self.eos_token]
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
    def vocab_size(self) -> int:
        return len(self.itos)

    def fit(self, texts: list[str]) -> None:
        chars = sorted({ch for txt in texts for ch in txt})
        for ch in chars:
            if ch not in self.stoi:
                self.stoi[ch] = len(self.itos)
                self.itos.append(ch)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_idx)
        ids.extend(self.stoi[ch] for ch in text)
        if add_eos:
            ids.append(self.eos_idx)
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for idx in ids:
            tok = self.itos[idx]
            if tok in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            out.append(tok)
        return "".join(out)


@dataclass(slots=True)
class MathSample:
    a: int
    b: int

    @property
    def src(self) -> str:
        return f"{self.a}+{self.b}"

    @property
    def tgt(self) -> str:
        return str(self.a + self.b)

    @property
    def full(self) -> str:
        return f"{self.a}+{self.b}={self.a + self.b}"


def _random_n_digit(n: int, rng: random.Random) -> int:
    """随机采样一个 `n` 位正整数。"""

    low = 10 ** (n - 1)
    high = 10**n - 1
    return rng.randint(low, high)


def build_math_samples(
    templates: list[tuple[int, int]],
    n_per_template: int,
    seed: int,
) -> list[MathSample]:
    """按位数模板构造加法样本，如 `(3,4)` 表示 3 位数 + 4 位数。"""
    rng = random.Random(seed)
    out: list[MathSample] = []
    for la, lb in templates:
        seen: set[tuple[int, int]] = set()
        while len(seen) < n_per_template:
            a = _random_n_digit(la, rng)
            b = _random_n_digit(lb, rng)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            out.append(MathSample(a=a, b=b))
    rng.shuffle(out)
    return out


class MathSeq2SeqDataset(Dataset):
    """Encoder-Decoder 数据集：输入 `a+b`，目标 `sum`。"""

    def __init__(self, samples: list[MathSample], tokenizer: CharTokenizer):
        self.samples = samples
        self.tok = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int], str]:
        """返回 `(src_ids, tgt_ids, target_text)`。"""

        s = self.samples[idx]
        src_ids = self.tok.encode(s.src, add_bos=False, add_eos=True)
        tgt_ids = self.tok.encode(s.tgt, add_bos=True, add_eos=True)
        return src_ids, tgt_ids, s.tgt


class MathDecoderOnlyDataset(Dataset):
    """Decoder-only 数据集：完整串 `a+b=sum`。"""

    def __init__(self, samples: list[MathSample], tokenizer: CharTokenizer):
        self.samples = samples
        self.tok = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], str, str]:
        """返回 `(full_ids, prompt_text, target_text)`。"""

        s = self.samples[idx]
        full = self.tok.encode(s.full, add_bos=True, add_eos=True)
        prompt = f"{s.a}+{s.b}="
        return full, prompt, s.tgt


def make_seq2seq_loader(
    ds: MathSeq2SeqDataset,
    batch_size: int,
    pad_idx: int,
    shuffle: bool,
) -> DataLoader:
    """构建 seq2seq DataLoader，并在 batch 内动态 padding。"""

    def collate_fn(batch):
        srcs, tgts, ans = zip(*batch)
        max_src = max(len(x) for x in srcs)
        max_tgt = max(len(x) for x in tgts)

        src = torch.full((len(srcs), max_src), pad_idx, dtype=torch.long)
        tgt = torch.full((len(tgts), max_tgt), pad_idx, dtype=torch.long)
        for i, ids in enumerate(srcs):
            src[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        for i, ids in enumerate(tgts):
            tgt[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return src, tgt, list(ans)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def make_decoder_only_loader(
    ds: MathDecoderOnlyDataset,
    batch_size: int,
    pad_idx: int,
    shuffle: bool,
) -> DataLoader:
    """构建 decoder-only DataLoader。"""

    def collate_fn(batch):
        fulls, prompts, ans = zip(*batch)
        max_len = max(len(x) for x in fulls)
        x = torch.full((len(fulls), max_len), pad_idx, dtype=torch.long)
        for i, ids in enumerate(fulls):
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return x, list(prompts), list(ans)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
