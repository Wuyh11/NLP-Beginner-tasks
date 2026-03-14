from __future__ import annotations

from collections import Counter
import re

import torch

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[\u4e00-\u9fff]")


class CountVectorizer:
    """简化版 BoW / N-gram 向量器。"""

    def __init__(self, ngram: int = 1, max_features: int = 20000, min_freq: int = 2):
        if ngram < 1:
            raise ValueError("ngram 必须 >= 1")
        self.ngram = ngram
        self.max_features = max_features
        self.min_freq = min_freq
        self.stoi: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def _tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def _to_ngrams(self, tokens: list[str]) -> list[str]:
        if self.ngram == 1:
            return tokens
        out: list[str] = []
        for n in range(1, self.ngram + 1):
            if len(tokens) < n:
                continue
            out.extend("_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
        return out

    def fit(self, texts: list[str]) -> None:
        """基于训练文本统计词频并构建词表。"""

        counter: Counter[str] = Counter()
        for text in texts:
            grams = self._to_ngrams(self._tokenize(text))
            counter.update(grams)

        items = [(tok, f) for tok, f in counter.items() if f >= self.min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        items = items[: self.max_features]

        self.stoi = {tok: idx for idx, (tok, _) in enumerate(items)}

    def transform_batch(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """将一个 batch 的文本转为稀疏计数向量。"""

        x = torch.zeros((len(texts), self.vocab_size), dtype=torch.float32, device=device)
        for row, text in enumerate(texts):
            counts: Counter[int] = Counter()
            for token in self._to_ngrams(self._tokenize(text)):
                idx = self.stoi.get(token)
                if idx is not None:
                    counts[idx] += 1
            if counts:
                idxs = torch.tensor(list(counts.keys()), dtype=torch.long, device=device)
                vals = torch.tensor(list(counts.values()), dtype=torch.float32, device=device)
                x[row, idxs] = vals
        return x
