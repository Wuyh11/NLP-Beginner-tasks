from __future__ import annotations

import math

import torch
from torch import nn


class TextCNN(nn.Module):
    """Kim-CNN 风格文本分类器。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        pad_idx: int,
        num_filters: int = 100,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, kernel_size=(k, embed_dim)) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # input_ids: [B, L] -> emb: [B, L, D]
        emb = self.embedding(input_ids)
        emb = emb.unsqueeze(1)
        # 2D 卷积 + 全局最大池化
        conv_out = [torch.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(feat, dim=2).values for feat in conv_out]
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat))


class TextRNN(nn.Module):
    """基于 LSTM/GRU 的文本分类器。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_classes: int,
        pad_idx: int,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn_type = rnn_type
        rnn_dropout = dropout if num_layers > 1 else 0.0

        if rnn_type == "gru":
            self.rnn = nn.GRU(
                embed_dim,
                hidden_size,
                batch_first=True,
                bidirectional=bidirectional,
                num_layers=num_layers,
                dropout=rnn_dropout,
            )
        else:
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_size,
                batch_first=True,
                bidirectional=bidirectional,
                num_layers=num_layers,
                dropout=rnn_dropout,
            )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        # 使用 pack 减少 padding 位置计算
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)

        if self.rnn_type == "lstm":
            h_n, _ = hidden
        else:
            h_n = hidden

        if self.rnn.bidirectional:
            rep = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            rep = h_n[-1]
        return self.fc(self.dropout(rep))


class PositionalEncoding(nn.Module):
    """标准正弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TextTransformer(nn.Module):
    """基于 TransformerEncoder 的文本分类器。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        pad_idx: int,
        nhead: int = 8,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # True 表示 padding 位置
        mask = input_ids.eq(self.pad_idx)
        emb = self.embedding(input_ids)
        emb = self.pos_encoding(emb)
        encoded = self.encoder(emb, src_key_padding_mask=mask)

        # 对非 padding 位置做平均池化
        valid = (~mask).unsqueeze(-1)
        summed = (encoded * valid).sum(dim=1)
        denom = valid.sum(dim=1).clamp_min(1)
        pooled = summed / denom
        return self.fc(self.dropout(pooled))
