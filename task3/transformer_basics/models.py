from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


def make_subsequent_mask(length: int, device: torch.device) -> torch.Tensor:
    """构造 subsequent mask：禁止看到未来 token。"""
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class RMSNorm(nn.Module):
    """RMSNorm 实现。"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE: Rotary Position Embedding。"""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("使用 RoPE 时要求 head_dim 为偶数")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def apply(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (self._rotate_half(x) * sin)


class MultiHeadAttention(nn.Module):
    """手写 Multi-Head Attention。"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能整除 nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.use_rope = use_rope
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base) if use_rope else None

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len, _ = query.shape
        k_len = key.size(1)

        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))

        if self.use_rope and self.rope is not None:
            q_cos, q_sin = self.rope._build_cos_sin(q_len, q.device, q.dtype)
            k_cos, k_sin = self.rope._build_cos_sin(k_len, k.device, k.dtype)
            q = self.rope.apply(q, q_cos, q_sin)
            k = self.rope.apply(k, k_cos, k_sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, None, :, :]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[:, None, :, :]

            if attn_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(attn_mask, -1e9)
            else:
                attn_scores = attn_scores + attn_mask.to(dtype=attn_scores.dtype)

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(torch.bool)
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        attn_prob = torch.softmax(attn_scores, dim=-1)
        attn_prob = self.drop(attn_prob)
        out = torch.matmul(attn_prob, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        out = self.out_proj(out)

        weights = attn_prob.mean(dim=1) if need_weights else None
        return out, weights


def _build_norm(d_model: int, norm_type: str, eps: float) -> nn.Module:
    norm = norm_type.lower()
    if norm == "layernorm":
        return nn.LayerNorm(d_model, eps=eps)
    if norm == "rmsnorm":
        return RMSNorm(d_model, eps=eps)
    raise ValueError("norm_type 仅支持 layernorm 或 rmsnorm")


class PositionalEncoding(nn.Module):
    """标准正弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    """TransformerEncoderLayer 基础实现：MHA + FFN + 残差 + LayerNorm。"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_dim: int,
        dropout: float,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-5,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=dropout,
            use_rope=use_rope,
            rope_base=rope_base,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = _build_norm(d_model, norm_type, norm_eps)
        self.norm2 = _build_norm(d_model, norm_type, norm_eps)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.drop(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.drop(ffn_out))


class DecoderLayer(nn.Module):
    """TransformerDecoderLayer 基础实现。"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_dim: int,
        dropout: float,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-5,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=dropout,
            use_rope=use_rope,
            rope_base=rope_base,
        )
        self.cross_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=dropout,
            use_rope=False,
            rope_base=rope_base,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = _build_norm(d_model, norm_type, norm_eps)
        self.norm2 = _build_norm(d_model, norm_type, norm_eps)
        self.norm3 = _build_norm(d_model, norm_type, norm_eps)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        tgt_key_padding_mask: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        self_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.drop(self_out))

        cross_out, _ = self.cross_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = self.norm2(x + self.drop(cross_out))

        ffn_out = self.ffn(x)
        return self.norm3(x + self.drop(ffn_out))


class DecoderOnlyLayer(nn.Module):
    """Decoder-only block：自注意力 + FFN。"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_dim: int,
        dropout: float,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-5,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=dropout,
            use_rope=use_rope,
            rope_base=rope_base,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = _build_norm(d_model, norm_type, norm_eps)
        self.norm2 = _build_norm(d_model, norm_type, norm_eps)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.drop(out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.drop(ffn_out))


@dataclass(slots=True)
class TransformerConfig:
    vocab_size: int
    pad_idx: int
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    max_len: int = 256
    norm_type: str = "layernorm"  # layernorm | rmsnorm
    norm_eps: float = 1e-5
    pos_encoding: str = "sinusoidal"  # sinusoidal | rope | none
    rope_base: float = 10000.0


class Seq2SeqTransformer(nn.Module):
    """Encoder-Decoder Transformer。"""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.pad_idx = cfg.pad_idx
        self.src_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        self.tgt_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        use_rope = cfg.pos_encoding.lower() == "rope"
        if cfg.pos_encoding.lower() == "sinusoidal":
            self.pos = PositionalEncoding(cfg.d_model, max_len=cfg.max_len)
        elif cfg.pos_encoding.lower() in {"rope", "none"}:
            self.pos = nn.Identity()
        else:
            raise ValueError("pos_encoding 仅支持 sinusoidal、rope、none")

        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    cfg.d_model,
                    cfg.nhead,
                    cfg.ff_dim,
                    cfg.dropout,
                    norm_type=cfg.norm_type,
                    norm_eps=cfg.norm_eps,
                    use_rope=use_rope,
                    rope_base=cfg.rope_base,
                )
                for _ in range(cfg.num_encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(
                    cfg.d_model,
                    cfg.nhead,
                    cfg.ff_dim,
                    cfg.dropout,
                    norm_type=cfg.norm_type,
                    norm_eps=cfg.norm_eps,
                    use_rope=use_rope,
                    rope_base=cfg.rope_base,
                )
                for _ in range(cfg.num_decoder_layers)
            ]
        )
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size)

    def encode(self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.pos(self.src_embed(src_ids))
        for layer in self.encoder:
            x = layer(x, src_key_padding_mask)
        return x

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        tgt_key_padding_mask: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.pos(self.tgt_embed(tgt_ids))
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.out_proj(x)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None,
        tgt_key_padding_mask: torch.Tensor | None,
        tgt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        memory = self.encode(src_ids, src_key_padding_mask)
        return self.decode(
            tgt_ids,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only Transformer。"""

    def __init__(self, cfg: TransformerConfig, num_layers: int = 4):
        super().__init__()
        self.pad_idx = cfg.pad_idx
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        use_rope = cfg.pos_encoding.lower() == "rope"
        if cfg.pos_encoding.lower() == "sinusoidal":
            self.pos = PositionalEncoding(cfg.d_model, max_len=cfg.max_len)
        elif cfg.pos_encoding.lower() in {"rope", "none"}:
            self.pos = nn.Identity()
        else:
            raise ValueError("pos_encoding 仅支持 sinusoidal、rope、none")
        self.layers = nn.ModuleList(
            [
                DecoderOnlyLayer(
                    cfg.d_model,
                    cfg.nhead,
                    cfg.ff_dim,
                    cfg.dropout,
                    norm_type=cfg.norm_type,
                    norm_eps=cfg.norm_eps,
                    use_rope=use_rope,
                    rope_base=cfg.rope_base,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.pos(self.embed(input_ids))
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.out_proj(x)
