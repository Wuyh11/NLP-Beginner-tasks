"""Task-3 数学加法训练模块。

目标：比较不同 Transformer 模块配置在“位数泛化”场景下的表现。
支持两种建模方式：
- `seq2seq`（编码器-解码器）；
- `decoder_only`（自回归）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from time import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from .data_math import (
    CharTokenizer,
    MathSample,
    MathDecoderOnlyDataset,
    MathSeq2SeqDataset,
    build_math_samples,
    make_decoder_only_loader,
    make_seq2seq_loader,
)
from .models import DecoderOnlyTransformer, Seq2SeqTransformer, TransformerConfig, make_subsequent_mask


@dataclass(slots=True)
class MathTrainConfig:
    run_name: str = ""
    output_dir: str = "task3/outputs/math"
    seed: int = 42
    device: str = "auto"

    # 数据划分：用于测试泛化
    train_templates: tuple[tuple[int, int], ...] = ((3, 3), (3, 4), (4, 3), (4, 4))
    test_templates: tuple[tuple[int, int], ...] = ((3, 5), (5, 3))
    n_train_per_template: int = 2000
    n_test_per_template: int = 400
    val_ratio: float = 0.1

    # 模型变种
    model_variant: str = "seq2seq"  # seq2seq | decoder_only
    num_layers: int = 4  # decoder_only 专用

    # Transformer
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    max_len: int = 64
    norm_type: str = "layernorm"  # layernorm | rmsnorm
    pos_encoding: str = "sinusoidal"  # sinusoidal | rope | none
    rope_base: float = 10000.0

    # 训练
    batch_size: int = 128
    epochs: int = 12
    lr: float = 2e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_device(preferred: str) -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def _split_train_val(samples: list[MathSample], val_ratio: float, seed: int) -> tuple[list[MathSample], list[MathSample]]:
    """按比例切分训练/验证样本。"""

    rng = random.Random(seed)
    arr = samples.copy()
    rng.shuffle(arr)
    n_val = max(1, int(len(arr) * val_ratio))
    return arr[n_val:], arr[:n_val]


def _build_tokenizer(train_samples: list[MathSample], test_samples: list[MathSample]) -> CharTokenizer:
    """基于 train+test 文本建立字符词表，避免 OOV。"""

    tok = CharTokenizer()
    texts: list[str] = []
    for s in train_samples + test_samples:
        texts.extend([s.src, s.tgt, s.full])
    tok.fit(texts)
    return tok


def _build_model(cfg: MathTrainConfig, tok: CharTokenizer) -> nn.Module:
    model_cfg = TransformerConfig(
        vocab_size=tok.vocab_size,
        pad_idx=tok.pad_idx,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        norm_type=cfg.norm_type,
        pos_encoding=cfg.pos_encoding,
        rope_base=cfg.rope_base,
    )
    if cfg.model_variant == "seq2seq":
        return Seq2SeqTransformer(model_cfg)
    if cfg.model_variant == "decoder_only":
        return DecoderOnlyTransformer(model_cfg, num_layers=cfg.num_layers)
    raise ValueError("model_variant 仅支持 seq2seq / decoder_only")


def _train_one_epoch_seq2seq(
    model: Seq2SeqTransformer,
    loader,
    optimizer,
    device: torch.device,
    pad_idx: int,
) -> float:
    model.train()
    ce = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_loss = 0.0
    total_tokens = 0

    for src, tgt, _ in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_pad = src.eq(pad_idx)
        tgt_pad = tgt_in.eq(pad_idx)
        tgt_mask = make_subsequent_mask(tgt_in.size(1), device)

        optimizer.zero_grad()
        logits = model(src, tgt_in, src_pad, tgt_pad, tgt_mask)
        loss = ce(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        n_tokens = int(tgt_out.ne(pad_idx).sum().item())
        total_loss += float(loss.item()) * max(n_tokens, 1)
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def _train_one_epoch_decoder_only(
    model: DecoderOnlyTransformer,
    loader,
    optimizer,
    device: torch.device,
    pad_idx: int,
) -> float:
    model.train()
    ce = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_loss = 0.0
    total_tokens = 0

    for x, _, _ in loader:
        x = x.to(device)
        x_in = x[:, :-1]
        y = x[:, 1:]
        key_pad = x_in.eq(pad_idx)
        attn_mask = make_subsequent_mask(x_in.size(1), device)

        optimizer.zero_grad()
        logits = model(x_in, key_padding_mask=key_pad, attn_mask=attn_mask)
        loss = ce(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()

        n_tokens = int(y.ne(pad_idx).sum().item())
        total_loss += float(loss.item()) * max(n_tokens, 1)
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def _predict_seq2seq(
    model: Seq2SeqTransformer,
    tok: CharTokenizer,
    expr: str,
    device: torch.device,
    max_new_tokens: int = 16,
) -> str:
    src_ids = tok.encode(expr, add_bos=False, add_eos=True)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad = src.eq(tok.pad_idx)
    memory = model.encode(src, src_pad)

    ys = [tok.bos_idx]
    for _ in range(max_new_tokens):
        tgt = torch.tensor([ys], dtype=torch.long, device=device)
        tgt_pad = tgt.eq(tok.pad_idx)
        tgt_mask = make_subsequent_mask(tgt.size(1), device)
        logits = model.decode(tgt, memory, tgt_mask, tgt_pad, src_pad)
        nxt = int(logits[0, -1].argmax().item())
        ys.append(nxt)
        if nxt == tok.eos_idx:
            break

    return tok.decode(ys[1:])


@torch.no_grad()
def _predict_decoder_only(
    model: DecoderOnlyTransformer,
    tok: CharTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 16,
) -> str:
    ids = tok.encode(prompt, add_bos=True, add_eos=False)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        key_pad = x.eq(tok.pad_idx)
        attn_mask = make_subsequent_mask(x.size(1), device)
        logits = model(x, key_padding_mask=key_pad, attn_mask=attn_mask)
        nxt = int(logits[0, -1].argmax().item())
        ids.append(nxt)
        if nxt == tok.eos_idx:
            break

    # 只返回 prompt 之后的增量
    return tok.decode(ids[len(tok.encode(prompt, add_bos=True, add_eos=False)) :])


@torch.no_grad()
def evaluate_exact_match(
    model: nn.Module,
    samples: list[MathSample],
    tok: CharTokenizer,
    model_variant: str,
    device: torch.device,
) -> float:
    model.eval()
    hit = 0
    for s in samples:
        if model_variant == "seq2seq":
            pred = _predict_seq2seq(model, tok, s.src, device=device)
        else:
            pred = _predict_decoder_only(model, tok, f"{s.a}+{s.b}=", device=device)
        hit += int(pred == s.tgt)
    return hit / max(len(samples), 1)


def run_training(cfg: MathTrainConfig) -> dict:
    """执行完整训练流程，并返回包含指标与路径的结果字典。"""

    set_seed(cfg.seed)
    device = auto_device(cfg.device)

    all_train = build_math_samples(list(cfg.train_templates), cfg.n_train_per_template, seed=cfg.seed)
    test_samples = build_math_samples(list(cfg.test_templates), cfg.n_test_per_template, seed=cfg.seed + 1)
    train_samples, val_samples = _split_train_val(all_train, cfg.val_ratio, cfg.seed)

    tok = _build_tokenizer(train_samples, test_samples)

    if cfg.model_variant == "seq2seq":
        train_ds = MathSeq2SeqDataset(train_samples, tok)
        train_loader = make_seq2seq_loader(train_ds, batch_size=cfg.batch_size, pad_idx=tok.pad_idx, shuffle=True)
    else:
        train_ds = MathDecoderOnlyDataset(train_samples, tok)
        train_loader = make_decoder_only_loader(train_ds, batch_size=cfg.batch_size, pad_idx=tok.pad_idx, shuffle=True)

    model = _build_model(cfg, tok).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    history: list[dict] = []
    best_val = -1.0
    best_state = None

    start = time()
    for epoch in range(1, cfg.epochs + 1):
        if cfg.model_variant == "seq2seq":
            train_loss = _train_one_epoch_seq2seq(model, train_loader, optimizer, device, tok.pad_idx)
        else:
            train_loss = _train_one_epoch_decoder_only(model, train_loader, optimizer, device, tok.pad_idx)

        val_exact = evaluate_exact_match(model, val_samples, tok, cfg.model_variant, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_exact": val_exact})
        tqdm.write(f"[math][{cfg.model_variant}] epoch={epoch} train_loss={train_loss:.4f} val_exact={val_exact:.4f}")

        if val_exact > best_val:
            best_val = val_exact
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_exact = evaluate_exact_match(model, test_samples, tok, cfg.model_variant, device)

    run_name = cfg.run_name or (
        f"math_{cfg.model_variant}_{cfg.norm_type}_{cfg.pos_encoding}_d{cfg.d_model}_h{cfg.nhead}"
    )
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{run_name}.pt"
    torch.save(model.state_dict(), model_path)

    result = {
        "run_name": run_name,
        "config": asdict(cfg),
        "device": str(device),
        "vocab_size": tok.vocab_size,
        "best_val_exact": best_val,
        "test_exact": test_exact,
        "history": history,
        "elapsed_sec": time() - start,
        "model_path": str(model_path),
    }

    (out_dir / f"{run_name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] {run_name}: val_exact={best_val:.4f}, test_exact={test_exact:.4f}, device={device}")
    return result
