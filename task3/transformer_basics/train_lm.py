from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from time import time

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from .data_lm import LMDataset, LMTokenizer, build_lm_split, build_synthetic_corpus, make_lm_loader
from .models import DecoderOnlyTransformer, TransformerConfig, make_subsequent_mask


@dataclass(slots=True)
class LMTrainConfig:
    run_name: str = ""
    output_dir: str = "task3/outputs/lm"
    seed: int = 42
    device: str = "auto"

    # 语料与 tokenizer
    n_sentences: int = 8000
    tokenizer_mode: str = "word"  # char | word
    vocab_size: int = 500

    # 训练样本
    block_size: int = 64
    batch_size: int = 64

    # 模型
    num_layers: int = 4
    d_model: int = 192
    nhead: int = 6
    ff_dim: int = 384
    dropout: float = 0.1
    max_len: int = 256
    norm_type: str = "layernorm"  # layernorm | rmsnorm
    pos_encoding: str = "sinusoidal"  # sinusoidal | rope | none
    rope_base: float = 10000.0

    # 优化
    epochs: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.01


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_device(preferred: str) -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def _build_model(cfg: LMTrainConfig, tok: LMTokenizer) -> DecoderOnlyTransformer:
    model_cfg = TransformerConfig(
        vocab_size=tok.vocab_size,
        pad_idx=tok.pad_idx,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        norm_type=cfg.norm_type,
        pos_encoding=cfg.pos_encoding,
        rope_base=cfg.rope_base,
    )
    return DecoderOnlyTransformer(model_cfg, num_layers=cfg.num_layers)


def _loss_on_batch(model: DecoderOnlyTransformer, x: torch.Tensor, y: torch.Tensor, pad_idx: int) -> torch.Tensor:
    key_pad = x.eq(pad_idx)
    attn_mask = make_subsequent_mask(x.size(1), x.device)
    logits = model(x, key_padding_mask=key_pad, attn_mask=attn_mask)
    ce = nn.CrossEntropyLoss(ignore_index=pad_idx)
    return ce(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def _train_one_epoch(model: DecoderOnlyTransformer, loader, optimizer, device: torch.device, pad_idx: int) -> float:
    model.train()
    total_loss = 0.0
    total_steps = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = _loss_on_batch(model, x, y, pad_idx)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1
    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate_perplexity(model: DecoderOnlyTransformer, loader, device: torch.device, pad_idx: int) -> float:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        loss = _loss_on_batch(model, x, y, pad_idx)
        total_loss += float(loss.item())
        total_steps += 1
    mean_loss = total_loss / max(total_steps, 1)
    return float(math.exp(mean_loss))


@torch.no_grad()
def generate_text(
    model: DecoderOnlyTransformer,
    tok: LMTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 40,
) -> str:
    model.eval()
    ids = tok.encode(prompt, add_bos=True, add_eos=False)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        key_pad = x.eq(tok.pad_idx)
        attn_mask = make_subsequent_mask(x.size(1), x.device)
        logits = model(x, key_padding_mask=key_pad, attn_mask=attn_mask)
        nxt = int(logits[0, -1].argmax().item())
        ids.append(nxt)
        if nxt == tok.eos_idx:
            break
    return tok.decode(ids)


def run_training(cfg: LMTrainConfig) -> dict:
    set_seed(cfg.seed)
    device = auto_device(cfg.device)

    corpus = build_synthetic_corpus(n_sentences=cfg.n_sentences, seed=cfg.seed)
    tok = LMTokenizer(mode=cfg.tokenizer_mode, vocab_size=cfg.vocab_size)
    tok.fit(corpus)

    split = build_lm_split(corpus, tok, train_ratio=0.8, val_ratio=0.1)
    train_ds = LMDataset(split.train_ids, block_size=cfg.block_size)
    val_ds = LMDataset(split.val_ids, block_size=cfg.block_size)
    test_ds = LMDataset(split.test_ids, block_size=cfg.block_size)

    train_loader = make_lm_loader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = make_lm_loader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = make_lm_loader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = _build_model(cfg, tok).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: list[dict] = []
    best_val_ppl = float("inf")
    best_state = None

    start = time()
    for epoch in range(1, cfg.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, device, tok.pad_idx)
        val_ppl = evaluate_perplexity(model, val_loader, device, tok.pad_idx)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_ppl": val_ppl})
        print(f"[lm] epoch={epoch} train_loss={train_loss:.4f} val_ppl={val_ppl:.3f}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_ppl = evaluate_perplexity(model, test_loader, device, tok.pad_idx)
    sample = generate_text(model, tok, prompt="the model", device=device, max_new_tokens=24)

    run_name = cfg.run_name or (
        f"lm_{cfg.tokenizer_mode}_{cfg.norm_type}_{cfg.pos_encoding}_v{cfg.vocab_size}"
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
        "best_val_ppl": best_val_ppl,
        "test_ppl": test_ppl,
        "sample_text": sample,
        "history": history,
        "elapsed_sec": time() - start,
        "model_path": str(model_path),
    }

    (out_dir / f"{run_name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] {run_name}: val_ppl={best_val_ppl:.3f}, test_ppl={test_ppl:.3f}, device={device}")
    return result
