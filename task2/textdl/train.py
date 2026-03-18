"""Task-2 训练主模块。

支持：
1) 三类模型（CNN/RNN/Transformer）；
2) 多种损失函数（CE/MSE/Focal）；
3) 多种优化器（Adam/AdamW/SGD）；
4) 可选 GloVe 初始化与 embedding 冻结。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from time import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW, Optimizer, SGD
from tqdm import tqdm

from .data import Vocab, build_dataloader, load_corpus
from .models import TextCNN, TextRNN, TextTransformer


class FocalLoss(nn.Module):
    """多分类 Focal Loss（缓解易分类样本主导）。"""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


@dataclass(slots=True)
class TrainConfig:
    """Task-2 训练与模型超参数配置。"""

    data_dir: str = "data"
    train_file: str = "new_train.tsv"
    test_file: str = "new_test.tsv"
    val_size: float = 0.1
    seed: int = 42
    min_freq: int = 1
    vocab_size: int = 50000
    max_len: int = 128
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.5
    embedding_dim: int = 200
    model_type: str = "cnn"  # cnn | rnn | transformer
    loss: str = "ce"  # ce | mse | focal
    optimizer: str = "adam"  # adam | adamw | sgd
    label_smoothing: float = 0.0
    # CNN
    num_filters: int = 100
    kernel_sizes: str = "3,4,5"
    # RNN
    rnn_hidden: int = 128
    rnn_type: str = "lstm"  # lstm | gru
    bidirectional: bool = True
    # Transformer
    nhead: int = 4
    transformer_layers: int = 2
    ff_dim: int = 512
    # GloVe
    glove_path: str = ""
    freeze_embedding: bool = False
    output_dir: str = "task2/outputs"
    run_name: str = ""
    device: str = "auto"


def set_seed(seed: int) -> None:
    """统一设置随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_kernel_sizes(raw: str) -> tuple[int, ...]:
    """将如 `"3,4,5"` 的字符串解析为卷积核大小元组。"""

    items = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("kernel_sizes 不能为空")
    return tuple(items)


def auto_device(preferred: str) -> torch.device:
    """根据参数自动选择 CPU/GPU。"""

    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def build_model(cfg: TrainConfig, vocab: Vocab, num_classes: int) -> nn.Module:
    """按配置构建 CNN / RNN / Transformer。"""

    if cfg.model_type == "cnn":
        return TextCNN(
            vocab_size=vocab.size,
            embed_dim=cfg.embedding_dim,
            num_classes=num_classes,
            pad_idx=vocab.pad_idx,
            num_filters=cfg.num_filters,
            kernel_sizes=parse_kernel_sizes(cfg.kernel_sizes),
            dropout=cfg.dropout,
        )
    if cfg.model_type == "rnn":
        return TextRNN(
            vocab_size=vocab.size,
            embed_dim=cfg.embedding_dim,
            hidden_size=cfg.rnn_hidden,
            num_classes=num_classes,
            pad_idx=vocab.pad_idx,
            rnn_type=cfg.rnn_type,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout,
        )
    if cfg.model_type == "transformer":
        if cfg.embedding_dim % cfg.nhead != 0:
            raise ValueError("embedding_dim 必须能被 nhead 整除")
        return TextTransformer(
            vocab_size=vocab.size,
            embed_dim=cfg.embedding_dim,
            num_classes=num_classes,
            pad_idx=vocab.pad_idx,
            nhead=cfg.nhead,
            num_layers=cfg.transformer_layers,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
        )
    raise ValueError("model_type 只支持 cnn / rnn / transformer")


def maybe_load_glove(embedding: nn.Embedding, vocab: Vocab, glove_path: str) -> float:
    """可选加载 GloVe 预训练向量，返回词表覆盖率。"""

    if not glove_path:
        return 0.0

    path = Path(glove_path)
    if not path.exists():
        raise FileNotFoundError(f"glove 文件不存在: {path}")

    vectors: dict[str, torch.Tensor] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            token = parts[0]
            vals = parts[1:]
            if len(vals) != embedding.embedding_dim:
                continue
            vectors[token] = torch.tensor([float(v) for v in vals], dtype=embedding.weight.dtype)

    hit = 0
    with torch.no_grad():
        for token, idx in vocab.stoi.items():
            vec = vectors.get(token)
            if vec is not None:
                embedding.weight[idx].copy_(vec)
                hit += 1

    return hit / max(vocab.size, 1)


def build_loss(cfg: TrainConfig, num_classes: int) -> nn.Module:
    """构建损失函数。"""

    if cfg.loss == "ce":
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    if cfg.loss == "focal":
        return FocalLoss(gamma=2.0)
    if cfg.loss == "mse":
        return nn.MSELoss()
    raise ValueError("loss 只支持 ce / focal / mse")


def compute_loss(loss_fn: nn.Module, logits: torch.Tensor, y: torch.Tensor, num_classes: int, loss_name: str) -> torch.Tensor:
    """统一计算不同损失函数的 batch loss。"""

    if loss_name == "mse":
        target = torch.zeros((y.size(0), num_classes), device=y.device)
        target.scatter_(1, y.unsqueeze(1), 1.0)
        probs = torch.softmax(logits, dim=1)
        return loss_fn(probs, target)
    return loss_fn(logits, y)


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> Optimizer:
    """构建优化器。"""

    if cfg.optimizer == "adam":
        return Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError("optimizer 只支持 adam / adamw / sgd")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[float, float]:
    """在给定数据集上计算 CE loss 与 accuracy。"""

    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    ce_loss = nn.CrossEntropyLoss()

    for x, lengths, y in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        logits = model(x, lengths)
        loss = ce_loss(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def run_training(cfg: TrainConfig) -> dict:
    """执行完整训练、验证、测试并保存结果。"""

    set_seed(cfg.seed)
    device = auto_device(cfg.device)

    corpus = load_corpus(
        data_dir=cfg.data_dir,
        train_file=cfg.train_file,
        test_file=cfg.test_file,
        val_size=cfg.val_size,
        random_state=cfg.seed,
    )

    vocab = Vocab(min_freq=cfg.min_freq, max_size=cfg.vocab_size)
    vocab.build(corpus.train.texts)

    train_loader = build_dataloader(
        corpus.train.texts,
        corpus.train.labels,
        vocab,
        max_len=cfg.max_len,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = build_dataloader(
        corpus.val.texts,
        corpus.val.labels,
        vocab,
        max_len=cfg.max_len,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    test_loader = build_dataloader(
        corpus.test.texts,
        corpus.test.labels,
        vocab,
        max_len=cfg.max_len,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    num_classes = max(corpus.train.labels + corpus.val.labels + corpus.test.labels) + 1
    model = build_model(cfg, vocab, num_classes).to(device)

    glove_coverage = 0.0
    if hasattr(model, "embedding"):
        glove_coverage = maybe_load_glove(model.embedding, vocab, cfg.glove_path)
        model.embedding.weight.requires_grad = not cfg.freeze_embedding

    optimizer = build_optimizer(cfg, model)
    loss_fn = build_loss(cfg, num_classes)

    history: list[dict] = []
    best_val_acc = -1.0
    best_state = None

    start = time()
    for epoch in range(1, cfg.epochs + 1):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}")
        for x, lengths, y in pbar:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = compute_loss(loss_fn, logits, y, num_classes, cfg.loss)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * y.size(0)
            seen += y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / max(seen, 1)
        # 每个 epoch 结束在验证集评估
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, device)

    run_name = cfg.run_name or f"{cfg.model_type}_{cfg.loss}_{cfg.optimizer}_lr{cfg.lr}"
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{run_name}.pt"
    torch.save(model.state_dict(), model_path)

    result = {
        "run_name": run_name,
        "config": asdict(cfg),
        "device": str(device),
        "vocab_size": vocab.size,
        "glove_coverage": glove_coverage,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": history,
        "elapsed_sec": time() - start,
        "model_path": str(model_path),
    }

    (out_dir / f"{run_name}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[Done] {run_name}: val={best_val_acc:.4f}, test={test_acc:.4f}, device={device}")
    return result


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    p = argparse.ArgumentParser(description="Task-2 深度学习文本分类训练")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--train-file", type=str, default="new_train.tsv")
    p.add_argument("--test-file", type=str, default="new_test.tsv")
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-freq", type=int, default=1)
    p.add_argument("--vocab-size", type=int, default=50000)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--embedding-dim", type=int, default=200)
    p.add_argument("--model", dest="model_type", choices=["cnn", "rnn", "transformer"], default="cnn")
    p.add_argument("--loss", choices=["ce", "mse", "focal"], default="ce")
    p.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    p.add_argument("--label-smoothing", type=float, default=0.0)

    p.add_argument("--num-filters", type=int, default=100)
    p.add_argument("--kernel-sizes", type=str, default="3,4,5")

    p.add_argument("--rnn-hidden", type=int, default=128)
    p.add_argument("--rnn-type", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--no-bidirectional", action="store_true")

    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--transformer-layers", type=int, default=2)
    p.add_argument("--ff-dim", type=int, default=512)

    p.add_argument("--glove-path", type=str, default="")
    p.add_argument("--freeze-embedding", action="store_true")
    p.add_argument("--output-dir", type=str, default="task2/outputs")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> None:
    """CLI 主函数：解析参数并执行训练。"""

    args = build_parser().parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        train_file=args.train_file,
        test_file=args.test_file,
        val_size=args.val_size,
        seed=args.seed,
        min_freq=args.min_freq,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        embedding_dim=args.embedding_dim,
        model_type=args.model_type,
        loss=args.loss,
        optimizer=args.optimizer,
        label_smoothing=args.label_smoothing,
        num_filters=args.num_filters,
        kernel_sizes=args.kernel_sizes,
        rnn_hidden=args.rnn_hidden,
        rnn_type=args.rnn_type,
        bidirectional=not args.no_bidirectional,
        nhead=args.nhead,
        transformer_layers=args.transformer_layers,
        ff_dim=args.ff_dim,
        glove_path=args.glove_path,
        freeze_embedding=args.freeze_embedding,
        output_dir=args.output_dir,
        run_name=args.run_name,
        device=args.device,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
