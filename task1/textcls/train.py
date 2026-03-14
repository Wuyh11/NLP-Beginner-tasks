from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from time import time

import numpy as np
import torch
from tqdm import tqdm

from .data import Corpus, load_corpus
from .features import CountVectorizer
from .model import LinearClassifier


@dataclass(slots=True)
class TrainConfig:
    """Task-1 训练超参数配置。"""

    data_dir: str = "data"
    train_file: str = "new_train.tsv"
    test_file: str = "new_test.tsv"
    val_size: float = 0.1
    random_state: int = 42
    ngram: int = 1
    max_features: int = 20000
    min_freq: int = 2
    batch_size: int = 64
    epochs: int = 12
    lr: float = 0.1
    weight_decay: float = 0.0
    loss: str = "ce"
    eval_every: int = 50
    output_dir: str = "task1/outputs"
    device: str = "cpu"


def set_seed(seed: int) -> None:
    """设置随机种子，保证结果可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def accuracy(model: LinearClassifier, vec: CountVectorizer, texts: list[str], labels: list[int], device: torch.device) -> float:
    """分批推理并计算准确率。"""

    model_pred: list[int] = []
    bs = 256
    for i in range(0, len(texts), bs):
        x = vec.transform_batch(texts[i : i + bs], device=device)
        pred = model.predict(x).tolist()
        model_pred.extend(pred)

    correct = sum(int(p == y) for p, y in zip(model_pred, labels, strict=True))
    return correct / max(len(labels), 1)


def run_training(cfg: TrainConfig) -> dict:
    """执行完整训练流程并保存结果 JSON。"""

    set_seed(cfg.random_state)
    device = torch.device(cfg.device)

    corpus: Corpus = load_corpus(
        data_dir=cfg.data_dir,
        train_file=cfg.train_file,
        test_file=cfg.test_file,
        val_size=cfg.val_size,
        random_state=cfg.random_state,
    )

    vec = CountVectorizer(ngram=cfg.ngram, max_features=cfg.max_features, min_freq=cfg.min_freq)
    vec.fit(corpus.train.texts)

    num_classes = max(corpus.train.labels + corpus.val.labels + corpus.test.labels) + 1
    model = LinearClassifier(input_dim=vec.vocab_size, num_classes=num_classes, device=device, seed=cfg.random_state)

    history: list[dict] = []
    step = 0
    tic = time()

    for epoch in range(1, cfg.epochs + 1):
        # 每个 epoch 打乱训练集顺序
        indices = list(range(len(corpus.train.texts)))
        random.shuffle(indices)

        pbar = tqdm(range(0, len(indices), cfg.batch_size), desc=f"epoch {epoch}")
        for pos in pbar:
            batch_idx = indices[pos : pos + cfg.batch_size]
            batch_text = [corpus.train.texts[i] for i in batch_idx]
            batch_label = torch.tensor([corpus.train.labels[i] for i in batch_idx], dtype=torch.long, device=device)

            x = vec.transform_batch(batch_text, device=device)
            loss, grad_w, grad_b = model.loss_and_grads(
                x,
                batch_label,
                loss_name=cfg.loss,
                weight_decay=cfg.weight_decay,
            )
            model.step(grad_w, grad_b, lr=cfg.lr)

            step += 1
            pbar.set_postfix(loss=f"{loss:.4f}")

            # 按固定间隔在验证集评估
            if step % cfg.eval_every == 0:
                val_acc = accuracy(model, vec, corpus.val.texts, corpus.val.labels, device=device)
                history.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "loss": loss,
                        "val_acc": val_acc,
                    }
                )

    final_val = accuracy(model, vec, corpus.val.texts, corpus.val.labels, device=device)
    final_test = accuracy(model, vec, corpus.test.texts, corpus.test.labels, device=device)

    result = {
        "config": asdict(cfg),
        "vocab_size": vec.vocab_size,
        "final_val_acc": final_val,
        "final_test_acc": final_test,
        "history": history,
        "elapsed_sec": time() - tic,
    }

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"ng{cfg.ngram}_{cfg.loss}_lr{cfg.lr}"
    (out_dir / f"result_{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] vocab={vec.vocab_size} val_acc={final_val:.4f} test_acc={final_test:.4f}")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Task-1 文本分类训练")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--train-file", type=str, default="new_train.tsv")
    p.add_argument("--test-file", type=str, default="new_test.tsv")
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ngram", type=int, default=1)
    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--loss", type=str, choices=["ce", "mse"], default="ce")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--output-dir", type=str, default="task1/outputs")
    p.add_argument("--device", type=str, default="cpu")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        train_file=args.train_file,
        test_file=args.test_file,
        val_size=args.val_size,
        random_state=args.seed,
        ngram=args.ngram,
        max_features=args.max_features,
        min_freq=args.min_freq,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        eval_every=args.eval_every,
        output_dir=args.output_dir,
        device=args.device,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
