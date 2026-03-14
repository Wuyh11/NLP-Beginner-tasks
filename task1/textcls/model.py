from __future__ import annotations

import torch


class LinearClassifier:
    """不依赖 torch.nn 的线性分类器。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        device: torch.device,
        seed: int = 42,
    ):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # 参数初始化：W 为高斯随机，b 为 0
        self.W = torch.empty((input_dim, num_classes), device=device).normal_(
            mean=0.0, std=0.02, generator=generator
        )
        self.b = torch.zeros((num_classes,), device=device)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W + self.b

    @staticmethod
    def _cross_entropy_with_grad(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回交叉熵损失及对 logits 的梯度。"""

        shifted = logits - logits.max(dim=1, keepdim=True).values
        exp = torch.exp(shifted)
        probs = exp / exp.sum(dim=1, keepdim=True)

        bs = logits.size(0)
        loss = -torch.log(probs[torch.arange(bs, device=logits.device), y] + 1e-12).mean()

        grad_logits = probs
        grad_logits[torch.arange(bs, device=logits.device), y] -= 1.0
        grad_logits /= bs
        return loss, grad_logits

    @staticmethod
    def _mse_with_grad(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 MSE 损失及对 logits 的梯度。"""

        target = torch.zeros_like(logits)
        target[torch.arange(logits.size(0), device=logits.device), y] = 1.0

        diff = logits - target
        loss = (diff * diff).mean()
        grad_logits = 2.0 * diff / (logits.size(0) * num_classes)
        return loss, grad_logits

    def loss_and_grads(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_name: str = "ce",
        weight_decay: float = 0.0,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        logits = self.logits(x)

        if loss_name == "ce":
            loss, grad_logits = self._cross_entropy_with_grad(logits, y)
        elif loss_name == "mse":
            loss, grad_logits = self._mse_with_grad(logits, y, logits.size(1))
        else:
            raise ValueError("loss_name 只支持 ce 或 mse")

        grad_W = x.t() @ grad_logits + weight_decay * self.W
        grad_b = grad_logits.sum(dim=0)
        return loss.item(), grad_W, grad_b

    def step(self, grad_W: torch.Tensor, grad_b: torch.Tensor, lr: float) -> None:
        """手写 SGD 参数更新。"""

        self.W -= lr * grad_W
        self.b -= lr * grad_b

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x).argmax(dim=1)
