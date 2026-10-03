from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device | str = "cpu") -> dict[str, float]:
    """Evaluate a model and return loss and accuracy on a DataLoader."""
    device = torch.device(device)
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * inputs.size(0)
            total_correct += (logits.argmax(1) == targets).sum().item()
            total_examples += inputs.size(0)

    avg_loss = total_loss / max(total_examples, 1)
    accuracy = (total_correct / max(total_examples, 1)) * 100.0
    return {"loss": avg_loss, "accuracy": accuracy}


def assert_finite_metrics(metrics: dict[str, float]) -> None:
    """Fail fast if a metric contains non-finite values."""
    for name, value in metrics.items():
        if not torch.isfinite(torch.tensor(float(value))).item():
            raise ValueError(f"Metric {name} is not finite: {value!r}")
