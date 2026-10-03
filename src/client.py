from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import validate_training_batch


@dataclass(frozen=True)
class DPConfig:
    """Client-side DP hyperparameters for Opacus training."""
    noise_multiplier: float
    max_grad_norm: float
    delta: float = 1e-5


def train_client_model(
    global_model: nn.Module,
    local_dataset: Any,
    batch_size: int,
    learning_rate: float,
    local_epochs: int,
    device: torch.device | str = "cpu",
    seed: int = 42,
    dp_config: DPConfig | None = None,
) -> tuple[nn.Module, float | None, float | None]:
    """Train a local client model using standard SGD or DP-SGD."""
    model = copy.deepcopy(global_model).to(device)
    criterion = nn.CrossEntropyLoss()
    local_loader = DataLoader(local_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    privacy_engine = None
    epsilon_value = None
    client_batch_size = batch_size

    if dp_config is not None and dp_config.noise_multiplier > 0:
        from opacus import PrivacyEngine

        model.train()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Secure RNG turned off.*",
                category=UserWarning,
                module="opacus",
            )
            privacy_engine = PrivacyEngine()
        model, optimizer, local_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=local_loader,
            noise_multiplier=dp_config.noise_multiplier,
            max_grad_norm=dp_config.max_grad_norm,
        )
        client_batch_size = local_loader.batch_size or batch_size

    model.train()
    for _ in range(local_epochs):
        for inputs, targets in local_loader:
            validate_training_batch([inputs, targets])
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Full backward hook is firing when gradients are computed with respect to module outputs.*",
                    category=UserWarning,
                )
                loss.backward()
            optimizer.step()

    if privacy_engine is not None and dp_config is not None:
        epsilon_value = privacy_engine.get_epsilon(dp_config.delta)

    usable_model = model._module if hasattr(model, "_module") else model
    return usable_model, epsilon_value, float(client_batch_size)
