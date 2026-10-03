from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from src.client import DPConfig, train_client_model
from src.data import partition_dataset, set_seed
from src.evaluation import evaluate_model
from src.fedavg import aggregate_state_dicts
from src.model import SmallMNISTCNN


@dataclass
class ExperimentConfig:
    """Configuration for a federated learning run."""
    clients: int = 5
    rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 0.01
    seed: int = 42
    noise_multiplier: float = 0.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    train_subset_size: int = 1500
    test_subset_size: int = 300
    device: str = "cpu"


def run_federated_round(
    global_model: torch.nn.Module,
    client_partitions: list,
    batch_size: int,
    learning_rate: float,
    local_epochs: int,
    seed: int,
    device: str = "cpu",
    dp_config: DPConfig | None = None,
) -> tuple[torch.nn.Module, list[dict[str, float]], list[float]]:
    """Run one complete federated round across all participating clients."""
    local_states = []
    sample_counts = []
    epsilon_values = []

    for client_index, partition in enumerate(client_partitions):
        model, epsilon, _ = train_client_model(
            global_model=global_model,
            local_dataset=partition,
            batch_size=batch_size,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
            device=device,
            seed=seed + client_index,
            dp_config=dp_config,
        )
        local_states.append(model.state_dict())
        sample_counts.append(len(partition))
        epsilon_values.append(epsilon if epsilon is not None else 0.0)

    averaged_state = aggregate_state_dicts(local_states, sample_counts)
    global_model.load_state_dict(averaged_state)
    return global_model, [], epsilon_values


def run_federated_training(
    train_dataset,
    test_dataset,
    config: ExperimentConfig,
    dp_enabled: bool = False,
) -> tuple[torch.nn.Module, list[dict[str, float]], list[float], list[float]]:
    """Train a federated model for a configured number of rounds."""
    set_seed(config.seed)
    global_model = SmallMNISTCNN()
    client_partitions = partition_dataset(train_dataset, config.clients, config.seed)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    history: list[dict[str, float]] = []
    epsilon_history: list[float] = []
    round_losses: list[float] = []

    dp_config = None
    if dp_enabled:
        dp_config = DPConfig(
            noise_multiplier=config.noise_multiplier,
            max_grad_norm=config.max_grad_norm,
            delta=config.delta / config.rounds,
        )

    for round_number in range(1, config.rounds + 1):
        global_model, _, round_epsilons = run_federated_round(
            global_model=global_model,
            client_partitions=client_partitions,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            local_epochs=config.local_epochs,
            seed=config.seed,
            device=config.device,
            dp_config=dp_config,
        )
        metrics = evaluate_model(global_model, test_loader, device=config.device)
        history.append({
            "round": float(round_number),
            "loss": float(metrics["loss"]),
            "accuracy": float(metrics["accuracy"]),
        })
        epsilon_history.append(max(round_epsilons, default=0.0))
        round_losses.append(float(metrics["loss"]))

    return global_model, history, epsilon_history, round_losses


def summarize_round_history(history: list[dict[str, float]]) -> dict[str, float]:
    """Return the final round summary for reporting."""
    if not history:
        raise ValueError("No federated training history was produced.")
    final = history[-1]
    return {"accuracy": float(final["accuracy"]), "loss": float(final["loss"])}
