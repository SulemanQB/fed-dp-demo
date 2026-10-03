from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mnist_datasets, partition_dataset, set_seed
from src.fedavg import aggregate_state_dicts
from src.model import SmallMNISTCNN
from src.server import DPConfig, ExperimentConfig, run_federated_round
from src.evaluation import evaluate_model


def smoke_test() -> None:
    """Run a tiny end-to-end FL smoke test to validate core components quickly."""
    print("====================================")
    print("FED-DP DEMO — SMOKE TEST")
    print("====================================")

    set_seed(7)
    train_dataset, test_dataset = load_mnist_datasets(train_subset_size=200, test_subset_size=50, seed=7)
    print("Dataset ............... PASS")

    partitions = partition_dataset(train_dataset, num_clients=2, seed=7)
    assert len(partitions) == 2
    print("Client partition ...... PASS")

    config = ExperimentConfig(
        clients=2,
        rounds=1,
        local_epochs=1,
        batch_size=32,
        learning_rate=0.01,
        seed=7,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        delta=1e-5,
        train_subset_size=200,
        test_subset_size=50,
    )
    global_model = SmallMNISTCNN()
    global_model, _, _ = run_federated_round(
        global_model=global_model,
        client_partitions=partitions,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        local_epochs=config.local_epochs,
        seed=config.seed,
        device=config.device,
        dp_config=None,
    )
    print("Local training ........ PASS")

    aggregated = aggregate_state_dicts(
        [client_model.state_dict() for client_model in [global_model, global_model]],
        [len(partitions[0]), len(partitions[1])],
    )
    assert aggregated is not None
    print("FedAvg aggregation .... PASS")

    metrics = evaluate_model(global_model, __import__("torch").utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False), device="cpu")
    assert metrics["loss"] >= 0.0 and metrics["accuracy"] >= 0.0
    print("Evaluation ............ PASS")

    dp_model = SmallMNISTCNN()
    dp_config = DPConfig(noise_multiplier=0.5, max_grad_norm=1.0, delta=1e-5)
    dp_model, dp_epsilon, _ = __import__("src.client", fromlist=["train_client_model"]).train_client_model(
        global_model=dp_model,
        local_dataset=partitions[0],
        batch_size=32,
        learning_rate=0.01,
        local_epochs=1,
        device="cpu",
        seed=7,
        dp_config=dp_config,
    )
    assert dp_model is not None
    assert dp_epsilon is not None
    print("DP training ........... PASS")

    for name, value in metrics.items():
        if any(not __import__("math").isfinite(v) for v in [value]):
            raise ValueError(f"Non-finite metric encountered: {name}={value}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    smoke_test()
