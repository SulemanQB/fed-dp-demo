from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.data import load_mnist_datasets, set_seed
from src.server import ExperimentConfig, run_federated_training, summarize_round_history

RESULTS_PATH = Path(__file__).resolve().parents[1] / "experiments" / "results.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated learning demonstration with differential privacy.")
    parser.add_argument("--mode", choices=["fedavg", "experiment", "smoke"], default="fedavg")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-multiplier", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--train-subset-size", type=int, default=1500)
    parser.add_argument("--test-subset-size", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        delta=args.delta,
        train_subset_size=args.train_subset_size,
        test_subset_size=args.test_subset_size,
    )


def _append_results(rows: list[dict], overwrite: bool = False) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "configuration",
        "num_clients",
        "rounds",
        "local_epochs",
        "epsilon",
        "delta",
        "noise_multiplier",
        "max_grad_norm",
        "accuracy",
        "test_loss",
        "seed",
        "run_id",
    ]

    exists = RESULTS_PATH.exists()
    if exists and not overwrite:
        with RESULTS_PATH.open("a", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            if file_obj.tell() == 0:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return

    with RESULTS_PATH.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_single_config(
    configuration_name: str,
    config: ExperimentConfig,
    dp_enabled: bool = False,
    run_id: str | None = None,
) -> dict:
    set_seed(config.seed)
    train_dataset, test_dataset = load_mnist_datasets(
        train_subset_size=config.train_subset_size,
        test_subset_size=config.test_subset_size,
        seed=config.seed,
    )
    _, history, epsilon_history, _ = run_federated_training(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
        dp_enabled=dp_enabled,
    )
    summary = summarize_round_history(history)
    episode_epsilon = float(sum(epsilon_history)) if epsilon_history else float("nan")
    row = {
        "configuration": configuration_name,
        "num_clients": config.clients,
        "rounds": config.rounds,
        "local_epochs": config.local_epochs,
        "epsilon": "N/A" if not dp_enabled else round(episode_epsilon, 6),
        "delta": config.delta,
        "noise_multiplier": config.noise_multiplier,
        "max_grad_norm": config.max_grad_norm,
        "accuracy": round(summary["accuracy"], 6),
        "test_loss": round(summary["loss"], 6),
        "seed": config.seed,
        "run_id": run_id or "manual",
    }
    return row


def run_experiment_suite(args: argparse.Namespace) -> list[dict]:
    base = _build_config(args)
    strong_noise = args.noise_multiplier * 3 if args.noise_multiplier > 0 else 1.5
    medium_noise = args.noise_multiplier * 2 if args.noise_multiplier > 0 else 1.0
    weak_noise = args.noise_multiplier if args.noise_multiplier > 0 else 0.5
    max_grad_norm = args.max_grad_norm if args.max_grad_norm > 0 else 1.0

    configs = [
        ("FedAvg", base, False),
        (
            "DP-Strong",
            ExperimentConfig(
                clients=base.clients,
                rounds=base.rounds,
                local_epochs=base.local_epochs,
                batch_size=base.batch_size,
                learning_rate=base.learning_rate,
                seed=base.seed,
                noise_multiplier=strong_noise,
                max_grad_norm=max_grad_norm,
                delta=base.delta,
                train_subset_size=base.train_subset_size,
                test_subset_size=base.test_subset_size,
                device=base.device,
            ),
            True,
        ),
        (
            "DP-Medium",
            ExperimentConfig(
                clients=base.clients,
                rounds=base.rounds,
                local_epochs=base.local_epochs,
                batch_size=base.batch_size,
                learning_rate=base.learning_rate,
                seed=base.seed,
                noise_multiplier=medium_noise,
                max_grad_norm=max_grad_norm,
                delta=base.delta,
                train_subset_size=base.train_subset_size,
                test_subset_size=base.test_subset_size,
                device=base.device,
            ),
            True,
        ),
        (
            "DP-Weak",
            ExperimentConfig(
                clients=base.clients,
                rounds=base.rounds,
                local_epochs=base.local_epochs,
                batch_size=base.batch_size,
                learning_rate=base.learning_rate,
                seed=base.seed,
                noise_multiplier=weak_noise,
                max_grad_norm=max_grad_norm,
                delta=base.delta,
                train_subset_size=base.train_subset_size,
                test_subset_size=base.test_subset_size,
                device=base.device,
            ),
            True,
        ),
    ]
    rows = []
    for index, (name, cfg, dp_enabled) in enumerate(configs, start=1):
        run_id = f"run-{index}"
        rows.append(run_single_config(name, cfg, dp_enabled=dp_enabled, run_id=run_id))
        if index == 1:
            print(f"[{name}] accuracy={rows[-1]['accuracy']:.4f}% loss={rows[-1]['test_loss']:.4f}")
        else:
            print(f"[{name}] epsilon={rows[-1]['epsilon']} accuracy={rows[-1]['accuracy']:.4f}% loss={rows[-1]['test_loss']:.4f}")
    return rows


def run_cli() -> None:
    args = _parse_args()

    if args.mode == "fedavg":
        config = _build_config(args)
        row = run_single_config("FedAvg", config, dp_enabled=False, run_id="baseline")
        print(f"FedAvg | Accuracy: {row['accuracy']:.2f}% | Loss: {row['test_loss']:.4f}")
    elif args.mode == "experiment":
        rows = run_experiment_suite(args)
        _append_results(rows, overwrite=args.overwrite)
        print(f"Saved experiment results to {RESULTS_PATH}")
    elif args.mode == "smoke":
        from scripts.smoke_test import smoke_test

        smoke_test()


if __name__ == "__main__":
    run_cli()
