from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MNIST_DATASET_ROOT = DATA_DIR / "MNIST"

MNIST_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible experiments."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mnist_datasets(
    train_subset_size: int | None = None,
    test_subset_size: int | None = None,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Load MNIST and optionally sample a smaller subset for a quick demo."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_dataset = datasets.MNIST(
        root=str(MNIST_DATASET_ROOT),
        train=True,
        download=True,
        transform=MNIST_TRANSFORM,
    )
    test_dataset = datasets.MNIST(
        root=str(MNIST_DATASET_ROOT),
        train=False,
        download=True,
        transform=MNIST_TRANSFORM,
    )

    if train_subset_size is not None:
        rng = np.random.default_rng(seed)
        subset_indices = rng.choice(len(train_dataset), size=train_subset_size, replace=False)
        train_dataset = Subset(train_dataset, subset_indices.tolist())
    if test_subset_size is not None:
        rng = np.random.default_rng(seed + 1)
        subset_indices = rng.choice(len(test_dataset), size=test_subset_size, replace=False)
        test_dataset = Subset(test_dataset, subset_indices.tolist())

    return train_dataset, test_dataset


def partition_dataset(dataset: Dataset, num_clients: int, seed: int = 42) -> list[Subset]:
    """Split a dataset into deterministic client subsets without overlap."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if num_clients > len(dataset):
        raise ValueError("num_clients cannot exceed dataset size.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    partitions = np.array_split(indices, num_clients)
    return [Subset(dataset, indices.tolist()) for indices in partitions]


def make_client_loader(dataset: Dataset, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    """Construct a DataLoader for a client's local dataset."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def validate_training_batch(batch: Iterable[torch.Tensor]) -> None:
    """Guardrail to catch NaN/Inf values before synchronization steps."""
    for tensor in batch:
        if not torch.isfinite(tensor).all():
            raise ValueError("Encountered NaN/Inf values in the training batch.")
