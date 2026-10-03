import torch

from src.data import partition_dataset


def test_partition_dataset_shape_and_coverage():
    data = torch.utils.data.TensorDataset(
        torch.arange(30, dtype=torch.float32).reshape(15, 2),
        torch.arange(15, dtype=torch.long),
    )

    clients = partition_dataset(data, num_clients=5, seed=42)

    assert len(clients) == 5
    total = sum(len(client) for client in clients)
    assert total == len(data)
    seen = set()
    for client in clients:
        for idx in client.indices:
            assert idx not in seen
            seen.add(idx)
    assert len(seen) == len(data)


def test_partition_dataset_is_reproducible():
    data = torch.utils.data.TensorDataset(
        torch.arange(60, dtype=torch.float32).reshape(30, 2),
        torch.arange(30, dtype=torch.long),
    )

    first = partition_dataset(data, num_clients=6, seed=7)
    second = partition_dataset(data, num_clients=6, seed=7)

    first_indices = [tuple(client.indices) for client in first]
    second_indices = [tuple(client.indices) for client in second]

    assert first_indices == second_indices
