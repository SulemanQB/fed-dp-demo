from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Sequence

import torch


def weighted_average(param_groups: Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]], sample_counts: Sequence[int]) -> torch.Tensor | list[torch.Tensor]:
    """Compute a weighted average across a set of model parameters or parameter groups."""
    if len(param_groups) != len(sample_counts):
        raise ValueError("Parameter groups and sample counts must have the same length.")
    if not param_groups:
        raise ValueError("At least one parameter group is required.")

    total_samples = sum(sample_counts)
    if total_samples <= 0:
        raise ValueError("Sample counts must sum to a positive number.")

    first = param_groups[0]
    if isinstance(first, torch.Tensor):
        return sum(param * count for param, count in zip(param_groups, sample_counts)) / total_samples

    combined: list[torch.Tensor] = []
    for idx in range(len(first)):
        weighted = torch.stack(
            [group[idx].to(dtype=first[idx].dtype) * count for group, count in zip(param_groups, sample_counts)],
            dim=0,
        ).sum(dim=0) / total_samples
        combined.append(weighted)
    return combined


def aggregate_state_dicts(state_dicts: Sequence[OrderedDict[str, torch.Tensor]], sample_counts: Sequence[int]) -> OrderedDict[str, torch.Tensor]:
    """Weighted-average a list of model state dictionaries using FedAvg."""
    if not state_dicts:
        raise ValueError("No client models were provided for aggregation.")
    if len(state_dicts) != len(sample_counts):
        raise ValueError("State dictionaries and sample counts must align.")

    aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
    total_samples = sum(sample_counts)
    for key in state_dicts[0].keys():
        stacked = torch.stack([
            state[key].to(dtype=state_dicts[0][key].dtype) * count for state, count in zip(state_dicts, sample_counts)
        ], dim=0)
        aggregated[key] = stacked.sum(dim=0) / total_samples
    return aggregated
