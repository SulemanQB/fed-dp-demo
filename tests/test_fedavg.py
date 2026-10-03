import torch

from src.fedavg import weighted_average


def test_weighted_average_single_parameter():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])

    result = weighted_average([a, b], [1, 3])

    expected = torch.tensor([2.5, 3.5])
    assert torch.allclose(result, expected)


def test_weighted_average_multiple_parameters():
    a = [torch.tensor([1.0, 2.0]), torch.tensor([5.0, 6.0])]
    b = [torch.tensor([3.0, 4.0]), torch.tensor([7.0, 8.0])]

    result = weighted_average([a, b], [2, 1])

    expected = [torch.tensor([1.6666667, 2.6666667]), torch.tensor([5.6666667, 6.6666667])]
    for actual, target in zip(result, expected):
        assert torch.allclose(actual, target)


def test_weighted_average_accounts_for_sample_count():
    p1 = torch.tensor([10.0, 20.0])
    p2 = torch.tensor([0.0, 0.0])

    result = weighted_average([p1, p2], [3, 1])

    assert torch.allclose(result, torch.tensor([7.5, 15.0]))
