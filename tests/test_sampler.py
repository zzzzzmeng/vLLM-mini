import torch

from vllm_mini.sampler import Sampler
from vllm_mini.sampling_params import SamplingParams


def test_sample_greedy_from_2d_logits():
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [1.5, 0.3, 0.7],
        ]
    )
    params = SamplingParams(temperature=0.0)

    next_tokens = Sampler().sample(logits, params)

    assert next_tokens.tolist() == [1, 0]
    assert next_tokens.shape == torch.Size([2])


def test_sample_greedy_from_3d_logits_uses_last_position():
    logits = torch.tensor(
        [
            [
                [9.0, 0.1, 0.2],
                [0.1, 0.2, 3.0],
            ]
        ]
    )
    params = SamplingParams(temperature=0.0)

    next_tokens = Sampler().sample(logits, params)

    assert next_tokens.tolist() == [2]
    assert next_tokens.shape == torch.Size([1])


def test_sample_temperature_returns_one_token_per_batch_item():
    torch.manual_seed(0)
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [1.5, 0.3, 0.7],
        ]
    )
    params = SamplingParams(temperature=1.0)

    next_tokens = Sampler().sample(logits, params)

    assert next_tokens.shape == torch.Size([2])
    assert next_tokens.dtype == torch.long
    assert torch.all((next_tokens >= 0) & (next_tokens < logits.shape[-1]))
