import pytest

from vllm_mini.sampling_params import SamplingParams


def test_default_sampling_params():
    params = SamplingParams()

    assert params.max_tokens == 16
    assert params.temperature == 0.0
    assert params.top_p == 1.0
    assert params.top_k == 0
    assert params.ignore_eos is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_tokens": 0}, "max_tokens"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": 1.1}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
        ({"top_k": -1}, "top_k"),
    ],
)
def test_invalid_sampling_params(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SamplingParams(**kwargs)
