from types import SimpleNamespace

import torch

from vllm_mini.llm import LLM
from vllm_mini.sampling_params import SamplingParams


class FakeRunner:
    def __init__(self, eos_token_id: int | None = None):
        self.tokenizer = SimpleNamespace(eos_token_id=eos_token_id)

    def encode(self, prompt: str) -> torch.Tensor:
        return torch.tensor([[len(prompt)]])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, input_ids.shape[-1], 8))

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class FakeSampler:
    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids
        self.index = 0

    def sample(self, logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        token_id = self.token_ids[self.index]
        self.index += 1
        return torch.tensor([token_id])


def build_fake_llm(token_ids: list[int], eos_token_id: int | None = None) -> LLM:
    llm = object.__new__(LLM)
    llm.runner = FakeRunner(eos_token_id=eos_token_id)
    llm.sampler = FakeSampler(token_ids)
    return llm


def test_generate_finishes_by_length():
    llm = build_fake_llm([3, 4])

    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=2, temperature=0.0))

    assert outputs == [
        {
            "text": "3 4",
            "token_ids": [3, 4],
            "finish_reason": "length",
        }
    ]


def test_generate_finishes_by_eos():
    llm = build_fake_llm([2], eos_token_id=2)

    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=4, temperature=0.0))

    assert outputs == [
        {
            "text": "2",
            "token_ids": [2],
            "finish_reason": "eos",
        }
    ]


def test_generate_keeps_prompt_order_for_multiple_prompts():
    llm = build_fake_llm([3, 4, 5, 6])

    outputs = llm.generate(["Hello", "I love"], SamplingParams(max_tokens=2, temperature=0.0))

    assert outputs[0]["token_ids"] == [3, 4]
    assert outputs[1]["token_ids"] == [5, 6]
