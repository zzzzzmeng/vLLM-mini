from dataclasses import dataclass

@dataclass
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    ignore_eos: bool = False

    def __post__init__(self):
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0")
        if self.temperature < 0.0 or self.temperature > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        if self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        if self.top_k < 0:
            raise ValueError("top_k must be greater than or equal to 0")