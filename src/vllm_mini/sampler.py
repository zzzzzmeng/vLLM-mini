import torch

from vllm_mini.sampling_params import SamplingParams

class Sampler:
    def sample(self, logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        # logits shape can be [batch, vocab] or [batch, seq, vocab].
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        if params.temperature == 0.0:
            return torch.argmax(logits, dim=-1)

        scaled_logits = logits / params.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)