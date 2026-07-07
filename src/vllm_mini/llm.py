import torch

from vllm_mini.model_runner import ModelRunner
from vllm_mini.sampler import Sampler
from vllm_mini.sampling_params import SamplingParams

class LLM:
    def __init__(self, model_name_or_path: str, device: str | None = None):
        self.runner = ModelRunner(model_name_or_path, device)
        self.sampler = Sampler()
    
    def generate(
            self,
            prompts: list[str],
            sampling_params: SamplingParams | None = None,
    ) -> list[dict]:
        params = sampling_params or SamplingParams()
        outputs = []
        for prompt in prompts:
            output = self._generate_one(prompt, params)
            outputs.append(output)
        return outputs

    def _generate_one(self, prompt: str, params: SamplingParams) -> dict:
        input_ids = self.runner.encode(prompt)
        prompt_length = input_ids.shape[-1]

        finish_reason = "length"

        for _ in range(params.max_tokens):
            logits = self.runner.forward(input_ids)
            next_token = self.sampler.sample(logits, params)

            input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
            token_id = next_token.item()
            eos_token_id = self.runner.tokenizer.eos_token_id

            if eos_token_id is not None and token_id == eos_token_id and not params.ignore_eos:
                finish_reason = "eos"
                break
        
        generated_ids = input_ids[0, prompt_length:].tolist()
        text = self.runner.decode(generated_ids)
        return {
            "text": text,
            "token_ids": generated_ids,
            "finish_reason": finish_reason,
        }
