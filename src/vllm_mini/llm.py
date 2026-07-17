import torch

from vllm_mini.model_runner import ModelRunner
from vllm_mini.sampler import Sampler
from vllm_mini.sampling_params import SamplingParams
from vllm_mini.sequence import Sequence, SequenceStatus


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

        for seq_id, prompt in enumerate(prompts):
            output = self._generate_one(seq_id, prompt, params)
            outputs.append(output)
        return outputs

    def _generate_one(self, seq_id: int, prompt: str, params: SamplingParams) -> dict:
        input_ids = self.runner.encode(prompt)
        sequence = Sequence(
            seq_id=seq_id,
            prompt=prompt,
            prompt_token_ids=input_ids[0].tolist(),
        )
        sequence.mark_running()

        for _ in range(params.max_tokens):
            logits = self.runner.forward(input_ids)
            next_token = self.sampler.sample(logits, params)

            token_id = next_token.item()
            sequence.append_token(token_id)

            input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
            eos_token_id = self.runner.tokenizer.eos_token_id

            if eos_token_id is not None and token_id == eos_token_id and not params.ignore_eos:
                sequence.finish(SequenceStatus.FINISHED_EOS)
                break

        if not sequence.is_finished:
            sequence.finish(SequenceStatus.FINISHED_LENGTH)

        text = self.runner.decode(sequence.generated_token_ids)

        return {
            "text": text,
            "token_ids": sequence.generated_token_ids,
            "finish_reason": sequence.finish_reason,
        }
