# examples/transformers_baseline.py
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_name = "sshleifer/tiny-gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    prompt = "Hello, vLLM-mini"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"device: {device}")
    print(f"prompt: {prompt}")
    print(f"output: {text}")
    print(f"elapsed: {elapsed:.3f}s")


if __name__ == "__main__":
    main()