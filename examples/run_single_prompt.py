import json
import time

from vllm_mini import LLM, SamplingParams


def main() -> None:
    model_name = "sshleifer/tiny-gpt2"
    device = "cpu"
    prompt = "Hello, vLLM-mini!"

    llm = LLM(model_name, device=device)
    params = SamplingParams(max_tokens=16, temperature=0.0)

    start = time.perf_counter()
    output = llm.generate([prompt], params)[0]
    elapsed_seconds = time.perf_counter() - start

    result = {
        "model": model_name,
        "device": device,
        "prompt": prompt,
        "text": output["text"],
        "token_ids": output["token_ids"],
        "token_count": len(output["token_ids"]),
        "finish_reason": output["finish_reason"],
        "elapsed_seconds": round(elapsed_seconds, 4),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
