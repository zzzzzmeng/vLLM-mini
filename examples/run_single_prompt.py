from vllm_mini import LLM, SamplingParams

def main() -> None:
    llm = LLM("sshleifer/tiny-gpt2", device="cpu")
    params = SamplingParams(max_tokens=16, temperature=0.0)
    output = llm.generate(["Hello, vLLM-mini!"], params)

    print(output[0])

if __name__ == "__main__":
    main()