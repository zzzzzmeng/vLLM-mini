# Sprint 1: Single Request MVP

## 目标

实现最小单请求生成闭环：

```python
from vllm_mini import LLM, SamplingParams

llm = LLM("your-model-path-or-id", device="cuda")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=16, temperature=0.0))
print(outputs[0]["text"])
```

## 不做什么

这一阶段不做：

- batching
- Scheduler
- paged KV cache
- continuous batching
- server
- Triton/CUDA kernel
- tensor parallel

## 建议文件

你可以从这些文件开始写：

```text
src/vllm_mini/
  __init__.py
  llm.py
  sampling_params.py
  sampler.py
  model_runner.py
examples/
  run_single_prompt.py
tests/
  test_sampling_params.py
  test_sampler.py
```

## 步骤

### Step 1. 环境 smoke test

任务：

- 用 Transformers 直接加载一个小模型。
- 直接调用 `model.generate` 生成 8 到 16 个 token。

验收：

- 你能确认模型、tokenizer、device 都可用。
- 记录模型名、显存占用和生成耗时。

### Step 2. SamplingParams

任务：

- 写一个 dataclass。
- 支持 `max_tokens`、`temperature`、`top_p`、`top_k`、`ignore_eos`。
- 做基本合法性校验。

验收：

- `max_tokens > 0`
- `temperature >= 0`
- `0 < top_p <= 1`
- `top_k >= 0`
- 单元测试覆盖默认值和非法参数。

### Step 3. Sampler greedy

任务：

- 输入最后一个位置的 logits。
- 当 `temperature == 0` 时返回 argmax token。

验收：

- 构造一个小 logits tensor，测试能选出最大值下标。
- 返回 shape 清晰，比如 `[batch_size]`。

### Step 4. LLM and ModelRunner

任务：

- `LLM` 负责 tokenizer、model runner、输出整理。
- `ModelRunner` 负责模型 forward。
- 先可以用 Hugging Face 模型的 `use_cache=True`。

验收：

- 单 prompt 能生成文本。
- 每步 decode 只追加一个 token。
- 达到 EOS 或 max_tokens 停止。

### Step 5. Experiment

任务：

- 写 `examples/run_single_prompt.py`。
- 输入固定 prompt。
- 打印 prompt、output、token 数、耗时。

验收：

- 一条命令可复现运行。
- 输出格式稳定，便于后续 benchmark 扩展。

## 推荐汇报格式

你完成每一步后，把下面信息发给我：

```text
完成 Step N
改动文件：
- ...

运行命令：
- ...

结果：
- ...

问题：
- ...
```

我会继续帮你 review 结构和提示下一步。
