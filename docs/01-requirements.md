# Requirements

## 学习目标

这个项目要让你真正理解一个 vLLM-like engine 的核心问题：

- 请求如何从 prompt 变成 token 序列。
- prefill 和 decode 为什么要拆开。
- KV cache 如何减少重复计算。
- 多请求如何被调度进同一个 batch。
- block manager 和 block table 如何支撑 paged KV cache。
- 采样、停止条件和输出拼接如何组成对外 API。

## 功能需求

### R1. Public API

优先级：MVP

提供一个简化接口：

```python
llm = LLM(model_path, device="cuda")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=16))
```

验收标准：

- 输入一个或多个 prompt。
- 返回每个 prompt 对应的 text、token_ids、finish_reason。
- 不暴露内部 Scheduler、Runner、KV cache 细节。

### R2. SamplingParams

优先级：MVP

支持基础生成参数：

- `max_tokens`
- `temperature`
- `top_p`
- `top_k`
- `ignore_eos`

验收标准：

- 参数有默认值和合法性校验。
- greedy 模式可复现。
- 后续可以扩展 stop words、presence penalty 等参数。

### R3. Model Loading

优先级：MVP

加载 Hugging Face causal LM 和 tokenizer。

验收标准：

- 能加载本地路径或 Hugging Face model id。
- 模型放到指定 device。
- 推理时进入 eval/no_grad 或 inference_mode。

### R4. Single Request Decode Loop

优先级：MVP

先实现单请求生成闭环。

验收标准：

- 完成 tokenization -> prefill -> next-token sampling -> decode loop -> detokenization。
- 支持 EOS 停止和 max_tokens 停止。
- 结果能和 Transformers `generate` 的 greedy 输出做基本对齐。

### R5. Batch Offline Generation

优先级：Phase 2

支持多个 prompt 一起离线生成。

验收标准：

- 支持不同 prompt 长度。
- 能为每个 Sequence 维护独立状态。
- 某个请求结束后不会影响其他请求继续 decode。

### R6. Explicit KV Cache

优先级：Phase 3

从依赖 Transformers 内部 `generate` 迁移到自己控制 prefill/decode 和 KV cache。

验收标准：

- prefill 产生 cache。
- decode 每次只输入新 token。
- 和无 cache 版本相比，输出一致。

### R7. Scheduler

优先级：Phase 4

实现最小调度器。

验收标准：

- 有 waiting、running、finished 三类状态。
- 能按 token budget 或 sequence 数限制 batch。
- 支持新请求在 decode 过程中加入。

### R8. Block Manager and Paged KV Cache

优先级：Phase 5

实现 block-based KV cache 管理。

验收标准：

- 每个 Sequence 有 block table。
- BlockManager 能 allocate/free。
- prompt 和 generated tokens 能映射到 slot。
- 暂时可以先用 PyTorch tensor 模拟，不必立刻写 Triton kernel。

### R9. Benchmark

优先级：Phase 2+

建立可重复 benchmark。

验收标准：

- 记录输入长度、输出长度、总 tokens、耗时、tokens/s。
- 至少有一个 Transformers baseline。
- 每次优化前后可对比。

## 非功能需求

- 可读性优先于炫技。
- 每个核心模块都要有最小单元测试。
- 先保证单卡单进程。
- 日志清楚，便于定位调度和 cache 状态。
- 所有阶段都能独立验收。

## 暂不做

- 多机多卡。
- 生产级 OpenAI API 兼容细节。
- 复杂量化内核。
- 自定义 CUDA kernel。
- 完整复刻 vLLM 的所有边界条件。

