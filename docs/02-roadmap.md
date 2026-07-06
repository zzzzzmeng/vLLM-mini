# Roadmap

## Phase 0. 项目准备

目标：确定技术栈、模型、测试方式。

交付物：

- Python 环境。
- 可加载的小模型。
- 空项目结构。
- 第一组测试命令。

验收：

- 能用 Transformers 直接生成一段文本。

## Phase 1. 单请求最小闭环

目标：实现 `LLM.generate` 的单 prompt 版本。

交付物：

- `SamplingParams`
- `LLM`
- `ModelRunner`
- `Sampler`
- 单请求 experiment
- 基础测试

验收：

- 输入一个 prompt，生成 `max_tokens` 内的文本。
- greedy 输出稳定。
- EOS 能正确停止。

## Phase 2. 离线 batch generation

目标：多个 prompt 一次性进入引擎。

交付物：

- `Sequence`
- `SequenceStatus`
- batch padding 或 ragged batch 处理策略
- 多请求输出整理
- batch benchmark

验收：

- 不同长度 prompt 能同时生成。
- 每条请求有独立 finish_reason。

## Phase 3. KV cache 显式化

目标：自己控制 prefill 和 decode，不依赖高层 `generate`。

交付物：

- prefill path
- decode path
- cache state
- cache correctness tests

验收：

- decode 每步只喂最新 token。
- 输出和无 cache 版本在 greedy 下基本一致。

## Phase 4. Scheduler

目标：引入请求生命周期和 continuous batching 雏形。

交付物：

- `Scheduler`
- waiting/running/finished queues
- token budget 或 max_num_seqs 限制
- step-based engine loop

验收：

- 已在 running 的请求可以继续 decode。
- 新请求可以在后续 step 加入。
- 结束请求能释放资源。

## Phase 5. Block Manager and Paged KV Cache

目标：模拟 vLLM 的 paged KV cache 核心思想。

交付物：

- `Block`
- `BlockManager`
- `block_table`
- slot mapping
- block allocation/free tests

验收：

- 每条 Sequence 的 token 位置能映射到 block slot。
- Sequence 结束时 block 被释放。
- 内存不足时 Scheduler 能延后请求。

## Phase 6. 性能和工程增强

目标：开始接近真实推理框架。

候选特性：

- prefix caching
- chunked prefill
- streaming output
- async request interface
- OpenAI-compatible HTTP server
- torch.compile
- CUDA graph
- Triton attention kernel

验收：

- 每加一个特性，都有 benchmark 和回退路径。

## Phase 7. 进阶主流特性

候选方向：

- tensor parallel
- quantization: INT8/INT4/AWQ/GPTQ
- speculative decoding
- LoRA adapter serving
- structured output
- function calling
- multimodal placeholder

原则：

- 一次只引入一个复杂变量。
- 每个特性先写最小设计文档，再写代码。

