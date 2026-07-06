# Project Brief

## 项目定位

本项目是一个教学型 mini inference engine。它借鉴 vLLM/nano-vLLM 的分层思路，但第一目标是学习和可维护性，不是 benchmark 第一。

## 成功标准

短期成功：

- 能用 `LLM.generate(prompts, sampling_params)` 对一个小模型完成离线生成。
- 支持 prompt tokenization、prefill、decode loop、采样、停止条件和基本测试。
- 代码结构能自然扩展出 Scheduler、Sequence、ModelRunner、KVCacheManager 等模块。

中期成功：

- 支持多 prompt batch 生成。
- 支持显式 KV cache 复用。
- 支持 continuous batching 的请求调度。
- 支持 block-based KV cache 管理，为 paged attention 做准备。

长期成功：

- 支持 OpenAI-compatible server。
- 支持 streaming。
- 支持 prefix caching、chunked prefill、CUDA graph、量化或 tensor parallel 中的若干特性。
- 有可复现 benchmark，能对比 Transformers baseline、nano-vLLM 或 vLLM 的吞吐和延迟。

## 推荐技术边界

MVP 阶段：

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- safetensors
- pytest

暂缓：

- Triton kernel
- 自写 CUDA
- Tensor parallel
- CUDA graph
- OpenAI API server
- 分布式部署

## 推荐模型

优先选小模型，降低调试成本：

- `Qwen/Qwen3-0.6B`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- 或任意本地已经下载好的小型 causal LM

如果显存紧张，先用 CPU 或更小的测试模型打通结构。

