# vLLM-mini

这是一个从零实现简化版 vLLM 的练习项目，目标不是复制生产级 vLLM，而是逐步建立大模型推理引擎的核心心智模型。

项目参考 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的学习型代码组织，但代码由你自己实现。我们会按阶段推进：先做能跑通的最小推理闭环，再逐步加入 batching、KV cache、paged cache、调度、服务化和性能特性。

## 目标

- 用清晰的小模块实现一个可读、可测试、可扩展的 LLM inference engine。
- 第一版只追求正确性和结构清楚，不追求极致性能。
- 每个阶段都有验收标准，避免一上来陷入 PagedAttention、Triton 或 CUDA 细节。
- 后续可以逐步叠加主流特性：continuous batching、paged KV cache、prefix caching、streaming、OpenAI-compatible API、CUDA graph、tensor parallel、量化、speculative decoding 等。

## 目录结构

```text
vLLM-mini/
  README.md
  pyproject.toml
  LICENSE
  docs/
    00-project-brief.md
    01-requirements.md
    02-roadmap.md
    03-architecture.md
    04-task-board.md
    05-sprint-01.md
    references/
      nano-vllm.md
  examples/
    run_single_prompt.py
  src/
    vllm_mini/
  tests/
```

## 当前工作方式

你写代码，我作为项目经理和技术 reviewer 协助：

1. 我把需求拆成小任务，明确输入、输出和验收标准。
2. 你完成一个小任务后，把代码或报错发给我。
3. 我检查设计、提示下一步、帮你定位问题。
4. 每个阶段通过后，再进入下一阶段。

## 先读什么

建议按这个顺序读：

1. [项目简报](docs/00-project-brief.md)
2. [需求拆解](docs/01-requirements.md)
3. [路线图](docs/02-roadmap.md)
4. [架构草案](docs/03-architecture.md)
5. [第 1 阶段任务](docs/05-sprint-01.md)
6. [nano-vllm 参考笔记](docs/references/nano-vllm.md)

第一步先完成 Sprint 1：跑通单请求、非 paged、非 batching 的最小生成闭环。
