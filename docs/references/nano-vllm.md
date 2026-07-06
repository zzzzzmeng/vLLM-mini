# nano-vllm Reference Notes

参考仓库：[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)

检查日期：2026-07-06

主分支参考提交：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`

## README 中值得借鉴的点

- 定位是 lightweight vLLM implementation built from scratch。
- 代码规模约 1200 行 Python，适合作为学习参考。
- 对外 API 接近 vLLM：`LLM` + `SamplingParams` + `generate`。
- 当前强调 offline inference。
- 优化特性包括 prefix caching、tensor parallel、torch compilation、CUDA graph 等。

## 顶层文件

```text
example.py
bench.py
pyproject.toml
README.md
LICENSE
nanovllm/
```

## 核心模块形状

```text
nanovllm/
  config.py
  llm.py
  sampling_params.py
  engine/
    block_manager.py
    llm_engine.py
    model_runner.py
    scheduler.py
    sequence.py
  layers/
    attention.py
    linear.py
    rotary_embedding.py
    sampler.py
    layernorm.py
    activation.py
    embed_head.py
  models/
    qwen3.py
  utils/
    context.py
    loader.py
```

## 对本项目的启发

我们先借鉴模块边界，不借鉴实现顺序。

推荐顺序：

1. `LLM` 和 `SamplingParams`
2. `Sampler`
3. `ModelRunner`
4. `Sequence`
5. `Scheduler`
6. `BlockManager`
7. `Paged KV Cache`
8. 性能特性

`nano-vllm` 已经包含较多优化特性。我们从零实现时不要第一天就追这些优化，否则会很容易在 attention kernel、block table 和模型权重加载里同时卡住。

