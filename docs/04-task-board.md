# Task Board

## Doing

- [ ] S1.1 选择目标小模型并确认能用 Transformers 直接生成。
- [ ] S1.2 创建最小 Python 包结构。
- [ ] S1.3 实现 `SamplingParams`。
- [ ] S1.4 实现 `Sampler` 的 greedy 模式。
- [ ] S1.5 实现单 prompt `LLM.generate`。
- [ ] S1.6 增加 smoke test 和单请求 experiment。

## Next

- [ ] S2.1 定义 `Sequence` 和 `SequenceStatus`。
- [ ] S2.2 支持多个 prompts 离线生成。
- [ ] S2.3 为 batch 生成增加测试。
- [ ] S2.4 增加 Transformers baseline benchmark。
- [ ] S3.1 拆分 prefill 和 decode。
- [ ] S3.2 显式维护 `past_key_values`。
- [ ] S3.3 对比 cache/no-cache 的 greedy 输出。
- [ ] S4.1 实现最小 Scheduler。
- [ ] S4.2 支持 waiting/running/finished 状态迁移。
- [ ] S4.3 增加 token budget 限制。
- [ ] S5.1 实现 Block 和 BlockManager。
- [ ] S5.2 实现 block allocation/free。
- [ ] S5.3 为每个 Sequence 维护 block table。
- [ ] S5.4 模拟 slot mapping。

## Later

- [ ] L1 prefix caching。
- [ ] L2 chunked prefill。
- [ ] L3 streaming output。
- [ ] L4 OpenAI-compatible server。
- [ ] L5 torch.compile。
- [ ] L6 CUDA graph。
- [ ] L7 Triton attention kernel。
- [ ] L8 tensor parallel。
- [ ] L9 quantization。
- [ ] L10 speculative decoding。

## Done

- [x] 建立项目规划目录。
- [x] 完成 MVP 需求拆解。
- [x] 完成阶段路线图。
- [x] 完成第一版架构草案。

