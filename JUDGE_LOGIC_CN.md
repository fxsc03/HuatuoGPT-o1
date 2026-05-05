# Judge 审核逻辑说明

本文档对应当前实现 `[cot_pipeline_accelerated.py](/home/fxs/LLM1.30/HuatuoGPT-o1/cot_pipeline_accelerated.py)`。

当前版本把 Judge 分成两类任务：

1. `chunk judge`：审核当前中间 Chunk
2. `final-chain verify`：审核完整 reasoning chain 和最终答案

两者都可以使用同一个小模型，但判定上下文不同，目标也不同。

## 1. 当前 Judge 用什么模型

推荐配置：

- 生成模型：远程 `Qwen/QwQ-32B`
- Chunk / Final Judge：本地 `models/Qwen2.5-7B-Instruct`
- Summarization：远程 `Qwen/QwQ-32B`

Judge 本地化后，远程 API 成本主要集中在 generator 和 summarize，而不是验证环节。

## 2. Chunk Judge 审什么

Chunk Judge 审的是“这一步能不能作为当前阶段的合格中间推理”，不是“你有没有提前答出最终答案”。

Judge 会收到：

- `question`
- `chunk_id`
- `chunk_name`
- `chunk_goal`
- `chunk_requirements`
- `confirmed_chunks`
- `candidate_chunk`

## 3. Chunk 1 / 2 的关键变化

这是当前实现里最重要的修正：

- 当 `chunk_id == 1` 或 `chunk_id == 2` 时，Judge prompt 不注入 `reference_answer`
- Judge 只能判断：
  - 逻辑是否通顺
  - 是否符合医学常识
  - 是否完成了当前 Chunk 的要求
  - 是否存在幻觉、矛盾、越界、偷跑未来结论

因此，对于 `Chunk 1` / `Chunk 2`：

- 没有写最终答案，不是错误
- 没有写病毒结构，不是错误
- 只要局部推理忠实、合理、没越界，就应该允许通过

## 4. Chunk 3 才引入标准答案

只有在 `chunk_id == 3` 时，Judge prompt 才会注入 `reference_answer`。

此时 Judge 会额外判断：

- 最终结论是否和标准答案一致
- 最终结论是否由前两个已验证 Chunk 支撑
- 是否引入了原题中没有的新证据

## 5. 同一个 Chunk 如何判定失败

流程是：

1. generator 先产出一个候选 Chunk
2. 代码先做 JSON / 结构解析
3. parse 成功后，交给小模型 Judge
4. 同一个候选 Chunk 最多审核 `max_chunk_judge_attempts=3` 次
5. 如果 3 次都 `pass=false`，直接 `hard_reject`

这里的设计故意偏保守：

- 不再让 generator 反复为一个被 Judge 判错的候选内容回炉
- 直接截断错误分支，省掉无效 token 消耗

## 6. Final-chain Verify 审什么

当 3 个 Chunk 都通过局部审核后，系统会把它们拼成一条完整 reasoning chain，再做一次终审。

终审会看到：

- `question`
- `reference_answer`
- `completed_reasoning_chain`

终审关注的是全局正确性：

- 最终答案是否正确
- 整条 reasoning chain 是否自洽
- 最终答案是否被前面的已验证 Chunk 支撑

## 7. 终审失败之后会发生什么

终审失败不会立刻在当前 Chunk 上做压缩，而是把这整条 reasoning chain 记成一次失败链。

随后：

- 如果失败链数量还没达到阈值，就把最近一条失败链作为负反馈上下文，用于下一轮从 `Chunk 1` 重新生成
- 如果失败链数量达到 `compression_trigger_fail_depth`，就调用 `summarize_lessons()` 做整链级压缩

这意味着压缩发生在“整链级重试”上，而不是“单个 Chunk 被拒”上。

## 8. 为什么这能减少误判

旧问题是：

- Judge 拿着标准答案去审 Chunk 1 / 2
- 于是会错误要求生成器在症状提取阶段就提前写出最终答案

当前逻辑修正后：

- 中间 Chunk 不给标准答案
- 中间 Chunk 只审阶段正确性
- 最终答案只在 Chunk 3 和 final-chain verify 阶段对比

所以现在 Judge 的一句话原则是：

中间步骤审“这一步想得对不对”，最终终审才审“整道题答得对不对”。
