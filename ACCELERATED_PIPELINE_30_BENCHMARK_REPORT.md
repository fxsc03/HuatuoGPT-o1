# 30 条加速版 PRM Pipeline 基准报告

## 1. 基准配置

- 日期：2026-04-13
- Python 环境：`/home/fxs/LLM1.30/HuatuoGPT-o1/.venv312`
- GPU：`NVIDIA RTX A6000`
- 输入文件：`/home/fxs/LLM1.30/HuatuoGPT-o1/data/medical_o1_verifiable_problem_30.json`
- 主流程入口：`/home/fxs/LLM1.30/HuatuoGPT-o1/search_for_complex_reasoning_path.py`
- 核心实现：`/home/fxs/LLM1.30/HuatuoGPT-o1/cot_pipeline_accelerated.py`

本次只跑主流程，只产出 `verified Long_CoT`，不包含 `Complex_CoT` 改写和最终 `Response` 生成。

## 2. 使用的模型

- 生成思维链：`Qwen/QwQ-32B`（硅基流动远程）
- Lessons Summary：`Qwen/QwQ-32B`（硅基流动远程）
- Chunk / Final Judge：`models/Qwen2.5-7B-Instruct`（本地 A6000）

补充说明：

- 运行中发现硅基流动上的 `Qwen/QwQ-32B` 不支持 JSON mode。
- 当前代码已补成“优先请求 JSON mode，若模型不支持则自动回退到普通输出 + 本地解析”。

## 3. 实际运行命令

```bash
./.venv312/bin/python search_for_complex_reasoning_path.py \
  --data_path data/medical_o1_verifiable_problem_30.json \
  --api_key "$SILICONFLOW_API_KEY" \
  --generator_model_name Qwen/QwQ-32B \
  --summary_model_name Qwen/QwQ-32B \
  --judge_backend local \
  --local_judge_model_path models/Qwen2.5-7B-Instruct \
  --local_judge_max_gpu_memory_gib 20 \
  --generator_backend online \
  --num_process 16 \
  --max_reasoning_chain_attempts 3 \
  --max_chunk_retries 3 \
  --max_chunk_judge_attempts 3 \
  --compression_trigger_fail_depth 2 \
  --use_json_mode \
  --mode_suffix benchmark30_accel_localjudge_online_v2
```

## 4. 输出位置

- 单样本目录：`/home/fxs/LLM1.30/HuatuoGPT-o1/output_data/medical_o1_verifiable_problem_30_benchmark30_accel_localjudge_online_v2`
- 聚合成功文件：`/home/fxs/LLM1.30/HuatuoGPT-o1/medical_o1_verifiable_problem_30_benchmark30_accel_localjudge_online_v2_12.json`

## 5. 30 条实测结果

### 5.1 总体结果

- 总输入：30
- 成功：12
- Hard reject：18
- 成功率：40.00%
- 总耗时：947.88 秒
- 总耗时：15 分 47.88 秒
- 平均耗时：31.60 秒 / 输入样本
- 吞吐：约 114.0 样本 / 小时

### 5.2 请求与重试统计

- Generator 请求数：128
- Local Judge 请求数：166
- Summary 请求数：3
- 平均 Generator 请求数：4.27 / 样本
- 平均 Judge 请求数：5.53 / 样本
- 平均 Summary 请求数：0.10 / 样本
- 平均 Chunk attempts：4.27 / 样本
- 触发 lessons summary 的样本数：2

### 5.3 Chunk 结果分布

- Chunk 1 accepted：36
- Chunk 1 parse_failed：14
- Chunk 1 small_reject_hard：5
- Chunk 2 accepted：32
- Chunk 2 parse_failed：5
- Chunk 2 small_reject_hard：4
- Chunk 3 accepted：25
- Chunk 3 small_reject_hard：7
- Final-chain verify 次数：25

## 6. Token 统计

### 6.1 远程模型 token

- Generator prompt tokens：99,317
- Generator completion tokens：116,898
- Summary prompt tokens：7,347
- Summary completion tokens：2,104

合计远程 token：

- Remote prompt tokens：106,664
- Remote completion tokens：119,002
- Remote total tokens：225,666

折算到单个输入样本：

- 平均 remote prompt：3,555.47 tokens / 样本
- 平均 remote completion：3,966.73 tokens / 样本
- 平均 remote total：7,522.20 tokens / 样本

### 6.2 本地 Judge token

- Judge prompt tokens：148,729
- Judge completion tokens：12,993
- Judge total tokens：161,722

这部分不产生硅基流动 API 费用，但会消耗本地 GPU 时间。

## 7. 费用估算

费用基于硅基流动官方价格页的 `Qwen/QwQ-32B`：

- 在线价格：输入 `¥1 / 1M tokens`，输出 `¥4 / 1M tokens`
- Batch 价格：输入 `¥0.5 / 1M tokens`，输出 `¥2 / 1M tokens`

参考：

- https://siliconflow.cn/pricing

### 7.1 本次 30 条实测费用

只计算远程 generator + summary，Judge 为本地推理，不计 API 费用。

- 全在线：约 `¥0.5827`
- 如果 future run 中仅把 generator 改为 Batch，而 summary 仍保持在线：约 `¥0.2992`

### 7.2 外推到 40,000 条输入

按本次 30 条的真实 token 均值直接线性外推：

- 当前配置预计耗时：约 `351.07 小时`，即 `14.63 天`
- 全在线预计费用：约 `¥776.90`
- Generator 走 Batch、Summary 仍在线：约 `¥398.96`

### 7.3 外推到全量 40,644 条输入

当前仓库中的全量数据文件 `data/medical_o1_verifiable_problem.json` 共 `40,644` 条。

- 当前配置预计耗时：约 `356.72 小时`，即 `14.86 天`
- 全在线预计费用：约 `¥789.40`
- Generator 走 Batch、Summary 仍在线：约 `¥405.38`

### 7.4 如果按当前成功率看有效产出

当前 30 条跑出的成功率是 `40%`。如果全量 40,644 条仍保持同一成功率：

- 预计成功样本数：约 `16,258`
- 全在线单个成功样本平均 API 成本：约 `¥0.0486`
- Generator Batch 方案下单个成功样本平均 API 成本：约 `¥0.0249`

## 8. 速度瓶颈判断

这轮基准里，主瓶颈已经不是远程 API，而是本地 Judge：

- 运行期间 A6000 GPU 利用率接近满载
- Local Judge 总调用数达到 `166`
- 按总时长粗算，平均约 `5.71 秒 / Judge 调用`
- Judge 目前是单模型、单实例、串行推理

这意味着：

1. 把 generator 从 online 改成 Batch，最明显的收益是降费用。
2. 在单张 A6000、当前 Judge 实现不变的前提下，Batch 对总时长的改善不会特别大。
3. 要把 40,644 条压到 1 到 2 天，必须继续压 Judge 吞吐，而不是只优化远程生成。

## 9. 结论

这版重构已经完成了以下目标：

- 主流程只产出 `verified Long_CoT`
- `Complex_CoT` 改写和最终 `Response` 已拆到独立脚本
- 小模型 Judge 已改成本地运行
- Generator 已保留 `online` / `batch` 双后端
- JSON mode 已接入，并对 `QwQ-32B` 增加了自动兼容回退

但从真实 30 条基准看，当前单卡版本的时间结论很明确：

- 现在跑完整个 `40,644` 条，大约要 `14.86 天`
- 在线费用大约 `¥789`
- 若 generator 改走 Batch，费用可降到约 `¥405`
- 真正卡速度的是本地 Judge，而不是远程 generator

## 10. 下一步建议

如果目标是把全量时间压到 1 到 2 天，下一步应优先做这些事：

1. 把本地 Judge 改成真正的批量推理，而不是当前串行 `generate`
2. 考虑用 vLLM / TGI / SGLang 起本地 Judge 服务，做并发和动态 batching
3. 多机或多进程切分输入数据，并行跑多个主流程分片
4. 让 generator 尽量走 Batch API，把远程成本先降下来
5. 若资源允许，为 Judge 单独准备第二张卡，避免生成和验证争抢一张 GPU
