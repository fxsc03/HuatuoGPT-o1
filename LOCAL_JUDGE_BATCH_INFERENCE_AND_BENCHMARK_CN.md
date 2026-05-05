# 本地 Judge Batch 推理实现与 20 条基准说明

## 1. 本次改动目标

这次改动只针对本地 `judge` 的吞吐问题。

旧实现的问题是：

- generator 虽然可以并发，但本地 judge 仍然是单条 `transformers.generate()`
- `chunk judge` 和 `final-chain verify` 都是逐条串行
- A6000 上 GPU 经常被单条 judge 长时间占用，导致整条流水线收尾很慢

现在的改动是：

- 把本地 judge 改成真正的 batch 推理
- 把 `chunk` 阶段拆成“先生成，再批量 judge”
- 把 `final-chain verify` 也改成批量 judge

## 2. 代码改动位置

- 主实现文件：[cot_pipeline_accelerated.py](/home/fxs/LLM1.30/HuatuoGPT-o1/cot_pipeline_accelerated.py)
- 兼容入口：[search_for_complex_reasoning_path.py](/home/fxs/LLM1.30/HuatuoGPT-o1/search_for_complex_reasoning_path.py)

核心新增点：

- `LocalJudgeClient.call_messages_batch()`
- `generate_chunk_online_without_judge()`
- `collect_chunk_candidates_online()`
- `collect_chunk_candidates_batch()`
- `run_local_chunk_judge_batches()`
- `verify_final_chains_local_batch()`

## 3. 当前批量 judge 的工作方式

### 3.1 Chunk 阶段

现在每个 chunk stage 的执行顺序变成：

1. generator 先并发生成候选 chunk
2. parse 成功的候选 chunk 暂存起来
3. 把这些候选 chunk 的 judge prompt 拼成一个 batch
4. 本地 judge 一次推理一个 batch
5. 批量写回 `accepted / small_reject_hard`

这样避免了“每个样本生成完就立刻单条进本地 judge”的串行瓶颈。

### 3.2 Final-chain verify

当多个样本同时完成 3 个 chunk 后：

1. 为每条完整 reasoning chain 构造 final verify prompt
2. 把这些 prompt 合并成 batch
3. 本地 judge 批量返回终审结果

### 3.3 新增参数

新增参数：

```bash
--local_judge_batch_size 8
```

默认值是 `8`。当前这版在你的 A6000 上先按 `8` 跑。

## 4. 当前实现的边界

本次改动只解决“本地 judge 串行”的问题，没有改这些逻辑：

- generator 仍然是远程在线生成
- 最终终审仍然依赖当前 judge prompt 的答案一致性判断
- `max_chunk_judge_attempts=3` 的逻辑仍然保留

也就是说：

- 现在本地 judge 吞吐更高了
- 但整条流水线的总时间，仍然会受到远程 generator 和长尾重试样本影响

## 5. 20 条实测命令

```bash
./.venv312/bin/python search_for_complex_reasoning_path.py \
  --data_path data/medical_o1_verifiable_problem_30.json \
  --limit_num 20 \
  --api_key "$SILICONFLOW_API_KEY" \
  --generator_model_name Qwen/QwQ-32B \
  --summary_model_name Qwen/QwQ-32B \
  --judge_backend local \
  --local_judge_model_path models/Qwen2.5-7B-Instruct \
  --local_judge_max_gpu_memory_gib 20 \
  --local_judge_batch_size 8 \
  --generator_backend online \
  --num_process 16 \
  --max_reasoning_chain_attempts 3 \
  --max_chunk_retries 3 \
  --max_chunk_judge_attempts 3 \
  --compression_trigger_fail_depth 2 \
  --use_json_mode \
  --mode_suffix benchmark20_localjudge_batch_v1
```

## 6. 20 条实测结果

输出文件：

- 单样本目录：[output_data/medical_o1_verifiable_problem_30_benchmark20_localjudge_batch_v1](/home/fxs/LLM1.30/HuatuoGPT-o1/output_data/medical_o1_verifiable_problem_30_benchmark20_localjudge_batch_v1)
- 聚合成功文件：[medical_o1_verifiable_problem_30_benchmark20_localjudge_batch_v1_6.json](/home/fxs/LLM1.30/HuatuoGPT-o1/medical_o1_verifiable_problem_30_benchmark20_localjudge_batch_v1_6.json)

总体结果：

- 输入：20
- 成功：6
- Hard reject：14
- 成功率：30%
- 总耗时：`709.42 秒`
- 总耗时：`11 分 49 秒`
- 平均耗时：`35.47 秒 / 输入样本`
- 吞吐：约 `101.5 输入样本 / 小时`

Judge 批量化已生效：

- `('chunk', 'local-batch')`: 80 次
- `('final-chain', 'local-batch')`: 10 次

### 6.1 Summary 与多轮整链统计

按这 20 条过程文件逐条统计：

- 触发过 `summary` 生成的样本数：`1 / 20`
- 真正带着 `summary` 进入后续整链重试的样本数：`1 / 20`
- 走过多轮整链推理的样本数：`2 / 20`

对应样本：

- `summary` 生成并实际使用：`process_id = 17`
- 多轮整链推理：`process_id = 13, 17`

更细一点看：

- `process_id = 13`：一共生成了 `2` 轮完整 reasoning chain，但没有触发 `summary`
- `process_id = 17`：一共生成了 `3` 轮完整 reasoning chain，期间生成了 `2` 次 lessons summary，并且确实把 summary 喂回了后续重试

这说明在这 20 条 benchmark 里：

- 大多数样本要么在 chunk 阶段被提前筛掉
- 要么首轮整链就结束
- 真正进入“整链失败 -> 总结压缩 -> 再重试”的样本比例并不高

### 6.2 失败样本是怎么失败的

这 20 条里失败样本一共 `14` 条，按终止位置分布如下：

- 卡在 `chunk 1`：`7` 条
- 卡在 `chunk 2`：`1` 条
- 卡在 `chunk 3`：`5` 条
- 卡在整链终审重试耗尽：`1` 条

对应样本编号：

- `chunk 1 hard reject`：`10, 19, 20, 5, 6, 7, 9`
- `chunk 2 hard reject`：`8`
- `chunk 3 hard reject`：`12, 14, 16, 3, 4`
- `final-chain retry exhausted`：`17`

#### 1. 卡在 Chunk 1 的 7 条

这组样本的共同特点是：题目本身更像公共卫生、统计、定义判断或非典型临床问答，生成器虽然能抽出表面线索，但 Judge 认为它没有真正完成 `Chunk 1` 的局部目标。

典型失败模式：

- 题目并不适合“症状提取与病理映射”模板，导致 chunk scope 不匹配
- 生成器在 `Chunk 1` 就提前往计算、下一步诊断、任务总结方向滑
- 个别样本出现 judge 返回解析失败

代表性例子：

- `process_id = 10`：生成器把题目泛化成国家计划生育项目背景，而不是贴着题目本身提取关键信息
- `process_id = 20`：在 `Chunk 1` 提前给出下一步诊断建议，越界
- `process_id = 5`：Judge 输出解析失败
- `process_id = 6, 7`：题目本质是计算题，Judge 认为 `Chunk 1` 的“病理映射”模板不贴题

#### 2. 卡在 Chunk 2 的 1 条

只有 `process_id = 8` 卡在 `Chunk 2`。

失败原因很明确：

- 题目要求先做鉴别诊断与排错
- 但生成器在 `Chunk 2` 提前开始给“初始诊断检查建议”
- Judge 认为这属于未来步骤内容，直接判为越界

#### 3. 卡在 Chunk 3 的 5 条

这组样本说明 `Chunk 3` 仍然是当前最敏感的失败点之一，因为这一段既要落最终结论，又要严格对齐参考答案和前两段已验证历史。

典型失败模式：

- 最终答案过宽，不够具体
- 引入前文未验证的新事实
- 最终结论没有严格贴合参考答案
- 把“结论”写成了治疗建议或其他任务类型输出

代表性例子：

- `process_id = 12`：答案写成 `Shigella or Campylobacter`，但参考答案要求更具体的 `Campylobacter`
- `process_id = 14`：在 `Chunk 3` 引入了前文未确认的 `ophthalmoplegia`
- `process_id = 16`：结论段没有给最终结论，反而滑向具体治疗方案
- `process_id = 3`：在终段引入了前文未充分验证的新诊断跳步
- `process_id = 4`：终段没有对齐参考答案里的 `Hydatid Cyst`

#### 4. 卡在 Final-chain 重试耗尽的 1 条

只有 `process_id = 17` 属于这种情况。

它的失败不是 chunk 质量差，而是整链终审时答案字面没有完全贴齐参考答案：

- 生成答案：`Burkholderia pseudomallei`
- 参考答案：`B. pseudomallei`

Judge 把这个视为答案不完全一致，于是它连续多轮整链重试，最终耗尽 `max_reasoning_chain_attempts`。

这说明当前 20 条 benchmark 里的长尾失败，至少有一部分不是医学推理错误，而是：

- 终审答案归一化不够宽松
- 对缩写 / 全称的字面一致性要求过严

#### 5. 一句话总结

这 14 条失败样本里，主要失败来源不是 parse，而是 `scope / answer alignment`：

- 前半段更多是“题型和 Chunk 模板不匹配”或“过早越界”
- 后半段更多是“最终答案不够具体”或“与参考答案字面未完全对齐”

远程 token 统计：

- generator total tokens：`87,043`
- summary total tokens：`6,016`

本地 judge token 统计：

- judge total tokens：`85,334`

## 7. 对 2000 条的时间估算

这里给两个口径。

### 7.1 如果你说的是“处理 2000 条输入”

按本次 20 条实测线性外推：

- `35.47 秒 / 条`
- `2000 条输入` 约 `70,942 秒`
- 也就是约 `19.7 小时`
- 大约 `0.82 天`

### 7.2 如果你说的是“最终拿到 2000 条成功样本”

按这次成功率 `30%` 粗算：

- 需要先跑约 `6667` 条输入
- 总时间约 `236,500 秒`
- 也就是约 `65.7 小时`
- 大约 `2.74 天`

## 8. 结果怎么理解

这轮 20 条 benchmark 说明两件事：

1. 本地 judge 的 batch 路径已经打通，`chunk judge` 和 `final-chain verify` 都在走 `local-batch`
2. 端到端总时间并没有出现数量级下降，说明现在的长尾时间不只来自 judge，也来自：
   - 远程 generator
   - 多轮 reasoning chain 重试
   - 终审里对最终答案字面一致性的严格要求

这次长尾样本里还观察到一个典型例子：

- `Burkholderia pseudomallei`
- `B. pseudomallei`

这类语义一致但字面不完全一致的答案，仍可能在终审中被拒掉，从而拖长总时间。

## 9. 当前结论

如果你现在就继续跑大规模数据，比较稳妥的时间判断是：

- `2000 条输入`：约 `20 小时`
- `2000 条成功样本`：约 `2.7 天`

如果后面还想继续压时间，优先级建议是：

1. 调整 final verify 的答案归一化，减少“语义正确但字面不一致”的长尾重试
2. 让 generator 尽量走 Batch API
3. 再考虑把本地 judge 从当前 `transformers` 批推理，继续升级到 vLLM / SGLang 服务化批推理

## 10. 新增模板筛选后的 20 条验证

本轮又加了一道前置筛选：

- 在真正生成 CoT 之前
- 先让 `Qwen/QwQ-32B` 判断当前题目是否适合这套三段式模板
- 不适合就直接标记为 `template_skip`

### 10.1 验证命令

```bash
./.venv312/bin/python search_for_complex_reasoning_path.py \
  --data_path data/medical_o1_verifiable_problem_30.json \
  --limit_num 20 \
  --api_key "$SILICONFLOW_API_KEY" \
  --generator_model_name Qwen/QwQ-32B \
  --summary_model_name Qwen/QwQ-32B \
  --judge_backend local \
  --local_judge_model_path models/Qwen2.5-7B-Instruct \
  --local_judge_max_gpu_memory_gib 20 \
  --local_judge_batch_size 8 \
  --generator_backend online \
  --num_process 16 \
  --max_reasoning_chain_attempts 3 \
  --max_chunk_retries 3 \
  --max_chunk_judge_attempts 3 \
  --compression_trigger_fail_depth 2 \
  --use_json_mode \
  --mode_suffix benchmark20_templategate_localjudge_batch_v1
```

输出目录：

- 单样本目录：[output_data/medical_o1_verifiable_problem_30_benchmark20_templategate_localjudge_batch_v1](/home/fxs/LLM1.30/HuatuoGPT-o1/output_data/medical_o1_verifiable_problem_30_benchmark20_templategate_localjudge_batch_v1)
- 聚合成功文件：[medical_o1_verifiable_problem_30_benchmark20_templategate_localjudge_batch_v1_5.json](/home/fxs/LLM1.30/HuatuoGPT-o1/medical_o1_verifiable_problem_30_benchmark20_templategate_localjudge_batch_v1_5.json)

### 10.2 结果统计

- 输入：`20`
- 成功：`5`
- `template_skip`：`7`
- `hard_reject`：`8`
- 总耗时：`1118.48 秒`
- 总耗时：约 `18 分 38 秒`

远程 token：

- `template_gate total tokens`：`16,599`
- `generator total tokens`：`132,044`
- `summary total tokens`：`9,580`

本地 judge token：

- `judge total tokens`：`90,678`

### 10.3 被模板筛掉的题

`template_skip` 的样本编号：

- `10, 11, 19, 2, 5, 6, 7`

这些题主要落在：

- 公共卫生 / 计划生育项目题
- 直接事实回忆题
- 统计 / 计算题
- 术语定义题

### 10.4 和“不加模板筛选”那轮怎么比

不加模板筛选时：

- 成功：`6 / 20`
- `hard_reject`：`14 / 20`
- 总耗时：`709.42 秒`

加模板筛选后：

- 成功：`5 / 20`
- `template_skip`：`7 / 20`
- `hard_reject`：`8 / 20`
- 总耗时：`1118.48 秒`

如果只看“进入主 CoT 流程”的题：

- 被筛后真正进入主流程的题：`13`
- 这 13 条里成功：`5`
- 筛后主流程成功率：`38.46%`

对比不加筛选那轮的整体成功率：

- 不加筛选：`30%`
- 加筛选后，主流程内部成功率：`38.46%`

### 10.5 结论

这道模板筛选的作用是：

- 先把明显不适合三段式临床 CoT 的题拦掉
- 减少后续在 `chunk 1/2/3` 上被无意义硬拒的数量

但这轮 20 条验证也说明：

1. 它确实提升了“进入主流程后”的成功率
2. 它会额外增加一轮远程调用，所以总墙钟时间变长了
3. 如果你的最终目标是“只保留真正适合临床三段式模板的题”，它是有意义的
4. 如果你的目标是“尽量多拿成功样本，不在乎题型是否为临床 case-style”，那这道筛选需要作为可选项，而不适合强制开启
