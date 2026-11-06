# FastSD

FastSD 是一个面向云边协同的投机解码研究项目。边缘端运行较小的 Draft 模型，一次生成多个候选 token；云端运行 Target 模型，对候选序列进行验证。边缘端保留被接受的前缀，接上云端返回的 token，再开始下一轮，直到达到停止条件。

仓库同时包含云边服务、直接调用解码算法的本地评测，以及统一数据和指标的多方法对比流程。下面按当前代码的执行顺序介绍各部分如何协作。

项目支持 TinyLlama Draft 与 Llama-2-7B Target 的云边实验，采用 token budget 调度和 `varlen` KV 前向。安装依赖后可按 [Llama 运行示例](scripts/command.md)启动服务并执行评测；下面介绍代码结构和执行流程。

## 1. 代码结构与职责

| 位置 | 主要职责 |
| --- | --- |
| [`edge/edge.py`](edge/edge.py) | `EdgeRunner` 管理请求与 Draft 进程；`EdgeClient` 封装云端 HTTP 通信 |
| [`cloud/cloud_service.py`](cloud/cloud_service.py) | 接收会话、Prefill 和 Verify 请求，将任务交给 Target worker，再分发结果 |
| [`src/engine.py`](src/engine.py) | `Decoding` 基类、模型加载、本地解码算法和云端批处理执行循环 |
| [`src/fastsd_scheduler.py`](src/fastsd_scheduler.py) | 请求分类、优先级计算、token 预算分配和执行计划 |
| [`src/speculative_sampling.py`](src/speculative_sampling.py) | 稀疏 Draft 概率编码、拒绝采样和严格匹配验证 |
| [`src/kvcache.py`](src/kvcache.py) | 单模型增量推理、概率历史和 KV Cache 回滚 |
| [`src/kvcache_batching.py`](src/kvcache_batching.py) | 按会话管理云端缓存，执行 padding 批处理 |
| [`src/kvcache_varlen.py`](src/kvcache_varlen.py) | Llama/TinyLlama 的变长前向计算，保留按会话分离的 DynamicCache，避免 padding token 进入模型 |
| [`src/cache_offload_policy.py`](src/cache_offload_policy.py) | 决定 Target KV Cache 是否从 GPU 卸载到 CPU |
| [`src/util.py`](src/util.py)、[`src/runtime.py`](src/runtime.py) | 参数、模型路径、随机种子、采样工具、设备与精度配置 |
| `benchmark/`、`comparison/` | 数据集评测、本地解码入口及 Assist、Ouroboros 等对比方法 |
| `baselines/specedge/` | 固定版本的 SpecEdge 官方子模块及数据、通信和 CPU 适配层 |
| [`scripts/eval_suite.py`](scripts/eval_suite.py)、[`configs/evaluation/`](configs/evaluation/) | 统一评测配置、请求清单、运行计划和结果分析 |
| `data/`、`tests/` | 数据集与算法、调度、缓存、协议和评测相关测试 |

## 2. 启动与初始化

### 云端

`cloud/cloud_service.py::main()` 解析公共参数，创建请求队列和响应队列，并通过 `spawn` 启动 Target worker。HTTP 服务默认监听 `127.0.0.1:8001`，地址和端口由 `CLOUD_SERVICE_HOST`、`CLOUD_SERVICE_PORT` 控制。

worker 创建 `CloudTargetWorker`，加载 tokenizer，再进入 `Decoding.run_target_process_batching()`。该循环加载 Target 模型，初始化按会话隔离的 KV Cache、任务队列、调度状态和预加载逻辑。HTTP 服务与模型执行位于不同进程，通过队列交换请求和结果。

### 边缘端

`edge/edge.py` 先解析服务地址、GPU 分配等边缘参数，再调用公共参数解析器。`EdgeRunner` 加载 tokenizer 和数据，`eval()` 根据 `num_drafts` 启动多个 Draft 进程。

每个 Draft 进程加载自己的模型副本，维护 HTTP 客户端，并处理分配给它的数据分片。Draft 可以运行在 GPU 或 CPU；设备、dtype 和 CPU 线程数由运行参数控制。云端和边缘端交换的是 token ID，因此 Draft 与 Target 必须具有兼容的 tokenizer 和词表映射。

两端采样参数分别解析：云端的 `temp/top_k/top_p` 决定 Target 概率分布，边缘端的采样设置决定 Draft 分布。

## 3. 一条请求的云边执行流程

### 3.1 读取请求并建立会话

边缘端支持 HumanEval、GSM8K、MGSM 和 MT-Bench。输入可以来自原始数据集，也可以来自统一评测生成的 `canonical.jsonl`。

每条请求包含 prompt 和任务标识。边缘进程按照数据分片和到达时间依次处理请求，向 `/session/init` 获取独立的 `session_id`，将 prompt 编码为 token，并为当前请求建立 Draft KV Cache 与随机数生成器。

### 3.2 首轮草拟与云端 Prefill

FastSD 启用延迟优先级协议时，首轮顺序为：

1. 边缘端通过 `/clock/sync` 估计两端时钟偏差，通过 `/prefill/arrival` 登记请求到达。
2. 云端仅记录到达时间，此时不向 Target worker 提交 Prefill。
3. 边缘端在本地完成 prompt Prefill 和首轮 Draft，记录本地计算耗时。
4. 边缘端通过 `/prefill` 上传 prompt 与首轮候选 token，并携带 prompt 长度等信息。
5. 云端校验请求，将其送入 Target worker，建立后续验证所需的缓存状态。

云端使用先前登记的到达时间计算调度等待时间，因此本地首轮草拟期间的等待也能被计入。

未使用该协议的路径由 `overlap_prefill_first_draft` 决定是否并行执行云端 Prefill 与本地首轮草拟；`/prefill/timing` 和 `/prefill/cancel` 分别用于相应路径的时序更新与取消。

### 3.3 生成候选并发送 Verify

每一轮以已经确认的前缀为基础，Draft 模型增量生成候选序列，`gamma` 控制本轮草拟长度。首轮已经生成的候选会直接用于验证。

边缘端向 `/verify` 发送会话标识、候选 token、前缀长度、验证方式和时序信息。拒绝采样模式还会发送候选对应的稀疏概率分布。启用 pipeline 时，协议可以只发送增量尾部，并根据云端反馈调整后续 `gamma`。

HTTP 层为请求分配 `response_key`，将会话映射为 worker 使用的 `proc_id`，把请求写入进程队列后异步等待。一个会话同时只能有一个正在等待处理的模型请求；不同会话可以同时排队。

### 3.4 云端执行并返回结果

Target worker 从队列取出请求，完成校验、分类、调度和模型前向计算，得到每个候选位置的 Target 概率。验证结束后返回：

- `accepted`：保留前缀的绝对长度，包含本轮之前已经确认的 token。
- `final_token`：拒绝位置的修正 token，或整轮接受后的额外 token。
- 云端执行耗时等信息，以及 pipeline 路径可能返回的 `suggested_gamma`。

worker 将结果写入响应队列，HTTP 进程中的分发线程根据 `response_key` 唤醒对应请求，再把结果交给边缘端。

### 3.5 合并结果并继续生成

边缘端将序列更新为：

```text
本轮输入序列的前 accepted 个 token + final_token
```

Draft KV Cache 回滚到被接受的位置。后续前向计算会补入尚未缓存的修正 token，再继续草拟。

如果启用了主动草拟，边缘端会在等待云端响应时继续生成候选。只有当前轮全部接受，且提前生成的第一个 token 与云端 `final_token` 一致时，后续候选才可以复用；拒绝采样路径同时保留对应的概率行。

边缘端在每轮结果合并后检查 EOS、最大生成长度，以及所选数据集停止策略，决定结束请求或进入下一轮。

## 4. 云端调度与批处理

FastSD 调度由 `src/fastsd_scheduler.py` 制定计划，`src/engine.py` 执行模型计算。

1. **任务分类。** Prefill 和 Verify 分别进入 short、mid、long 长度类别。阈值根据近期前缀长度的滑动窗口调整。
2. **计算优先级。** 综合本地草拟耗时、上传与下载耗时、历史接受率和等待时间选择请求；尚未满足执行条件的 Prefill 不会被选入计划。
3. **分配执行预算。** `plan_iteration()` 使用 short:mid:long 为 `6:3:1` 的轮转顺序，并结合可用请求、优先级和等待情况选择任务。`token_budget` 限制本轮处理的 token 总量，`batch_size` 限制选入的逻辑任务数。
4. **切分任务。** Prefill 可以按 chunk 执行；较长的 Verify 也可以跨多个调度轮次推进。逻辑请求由 `WorkItem` 记录进度，单次执行片段由 `ExecutionSlice` 描述。
5. **执行计划。** 同一计划内先执行 Verify 批次，再执行 Prefill 批次。成功后提交进度；没有完成的任务保留在后续计划中。
6. **完成逻辑请求。** Verify 遇到拒绝，或所有候选验证完成后，才向边缘端返回本轮结果。内部片段全部接受但逻辑轮次尚未结束时，不在片段之间插入额外 token。

`server_sched_mode=vanilla` 和 `pipeline` 使用严格先到先服务的执行路径，不经过 FastSD 的多队列计划与预加载调度。边缘端的主动草拟、pipeline 协议与云端调度模式是分别配置的。

## 5. 验证规则与概率计算

`src/speculative_sampling.py` 提供两种验证方式。

### 拒绝采样：`verify_method=rejection`

Draft 使用 `draft_top_k` 构建完整的稀疏支持集，并在该集合内归一化得到分布 `q`。边缘端保存每个候选位置的 token ID 和概率，云端使用 Target 分布 `p` 逐位置验证。

对于候选 token `x`：

- 以 `min(1, p(x) / q(x))` 的概率接受。
- 首次拒绝时，从归一化后的 `max(p - q, 0)` 分布中采样修正 token，并结束本轮。
- 全部候选接受时，从下一个位置的 Target 分布采样额外 token。

稀疏概率是实际 Draft 分布的完整支持，而不是丢失尾部概率后的近似摘要。代码会检查概率归一化、候选是否位于支持集中，以及候选数量与概率行数是否一致。这一规则用于保持配置后的 Target 采样分布。

### 严格匹配：`verify_method=greedy`

依次比较 Draft token 和对应位置 Target 概率的最大值 token，接受连续匹配的前缀，在首次不匹配处结束验证。该规则与拒绝采样的概率接受机制不同。

## 6. KV Cache 的生命周期

边缘端使用 `KVCacheModel` 保存 Draft 的 KV Cache 和概率历史。每次草拟只计算未缓存的后缀，验证返回后裁剪未接受的部分。

云端通过 `KVCacheModel_batching` 按会话维护 `_past_key_values` 与 `_prob_history`。Prefill 建立缓存，Verify 追加未缓存 token 的状态，再按验证结果回滚；不同会话的缓存相互独立。

云端前向计算有两条路径：

| 路径 | 执行方式 |
| --- | --- |
| `varlen` | 将真实 token 拼接后执行线性层，注意力按序列各自的长度和缓存计算；当前 Llama/TinyLlama 模型可走此路径 |
| `padded` | 对齐输入和缓存长度，使用 attention mask、position IDs 执行批处理，再拆回各会话 |

默认请求使用 `varlen`；模型结构不受支持时直接报错，避免无意中运行 padding 路径。只有显式指定 `--kv_batch_mode padded` 才使用旧的补齐实现。调度器把 Verify 和 Prefill 分批执行并优先处理 Verify；`token_budget` 是容量上限，不保证每轮恰好填满。

缓存驻留由 `cache_offload_policy.py` 和云端执行循环共同控制：Prefill 后可以将缓存卸载到 CPU，Verify 根据缓存是否被选为预加载对象决定是否保留在 GPU。FastSD 使用同一调度器预测下一批 Verify 会话，并由后台线程搬运缓存，使预加载与当前计算重叠。`FASTSD_DISABLE_TARGET_CACHE_OFFLOAD` 可以关闭卸载。

## 7. 本地解码与基线流程

`benchmark/` 中的入口直接复用 `Decoding`，不一定经过 HTTP 服务。

| 执行路径 | 主要流程 |
| --- | --- |
| `small` / `large` | 仅使用 Draft / Target 模型，逐 token 自回归生成 |
| `sd` | 在同一进程中执行 Draft 草拟、Target 验证和双端缓存回滚 |
| `para_sd` 及消融入口 | 使用独立进程协调 Draft 和 Target，执行相应并行解码策略 |
| `benchmark/eval_unified.py` | 读取 canonical 请求，执行本地 `small` 或 `sd`，输出统一请求指标 |
| `benchmark/eval_target_only.py` | 执行 Target-only greedy，作为 token 一致性和质量参照 |
| `baselines/specedge/integration/` | 将统一请求和指标接入 SpecEdge 的 Tree、SpecExec 与主动草拟流程；CPU 路径标记为 `specedge_cpu_adapted` |

统一云边对比中的 `standard_sd` 使用关闭主动草拟和 pipeline 的 vanilla 云边路径，与本地 `eval_unified.py --eval_mode sd` 属于不同的运行拓扑。

## 8. 数据与评测结果流转

`scripts/eval_suite.py` 组织多方法评测，流程为：

1. **解析配置。** 从配置 JSON 读取数据集、模型、生成参数、设备、解释器和主机路径；当前 Llama 模板见 [`configs/evaluation/llama2_7b_tinyllama_four_method_cpu.json`](configs/evaluation/llama2_7b_tinyllama_four_method_cpu.json)。
2. **准备请求。** `prepare` 调用 `src/evaluation.py`，将原始数据转换为 canonical 请求，固定全局编号、prompt、参考答案和到达时间，并生成 workload hash。
3. **生成运行计划。** `plan` 根据配置输出各方法的云端、边缘端命令，记录运行路径和解释器。不同方法使用同一份请求清单。
4. **执行推理。** 各方法记录请求的生成文本、token、时延和资源相关信息；配置的 warmup 请求不计入正式请求指标。
5. **归一化与比较。** `normalize` 汇总方法输出，`compare` 比较统一指标；`paired`、`parity` 和 `resources` 分别处理配对分析、token 一致性与资源记录，`analyze` 生成多数据集、多方法分析报告。

云边运行的请求级指标位于 `exp/<exp_name>/edge_metrics_proc*.jsonl`，汇总位于 `edge_metrics_summary.json`。统一评测文件位于 `exp/comparison/<run_id>/`，包含请求清单、运行 manifest、命令记录和各方法结果。

主要统计项包括 TTFT、TPOT、请求端到端时延、系统输出吞吐、接受率，以及草拟、等待和云端验证耗时。通信量统计的是应用层序列化数据，不包含 TCP/IP 或 SSH 封装；可选能耗服务记录云端 GPU 功耗与能量。

统一评测的 MT-Bench 请求只使用第一轮 prompt；`target_only` 用于参照，不计入四方法性能对比。完整评测需要先把模板中的模型路径、网络地址和 CPU 绑定改为实际环境。

Llama-2-7B 的一次单卡与两卡验证数据见 [实验记录](docs/LLAMA_NODE2_VALIDATION.md)；其中的服务器路径仅用于记录该次运行环境。
