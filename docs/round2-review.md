# FastSD 第二轮审阅落实

依据 `第二轮.md`，基于 `update@b025845` 更新公共 `src/`、`edge/`、`cloud/`。历史实验 runtime 快照不随公共代码改变；本轮提供独立冻结入口。

## 缓存和模型支持范围

- Target 启动时检查实际 attention 层、`layer_types` 和 `use_sliding_window`。启用滑窗层时明确拒绝，两个 KV 后端都执行此检查；单独存在 `config.sliding_window` 数值不会触发拒绝。不修改 checkpoint 配置。
- 完整因果注意力保留全部已接受前缀的 KV。现有概率窗口继续使用绝对位置；KV rollback 只删除拒绝候选的后缀。`FASTSD_KV_BLOCK_TOKENS` 仍然只是扩容粒度。
- `CacheSnapshot` / `KVTransferTicket` 保存会话 epoch、内容版本、长度、源身份、源 lease、目标 cache、producer/copy/consumer 事件、接管者和预算。
- `submit_h2d` / `submit_d2h` 只准备和提交复制；只有执行器的 `bind_to_consumer` 为该消费者建立事件依赖。分块未全部提交时没有最终 `copy_done`，接管 pending ticket 后继续完成原传输。
- 后台只发布候选表，正式缓存由执行器安装。reserve/activate 更新所有权，不更新内容版本。会话回收后再次使用相同 PID 会获得新的 epoch。
- 每个执行阶段均进入明确的 compute stream。实际写入者通过 `mark_computed(..., producer_stream=...)` 记录事件，后台不猜测生产者流。
- 缓存登记锁、manager 元数据锁内不进行分配、复制或同步。源和目标由 ticket/lease 持有；取消后等待已经提交的 DMA 和最后消费者事件，再释放存储。覆盖源有效区前先终止未提交的读取并等待该源已提交的读取。

## 预取模式和预算

| 参数 | 默认值 | 行为 |
| --- | --- | --- |
| `--kv_prefetch_mode` | `protect_forward` | 已准入计划或 GPU 计算事件未完成时停止预测性 GPU 提交；已提交小块在下批 forward 前排空，延迟计入启动时间。 |
| `--kv_prefetch_mode overlap` | 显式开启 | 已准备好的候选可在计算期间分块复制。大块 GPU/pinned 分配仍只在空闲准备阶段进行。 |
| `--kv_prefetch_mode cpu_prepare` | 显式开启 | 后台只维护预测和快照；GPU 加载全部走实际消费者的按需路径。 |
| `--kv_prefetch_mb` | 64 MiB | 所有未释放预测性目标 buffer 的总预算，包括未完成和 ready 候选。 |
| `--kv_prefetch_chunk_kb` | 4096 KiB | 每次预测性提交的最大块；跨所有 ticket 最多一个预测性小块在途。 |
| `--kv_headroom_mb` | 512 MiB | 检查当前可用显存后保留的执行余量；无法查询或不足时跳过预取。 |
| `--kv_overlap_max_slowdown` | 0.1 | 相同 forward 形状已有无预测性 DMA 的基线后，连续三次超过扰动预算则退回保护模式。 |

预测集合变化只回收失效的候选副本，不卸载正常会话。正式卸载仍由执行边界的缓存策略决定。H2D、D2H、预测 H2D 使用独立 stream；消费者接管后用按需队列继续自己的剩余块。

这里的显存限制是保守的预测性缓存预算和显存余量检查，不是覆盖任意模型、第三方内核 workspace 和其他进程的硬峰值显存准入证明。`overlap` 的回退依据是在线测量，无法保证首次重叠或未知形状没有带宽竞争。保护模式不承诺下一批零启动延迟；`cpu_prepare` 可以避免预测性 GPU DMA 的这部分代价。

## 接受率、γ、ahead 和调度

- 接受率继续保留 128 个逻辑轮，按位置维护条件风险集和成功数，退出窗口时增量扣除。混合长度验收样本得到 `S1=0.8`、`S2=0.64`、`E2=2.44`。
- 残余 Verify 的收益以已接受位置为条件；非最终片段全接受时不计 bonus，发生拒绝时仍计 correction。缓存生存曲线和切片收益，候选评分不反复扫描 128 轮历史。
- 所有服务模式通过同一个 `GammaController.finish` 在逻辑轮结束时更新一次。内部切片不更新 γ；CUDA 服务使用完整逻辑轮累计 forward GPU 时间。
- 新协议携带 `fresh_decode_tokens` 和 `draft_timing_valid`。全复用没有有效物理 decode 观测，服务器沿用上一次 EMA，不再退回 `lag/gamma`。旧客户端未携带有效性字段时保留兼容路径。第一轮 Prefill 与后续 decode 分开计时。
- ahead 使用 enabled/cooldown/probe。没有实际计算不写收益观测；probe 至少需要 bridge 加一个候选。只有下一轮实际消费了候选才记 reused，终止时未消费的候选计零收益。
- 候选评分、已选 batch 和实际观测共用 `cost_features()`，统一 cached context、bridge 和 forward token 数。
- 预测和提交共用纯函数 `transition_work_item()`，统一部分 Verify、Prefill 剩余长度分类、cycle aging 和完成状态。lookahead 使用投影对象的特征，不读取原对象的旧 cursor。

## 指标解释

| 指标 | 口径 |
| --- | --- |
| `forward_gpu_ms` | 本批数据依赖已满足后，模型 forward 起止 CUDA event 的区间；逻辑请求跨片段时累计。 |
| `batch_start_delay_ms` | 计划就绪到实际启动 forward 的墙钟延迟，包含保护模式排空在途小块和本批准备过程。 |
| `exposed_h2d_stall_ms` | 本消费者的主机准备/分配/提交时间，加上其计算流等待自身复制事件的暴露时间。 |
| `prefetch_ready_hit` | 计划接管时匹配 ticket 已完整就绪。 |
| `prefetch_pending_hit` | 接管匹配的在途或待继续提交 ticket，复用同一次 H2D。 |
| `prefetch_demand_miss` | 没有匹配候选，消费者发起 GPU 加载。 |
| `prefetch_wasted_bytes` | 未被使用而失效或取消的候选已提交字节数。 |
| `prefetch_inflight_bytes` | 尚未就绪候选的目标 buffer 预算。`prefetch_dma_inflight_bytes` 单独表示实际未完成小块字节数。 |
| `request_e2e_ms` | Edge 既有完整请求时延，包含 Prefill、草拟、网络、队列、传输和验证。 |
| `plan_ms_*` / `lookahead_ms_*` | 主控构建真实计划/预测计划的总时间、次数和最大值。 |

批次指标会出现在同批每个请求的响应中，适合按请求解释，不能跨请求求和当作 GPU 总忙时。启动延迟包含数据等待，因此它和 H2D stall 也不能直接相加。Edge 的新分项累计覆盖 Verify 轮，Prefill 分项另随 Prefill 响应返回。

## 新实验包

`scripts/round2_experiment.py prepare <新目录>` 冻结当前源码、逐文件 SHA256、Git SHA、dirty 状态和对照矩阵。可用 `--common-argv-json` 提供所有变体共享的模型、数据和设备参数。

冻结后运行包内的脚本：

```bash
python <包目录>/runtime/scripts/round2_experiment.py run <包目录> protect_forward cloud -- <服务参数>
python <包目录>/runtime/scripts/round2_experiment.py run <包目录> protect_forward edge -- <端侧参数>
```

运行前验证冻结文件；记录实际导入路径、解析后的参数、Python/PyTorch/Transformers 版本及关键环境配置。固定 γ、同后端、同 FCFS 的前四组为 off / protect_forward / overlap / cpu_prepare；随后分别开启 ahead、收益 γ、progress 排序。每次运行使用独立结果目录。

`scripts/benchmark_round2_runtime.py` 提供 32/64 候选开销探测，以及可选的微型 Llama GPU 时间线，显式检查输出一致和保护模式无预测性 DMA 重叠。该程序不启动正式模型的长时间压测。

最终测试、微型 GPU 时间线与 planner 数值见本轮修改记录和保存的验证日志。需要外部 Llama checkpoint 的三项测试保持显式跳过；微型随机模型的 CPU/GPU 正确性和真实异步复制测试会实际执行。
