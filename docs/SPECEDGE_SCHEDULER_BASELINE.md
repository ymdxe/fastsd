# SpecEdge-Sched 云端调度基线

`--server_sched_mode specedge` 在 FastSD 的共享执行后端复现 SpecEdge 服务端的 dynamic FIFO 组批。实验标识为 `specedge_sched`，图例为 **SpecEdge-Sched**。本次实现基于 FastSD `f1ecb87cff7f26a8d5aeade616656c0fb2a9348c`。组批条件与阶段顺序来自官方 `InferenceController.loop`、`_check_batch_condition` 和 `_inference`。[SpecEdge 固定版本源码](https://github.com/kaist-ina/specedge/blob/1edcaf02ffc41a7b57726450c5357ed216a3b9bc/src/strategy/server_verify/specexec/grpc.py)

## 调度与共享协议

接收线程将 HTTP 进程队列中的消息发布到 worker 的 FIFO。worker 空闲时等待首条合法模型请求，再通过 `get_nowait()` 持续接收，达到 B 或观察到空队列即封批。已有工作的批次收集等待为零。单个控制/执行线程按完整 RPC 执行，模型忙时接收线程仍可发布新任务。B 是请求数上限，`effective_token_budget` 为 `null`。合法长 prompt 可以独占一次完整 Prefill。

FastSD 将 Prefill 和 Verify 拆成独立 RPC。先按 FIFO 确定本批成员，再按原相对顺序逐个执行 Prefill，最后一次调用共享验证后端处理所有选中的 Verify。例如 B=4，队列 `V_A,P_B,V_C,P_D,V_E` 的首批依次执行 `P_B`、`P_D`、`Verify[A,C]`，E 留到下一批。已完成 Prefill 的 session 可以立即发布下一轮 Verify。它与之后的任务一起参加下一次选批。

接收侧为控制消息记录发布序号，执行侧在进入每个阶段前检查已发布的取消。已封批的取消会缩小本批。控制消息通过现有 HTTP `response_key` 和 session 生命周期协议完成，KV 的释放由执行侧负责。队列中等待的 session 只保活，已选 session 的 KV 才有保护。每个物理阶段推进一次缓存使用时钟。零预算时 KV 常驻，正预算时显式使用 LRU 和按需恢复。OOM 通过错误响应和 worker 退出报告，调度器保持原选批及输入长度。

## 三组受控配置

| 结果标识 | 云端选批 | KV 驻留与预取 | 适用问题 |
| --- | --- | --- | --- |
| `specedge_sched` | 就绪 RPC 的 dynamic FIFO，B 条即满 | 常驻，或 LRU + demand restore，预测预取关闭 | 共享单链端侧下的 SpecEdge 云端调度基线 |
| `fastsd_full` | 原 FastSD 分类、优先级、WRR 及 token budget | 成本感知驻留，允许原预测预取；零预算时常驻 | 完整云端方法的综合收益 |
| `fastsd_sched_control` | 同一 FastSD 调度器 | 与基线相同的预算、LRU、关闭预测预取 | 控制缓存策略后的调度差异 |

三组都固定端侧 gamma，打开增量 tail/bridge 协议和首轮 Prefill/Draft 重叠，关闭 proactive。FastSD 保留为优先级提供输入的 timing 协议。目标模型、草稿模型、tokenizer、工作负载哈希、实际输入 token 哈希、物理 draft 数、线程、B、采样、诊断与停止条件应在对比记录中一致。示例配置使用 4 个 CPU draft、B=16。实际主实验请把双方统一设为经验证的资源配置。

## 启动和结束一次比较

以下为仓库根目录下的 Bash 命令。两台主机均使用同一提交、兼容的运行环境和相同的 workload。每台主机将 `TARGET_MODEL` 和 `DRAFT_MODEL` 指向本地 checkpoint 目录。云端的投机服务也会加载 draft tokenizer。更换硬件、dtype、采样方式、输出长度或 B 时，复制 `configs/baselines/specedge_sched.example.json`，通过 `--config` 向两端传入同一份配置。示例 JSONL 含 16 条带固定到达时间的短验收请求。

```bash
export TARGET_MODEL=/models/target DRAFT_MODEL=/models/draft
export WORKLOAD="$PWD/configs/baselines/specedge_smoke.jsonl"
export SERVER_URL=http://127.0.0.1:8001
METHOD=specedge_sched
COMMON=(--method "$METHOD" --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL"
        --workload "$WORKLOAD" --gamma 4 --clients 4 --kv-budget-mb 0
        --run-dir "$PWD/exp/smoke-g4-c4")
```

先在云端运行第一条命令。第二个终端设置相同变量，等待 `/health` 返回 `status=ok` 后启动 edge。两机实验可使用 SSH 转发访问云端回环地址，或用 `--bind`、`--port` 和 `--server-url` 显式设置受信网络地址。`--print-only` 输出参数而无需加载模型。每次运行使用新的结果目录，以免追加到上一次的请求日志。

```bash
python scripts/run_specedge_scheduler.py --role cloud "${COMMON[@]}"
# 在端侧终端
curl -fsS "$SERVER_URL/health"
python scripts/run_specedge_scheduler.py --role edge "${COMMON[@]}" --server-url "$SERVER_URL"
curl -fsS -X POST "$SERVER_URL/exit"
```

`/exit` 先关闭接收，worker 执行已接受的尾批，写出 `scheduler_metrics.json`。此后可在云端终端用 Ctrl+C 退出 HTTP 服务。分别将 `METHOD` 置为 `fastsd_full`、`fastsd_sched_control` 并重建 `COMMON` 即可运行另两组。在固定 workload 上比较 `--gamma 8`、`--clients 16` 和正 KV 预算时，也要重建参数并更换运行目录。拒绝采样验收将配置中的 `verify_method` 改为 `rejection`，共同设定温度和 `draft_top_k`。

## 记录与验收

- `<run-dir>/<method>/<role>/manifest.json` 保存 Git SHA、官方参考提交、方法范围、checkpoint 和 tokenizer 标识、dtype、主机设备、后端、命令、工作负载哈希及 arrival seed。worker 启动时还输出实际加载的 device/dtype 和有效预取状态。
- 云端 `scheduler_metrics.json` 的 `fastsd_scheduler` 对象包含 `admission_batches`、`admitted_requests`、`prefill_forwards`、`verify_forwards`、`verify_rows`、`query_tokens`。`verify_rows / verify_forwards` 表示平均物理 Verify 批大小。有工作的阶段才计数。查看 `lookahead_build_count`、`prefetch_candidates`、`prefetch_bytes` 以及独立的 `demand_restore_bytes`、`demand_restore_stall_ms`。
- 短验收时在双方对等启用 `FASTSD_DIAGNOSTIC_DIR`。云端 `events-*.jsonl.gz` 中的 `specedge_seal` 包含成员、worker 序号、排队时间和封批原因，`specedge_phase` 包含实际阶段、执行时刻及有效 query token 数。`_diag_http_queue_mono_ns` 是云端发布时刻，`worker_enqueue_monotonic` 是 worker 发布时刻。端侧 `scheduled_arrival_s` 与 `arrival_lag_ms` 使用 workload 的时间基准。正式运行关闭细粒度 trace 和 `measure_forward_gpu`。统计每批 GPU 时长时按 forward 计数一次。
- 端侧 `edge_metrics_proc*.jsonl` 和 `edge_metrics_summary.json` 沿用共享的 TTFT、TPOT、E2E、接受数、草拟数、验证轮数和最终输出 token 指标。`coverage.json` 给出完成、失败和未完成样本，`failed_requests_proc*.jsonl` 保存发生异常的样本和错误。比较吞吐时两组使用相同测量窗口，分子为最终提交的输出 token。需要能耗时在双方同一配置中启用 `measure_energy`。

```bash
python -m unittest discover -s tests -p 'test_specedge*.py' -v
```

2026-11-26 在 Windows CPU、PyTorch 2.6.0 和 Transformers 4.57.6 环境验证。12 项测试涵盖封批时序、取消与释放、错误清理、HTTP 幂等与冲突、TTL、模式隔离、旧模式回归、受控拒绝采样和 EOS。随机初始化的小型 Llama 在一次真实 varlen Verify 中验证三个不同长度的 session，分别得到零接受、部分接受、全接受，均与逐 token 贪心参考一致。两轮通过实际 cloud/edge HTTP 服务运行的双客户端冒烟采用 gamma 4 贪心和 gamma 8 拒绝采样，每轮完成 4/4 个请求，结束时活动缓存数和预取字节数均为零。低负载 HTTP 冒烟中 Verify 均为单行，多行验证来自固定轨迹的真实模型测试。

目标 GPU 验收待在实验硬件上运行。用上述命令先做 gamma 4/8 的 2–4 session 短轨迹，再做 16-client，检查多个 Verify 共用一次 forward、阶段时间互斥、与同 checkpoint 的贪心逐 token 结果一致，以及 EOS 与不足 gamma 的尾轮。在容量受限配置检查实际发生的 D2H 和 demand H2D、零预取、LRU 次序；在模型加载、首次 Prefill、稳态和退出四个时点采集 RAM/显存，并与释放后的活动 session 数对照。CPU 冒烟数据仅用于功能验收。

## 论文表述

为控制端侧草拟和推理后端差异，我们在统一的单链投机解码框架中复现 SpecEdge 的服务端动态 FIFO 组批策略，记为 SpecEdge-Sched。该基线按就绪任务的入队顺序选批，队列暂时为空或达到请求数上限时立即执行。端侧草拟、模型执行、采样与通信实现由各方法共享。在线 Prefill 沿用共享框架的独立 RPC 协议，显存受限配置采用 LRU 淘汰和按需恢复作为运行时适配。该比较覆盖服务端调度机制。
