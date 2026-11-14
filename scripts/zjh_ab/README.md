# KV prefetch 参数与验收

默认使用 h1-only 最小预取，保留 legacy ticket/coordinator/arena 路径供回退；实际 WRR/softfill 调度策略不变。
模型路径继续由原有 `params.env` 或环境变量提供。

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `FASTSD_WAIT_LAMBDA` | `0.01` | exp 与分段公式共用的 lambda |
| `FASTSD_WAIT_LINEAR_UNTIL_S` | `0` | `0` 为纯 exp；正数开启分段公式 |
| `FASTSD_WAIT_LINEAR_ALPHA` | 自动 | 未指定时取 `lambda * exp(lambda*T0)`；显式 `0` 有效 |
| `KV_PREFETCH_IMPLEMENTATION` | `h1` | `h1` 为最小路径；`legacy` 回退旧实现 |
| `KV_PREFETCH_LOOKAHEAD_ROUNDS` | `1` | legacy 预测深度 K；h1 固定只预测一轮 |
| `KV_PREFETCH_HORIZON_LIMIT` | `1` | legacy 的搬运深度；h1 固定为 1 |
| `KV_PREFETCH_H2_MAX_PIDS` | `4` | legacy h2 pid 上限；h1 不执行 h2/h3 |
| `KV_PREFETCH_RATE_MB_S` | `0` | 计算期间的提交速率上限，MiB/s；`0` 表示不限速 |
| `MEASURE_ENERGY` | `0` | 设为 `1`/`true` 后记录 target GPU system energy、平均/峰值功耗和 P95/P99 |
| `KV_PREFETCH_IDLE_RATE_MB_S` | `0` | GPU 空闲时的提交速率，MiB/s；`0` 表示不限速 |
| `KV_PREFETCH_CHUNK_KB` | `4096` | 一组提交的最大 KiB；普通 demand ticket 的 tensor 分块上限 |
| `KV_PREFETCH_SPECULATIVE_CHUNK_KB` | `4096` | 预取 ticket 的单块上限；默认跟随普通 chunk |
| `KV_PREFETCH_BEFORE_SUBMIT` | `0` | legacy 的 submit 前预测；h1 始终在实际提交后预测 |
| `FASTSD_IDEMPOTENCY_MAX_ENTRIES` | `4096` | target 已完成成功响应的 LRU 容量 |
| `FASTSD_IDEMPOTENCY_TTL_S` | `1800` | 成功响应最后一次访问后的保留秒数 |

Edge 每个有状态 POST 都生成 `request_seq` 和 `request_id`。序号按
`(session_id, task_id)` 递增，ID 编码 `POST/route + session_id + task_id + request_seq`；
同一逻辑请求的网络重试复用完整 JSON 请求体。仅连接中断类错误最多重试三次，
等待 `0.5/1/2` 秒，重试前关闭并重建对应的 HTTP Session。
4xx、5xx、业务错误和 timeout 均不自动重发，包括时钟校验的 422。
重试不会初始化新 target session 或重新生成 draft。

Target 在 HTTP 层合并同 key 的 in-flight 请求，仅缓存成功响应；HTTP 连接
取消不会取消共享操作。缓存独立于 KV session，session 释放后在 LRU/TTL
保留期内仍能 replay。响应中的 `replayed` 区分首次执行与缓存/共享响应。
`runall` 收集 `target/idempotency_metrics.json`，包含 `idempotent_cache_hit`、
`idempotent_inflight_wait`、`cache_evictions`；`/health` 同时返回即时统计。
Edge 逐任务 JSONL 记录 `retry_count`、`last_error`、`replayed` 和 `replayed_count`，
最终失败的连接尝试也写入 edge 日志。

Offload 保留预算、水位和 session 保护规则。仅在需要回收显存时，对 GPU 上
未保护的 session 计算 `score = (draft_total_time + transport_rtt) / accept_prob - idle`，
按 score 降序搬运到 CPU，直到释放足够字节。`draft_total_time` 使用最近一轮
请求的 `lag`，接受率沿用调度器的平滑估计；idle 使用单调时钟秒数。

等待公式为 `exp(lambda*t)`。分段模式在 `t <= T0` 时使用
`beta + alpha*t`，其中 `beta = exp(lambda*T0) - alpha*T0`，保证值连续；
自动 alpha 还保证斜率连续。lambda、T0、alpha 必须有限且非负；lambda=0
关闭等待增长。环境变量在进程启动时读取，实际配置写入 `wait_priority_config`。

## 最小 h1 路径

每轮实际 plan 提交后预测下一轮 verify pids。只准备 CPU resident、非 protected、
非 active/reserved、非 in-transfer 的 session；CPU cache 的 D2H producer event
也必须已完成，源 tensor 必须 pinned。每个 pid 最多一个未退休 ticket。

创建前检查 `free >= ticket_bytes + 256 MiB`，不足则请求执行端在下一个安全边界
按 offload 排名回收，再复查；仍不足跳过本次预取。执行端不等待这次预取重试，
预取线程不直接修改正式 KV。`KV_PREFETCH_MB` 限制所有未退休 destination 总字节，
包括 demand 已弃用但 DMA 尚未完成的 buffer；它不再预分配 arena。

snapshot 包含 cache 对象身份、session_epoch、content_version、valid_len，分配后
再次校验；全部 GPU destination 和 event 在 copy 前创建。独立 CUDA stream 一次
提交完整 KV 的异步 H2D，中途不分配显存、不检查 chunk 显存、不取消或 supersede。

执行端真正 acquire KV 时才从 candidate 表 pop。只有 event.query 已完成且 snapshot
匹配才安装 destination；pending/stale 都走普通 demand restore，不等待 speculative
stream，也不推进 KV。用完 retire；未完成的旧 copy 保留源 pinned cache、destination
和 event，event 完成后才能释放，session 释放和版本变化也遵守这一规则。
GPU 安装后清除 residency record 的旧 CPU cache 引用；有在途 copy 时 ticket 仍保持源生命周期。

回退：target 加 `--kv_prefetch_implementation legacy`，或运行
`KV_PREFETCH_IMPLEMENTATION=legacy bash scripts/zjh_ab/runall.sh`。
legacy 的 horizon、chunk、rate、calibration、arena 和 event pool 参数仍保留；
这些控制不参与最小 h1 的 copy。`--kv_prefetch_mode cpu_prepare` 不提交 h1 DMA。

## 本轮两组对照

在原有拓扑环境变量已设置的 node1 上运行：

```bash
RUN_TAG=h1_pair_$(date +%Y%m%d_%H%M%S) bash scripts/zjh_ab/runall.sh
```

只跑 `<tag>_noprefetch`、`<tag>_prefetch` 两组，各 64 个请求（两台 edge 各 32）。
同一数据划分、arrival schedule 和 seed；默认 gamma=8、token budget=32、
KV budget=1800 MiB、prefetch destination 上限=256 MiB。
`DIAG_GAMMA`、`DIAG_BUDGET`、`DIAG_KV_BUDGET_MB` 可同时修改两组的参数；
`TOTAL_REQUESTS` 可改请求数。`DIAGNOSTIC_ONLY` 不再切换到多组矩阵。
失败启动、edge 失败、收集失败或报告 gate 失败都会使 runner 非零退出。

报告严格检查唯一任务数量、逐任务 output_token_ids（旧记录回退 output_text）、
CUDA OOM、copy 错误、TTFT/TPOT 的 p50/p95 和总 H2D（包含 demand 与 speculative）。
runall 延迟容差默认 0（不退化），H2D 增长容差默认 10%；可分别设
`PREFETCH_LATENCY_TOLERANCE`、`PREFETCH_H2D_TOLERANCE`。
无重复 KV 推进由重试重放和 ready/pending acquire 回归测试覆盖，输出比较提供实验侧检查。

可设 `PREFETCH_LEGACY_REFERENCE=<完整旧版 condition-dir>`，比较相同任务集的
ready 命中数；参考数据需同模型、调度和预算。未提供参考时明确标记 UNVERIFIED，
不会宣称“ready 不下降”已通过；不完整历史数据也不能通过完整性检查。
`runtime/diag_final` 与 `runtime/overnight` 保留为历史分析数据，不替代本轮 n=64 验收。

## 指标与报告

合并两台 edge 的 `edge_metrics_proc*.jsonl` 和 `edge_metrics_summary*.json`
到同一条件的 `edge/`，文件名保留主机区分，避免同名覆盖；target 的
`scheduler_metrics.json` 放在该条件的 `target/`。

```bash
python3 scripts/zjh_ab/accept.py <condition-dir> <another-condition-dir> \
  --prefetch-pair --expected-requests 64 --json-output /tmp/kv-prefetch-report.json
```

`n < 64` 标记 incomplete，并使完整性 gate 失败；小型 smoke 可显式调整
`--expected-requests`。旧 run 缺新字段时显示 `n/a`，ready=0 时每 ready
的成本为 null，不能解释成零成本。

主表 tok/s 使用两台 edge 的 benchmark 起止 epoch 包络，包含进程启动、
warmup 和负载空隙。旧 summary 只有 wallclock_s 时回退并标记
`legacy_wallclock_duration`；没有 summary 时回退为 `legacy_request_span`。
旧数据无法恢复精确的跨主机 benchmark 起止边界。busy 吞吐仅作诊断。

| 指标 | 口径 |
| --- | --- |
| `prefetch_refresh_count` | 实际重新预测的次数，不含 signature 相同的跳过 |
| `lookahead_build_count` | 预测 plan 的 build 次数；h1 每次 refresh 最多 1 次 |
| `prediction_snapshots_consumed` | 各 horizon snapshot 被实际 plan 消费的总数 |
| `actual_plans` | 成功提交给 executor 的实际 plan 数 |
| `ready_per_actual_plan` | ready cache 次数 / actual plans，可大于 1，不是概率 |
| `h2d_bytes_per_ready`、`copies_per_ready` | 全部 H2D bytes / ready，及全部 copies / ready；copies 包括 D2H |
| `prefetch_ready_hit` | demand acquire 时已完成且 snapshot 匹配的次数 |
| `prefetch_pending_hit` | claim 遇到未完成 event，立即走 demand restore 的次数 |
| `prefetch_demand_miss` | 所有 CPU cache demand restore 次数，包含 pending 回退 |
| `prefetch_skipped_no_memory` | admission 预算/空闲显存不足而跳过的次数 |
| `prefetch_skipped_stale` | 分配、claim 或完成回收时 snapshot 失效的次数 |
| `prefetch_copy_ms`、`prefetch_copy_count` | 已完成 speculative copy 的 CUDA event 总毫秒和次数 |
| `prefetch_lead_ms`、`prefetch_lead_count` | 从 destination 准备到 claim 的总毫秒和样本数 |
| `prefetch_bytes` | 完整提交的 speculative H2D 累计字节 |
| `prefetch_candidates` | 当前 candidate 表大小；shutdown 后归零 |
| `prefetch_candidates_created`、`prefetch_candidates_peak` | 累计创建 ticket 数、candidate 峰值 |
| `prefetch_reserved_bytes_peak` | 未退休 destination 的最大总字节，含 draining buffer |

`prefetch_copy_ms / prefetch_copy_count` 是起止 CUDA event 之间的时间，
包含区间内调度和资源竞争，不包含 start event 之前的 stream 排队，不能直接解释为 PCIe 带宽。
legacy 仍保留其 horizon、intent、retire 和最近 ticket 明细，耗时口径仍为
host 首次提交至观察到 event 完成，不应直接与 h1 的 CUDA event 时间等同。
ready=0 时每 ready 成本为 null；历史缺失字段显示 n/a。

## 验证

```bash
python3 -m unittest discover -s tests -v
```

CUDA 测试需要可用 GPU；关闭流程测试需要 asyncio 跨线程唤醒使用的本地
socketpair。测试环境禁止该系统调用时，需要在允许 socketpair 的环境中跑。

## runall 的启动、停止和失败清理

当前默认拓扑为 node1 CPU（控制器 + edge）、node2 CPU（edge）、node2 GPU2
（target）。两边各 16 workers / 82 条 HumanEval，draft 均为 Qwen3-0.6B。

在 node1 CPU 仓库根目录启动，继续使用 nohup：

```bash
mkdir -p /mnt/nfs0/zhangh/fastsd_exp/logs
TAG=n1n2_gpu2_$(date +%Y%m%d_%H%M%S)
nohup env RUN_TAG="$TAG" TARGET_GPU=2 bash scripts/zjh_ab/runall.sh \
  >"/mnt/nfs0/zhangh/fastsd_exp/logs/runall_${TAG}.log" 2>&1 </dev/null &
```

停止最新注册的实验（也可追加具体 RUN_TAG）：

```bash
bash scripts/zjh_ab/stop_runall.sh
# bash scripts/zjh_ab/stop_runall.sh n1n2_gpu2_YYYYMMDD_HHMMSS
```

`runall.sh` 在 `runtime/runall/<RUN_TAG>/` 写入控制器身份和任务清单。
每个 target、edge、隧道由 `managed_job.py` 启动独立 supervisor，保存随机任务标识、
退出状态并清理其子进程（包括 timeout 创建的新进程组）。失败、SIGTERM、Ctrl-C 都会
触发本轮清理；主脚本被 SIGKILL 后仍可用 stop 命令按清单回收。清理可重复执行，
不会删除实验数据，也不会按进程名字或 GPU 编号批量杀进程。

端口检查在加载模型前执行；端口被其他运行占用时立即失败。HTTP 健康检查绕过代理，
并核对 target 监听端口属于本轮任务。隧道健康检查默认 45 秒（TUNNEL_TIMEOUT）。
edge 非零退出或 supervisor 丢失会终止本轮，不会按正常完成继续下一轮。

旧版 runall 启动的进程没有任务记录，仍需按已确认的 PID/端口做一次清理；新脚本
不会自动接管或杀掉这些未知进程。若远程 SSH 不可达，清理会明确失败并保留记录，
恢复连接后重新执行同一条 stop 命令即可。
