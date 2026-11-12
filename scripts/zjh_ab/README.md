# KV prefetch 参数与验收

本轮修改保持实际 WRR/softfill 调度策略，只调整等待项配置、预测副本和 KV 预取。
模型路径继续由原有 `params.env` 或环境变量提供。

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `FASTSD_WAIT_LAMBDA` | `0.01` | exp 与分段公式共用的 lambda |
| `FASTSD_WAIT_LINEAR_UNTIL_S` | `0` | `0` 为纯 exp；正数开启分段公式 |
| `FASTSD_WAIT_LINEAR_ALPHA` | 自动 | 未指定时取 `lambda * exp(lambda*T0)`；显式 `0` 有效 |
| `KV_PREFETCH_LOOKAHEAD_ROUNDS` | `1` | 预测深度 K，与实际搬运深度独立 |
| `KV_PREFETCH_HORIZON_LIMIT` | `1` | 实际预取的最大 horizon；`0` 保留预测但不搬运 |
| `KV_PREFETCH_H2_MAX_PIDS` | `4` | 每次 intent 的 h2 pid 上限；`0` 禁止 h2 |
| `KV_PREFETCH_RATE_MB_S` | `0` | 计算期间的提交速率上限，MiB/s；`0` 表示不限速 |
| `MEASURE_ENERGY` | `0` | 设为 `1`/`true` 后记录 target GPU system energy、平均/峰值功耗和 P95/P99 |
| `KV_PREFETCH_IDLE_RATE_MB_S` | `0` | GPU 空闲时的提交速率，MiB/s；`0` 表示不限速 |
| `KV_PREFETCH_CHUNK_KB` | `4096` | 一组提交的最大 KiB；普通 demand ticket 的 tensor 分块上限 |
| `KV_PREFETCH_SPECULATIVE_CHUNK_KB` | `4096` | 预取 ticket 的单块上限；默认跟随普通 chunk |
| `KV_PREFETCH_BEFORE_SUBMIT` | `0` | `1` 在当前 plan 完成 reserve 后、executor submit 前预测 |

等待公式为 `exp(lambda*t)`。分段模式在 `t <= T0` 时使用
`beta + alpha*t`，其中 `beta = exp(lambda*T0) - alpha*T0`，保证值连续；
自动 alpha 还保证斜率连续。lambda、T0、alpha 必须有限且非负；lambda=0
关闭等待增长。环境变量在进程启动时读取，实际配置写入 `wait_priority_config`。

无 calibration 时，计算窗口速率取
`min(RATE, IDLE_RATE * online_fraction)`，随后仍可因 forward 干扰而回退。
速率控制是带一个 chunk 初始额度的 token bucket，不是 PCIe 带宽承诺。
若要做“解除软件限速”的机制实验，可将 `KV_PREFETCH_ONLINE_FRACTION=1`、
`KV_PREFETCH_RATE_MB_S=0`/`KV_PREFETCH_IDLE_RATE_MB_S=0`，并增大
`KV_PREFETCH_SPECULATIVE_CHUNK_KB`。这只移除本项目的 token-bucket 和分块限制，
不会突破实际 PCIe/NVLink、CPU pinned memory 或单个 speculative chunk 的串行提交能力。
新配置下 `K=3` 默认也只搬 h1；如需复现旧 union 搬运，必须显式设置
`HORIZON_LIMIT=3` 及足够大的 `H2_MAX_PIDS`。

## 推荐的下一轮对照

今天用于定位 FastSD 落后标准 SD 的诊断矩阵可直接运行：

```bash
DIAGNOSTIC_ONLY=1 MEASURE_ENERGY=1 RUN_TAG=diag_$(date +%Y%m%d_%H%M%S) \
  bash scripts/zjh_ab/runall.sh
```

它固定 `gamma=8`、`token_budget=32`、KV budget=2GiB、prefetch rate unlimited，依次比较
标准 SD offload、FastSD 无 prefetch、完整 FastSD，以及分别关闭 pipeline、proactive、latency
priority 的三个 FastSD 变体。结果会在每个 condition 的 `target/scheduler_metrics.json`、
`target/energy_metrics.json`、`edge/edge_metrics_summary*.json` 和 `report.json` 中保留。

保持 target Qwen3-8B、CPU draft Qwen3-0.6B、32 drafts、batch=8、
token budget=32、gamma=8、max tokens=400、KV budget=2048 MiB、
prefetch=256 MiB、每 worker arrival rate=1.6、edge threads=4 不变。

先在 target 所在主机启动下列配置；每次使用新的 `RUN_NAME`，另在原有两台
edge 上用相同 run 名及原有 82/82 数据划分启动客户端。这里的命令只启动 target。

```bash
RUN_NAME=kv_h1_protected_k3_r01 \
FASTSD_WAIT_LAMBDA=0.01 FASTSD_WAIT_LINEAR_UNTIL_S=0 \
KV_PREFETCH_LOOKAHEAD_ROUNDS=3 KV_PREFETCH_HORIZON_LIMIT=1 \
KV_PREFETCH_H2_MAX_PIDS=0 KV_PREFETCH_MB=256 \
KV_PREFETCH_RATE_MB_S=256 KV_PREFETCH_IDLE_RATE_MB_S=1024 \
KV_PREFETCH_CHUNK_KB=4096 KV_RESIDENCY_BUDGET_MB=2048 \
NUM_DRAFTS=32 BATCH_SIZE=8 TOKEN_BUDGET=32 GAMMA=8 MAX_TOKENS=400 \
./scripts/zjh_ab/serve.sh fastsd
```

建议依次单独比较以下变化：

1. 同预算的 `fastsd-noprefetch` 与上面的 h1 预取。
2. K=1 与 K=3，保持 `HORIZON_LIMIT=1`，观察预测成本和 h1 ready。
3. `HORIZON_LIMIT=2, H2_MAX_PIDS=4`，观察额外 bytes/eviction 是否值得。
4. chunk=512/1024/4096 KiB，RATE=256/512/1024 MiB/s；一次只改一个参数。
5. `BEFORE_SUBMIT=1`，测量提前触发收益及当前 plan 的延迟。
6. lambda=0.005/0.01/0.02；分段 T0=5/30、alpha=0.02 作为独立实验。

标准 SD 若要比较 KV 压力，也必须使用相同非零 residency budget。
原始表中 SD 的 budget=0、全驻留，不能作为同 KV 压力的对照。
旧远程 ablation 脚本在 SSH 内设置自己的参数；新的环境变量应放在远端
`env ... serve.sh` 命令中，不能仅设在发起 SSH 的本机。

## h1 保护规则

预取 intent 用 `PrefetchTarget(pid, horizon, prediction_tick)`；旧 pid-only
调用仍支持。pid 去重取最早 horizon，h1 先分配预算、先提交 DMA。
已有有效 h1 不会因降为 h2/h3 或同一 actual plan 内的 refresh 而 retire；
即使当前 intent 只包含 h2/h3，保留的 h1 也继续推进。
新 h1 需要空间时优先 retire h2/h3，等待旧 DMA 排空才复用内存。
如果 h1 分配失败，本轮不会继续给 h2/h3 分配剩余预算。

显式空 intent、cache generation/epoch 变化仍会清理候选；actual plan
推进后，新的 h1 intent 可以清理已经不再预测到的旧 h1。这样保护不会阻止
正常的失效处理。耗时准备期间收到低优先级刷新也不会丢弃有效 h1 ticket。

## 指标与报告

合并两台 edge 的 `edge_metrics_proc*.jsonl` 和 `edge_metrics_summary*.json`
到同一条件的 `edge/`，文件名保留主机区分，避免同名覆盖；target 的
`scheduler_metrics.json` 放在该条件的 `target/`。

```bash
python3 scripts/zjh_ab/accept.py <condition-dir> <another-condition-dir> \
  --expected-requests 164 --json-output /tmp/kv-prefetch-report.json
```

`n < 164` 标记 incomplete，并使完整性 gate 失败；小型 smoke 可显式调整
`--expected-requests`。旧 run 缺新字段时显示 `n/a`，ready=0 时每 ready
的成本为 null，不能解释成零成本。

主表 tok/s 使用两台 edge 的 benchmark 起止 epoch 包络，包含进程启动、
warmup 和负载空隙。旧 summary 只有 wallclock_s 时回退并标记
`legacy_wallclock_duration`；没有 summary 时回退为 `legacy_request_span`。
旧数据无法恢复精确的跨主机 benchmark 起止边界。busy 吞吐仅作诊断。

| 指标 | 口径 |
| --- | --- |
| `prefetch_refresh_count` | 实际重新预测的次数，不含 signature 相同的跳过 |
| `lookahead_build_count` | 预测 plan 的 build 次数；K=3 通常每次 refresh 为 3 次 |
| `prediction_snapshots_consumed` | 各 horizon snapshot 被实际 plan 消费的总数 |
| `actual_plans` | 成功提交给 executor 的实际 plan 数 |
| `ready_per_actual_plan` | ready cache 次数 / actual plans，可大于 1，不是概率 |
| `h2d_bytes_per_ready`、`copies_per_ready` | 全部 H2D bytes / ready，及全部 copies / ready；copies 包括 D2H |
| `prefetch_intent_by_horizon` | 最新 intent 的 pid 数，当前值 |
| `prefetch_candidate_by_horizon` | worker 最近一次观察到的候选数量，当前值 |
| `prefetch_candidates_created_by_horizon` | 按创建 horizon 的累计候选数 |
| `prefetch_retired_by_horizon` | worker 因 intent 变化/内容失效而 retire 的累计候选数 |
| `prefetch_skipped_horizon` | 去重 pid 因 horizon limit 或 h2 限额被跳过的累计次数 |
| `prefetch_bytes_by_horizon` | 源自各 horizon ticket 的实际 H2D bytes，含后续 demand 补完 |
| `speculative_bytes_by_horizon` | 其中在 demand claim 前实际提交的 bytes |
| `kv_transfer_by_horizon` | tickets、bytes、完成/retire 数、耗时与有效速率 |

bytes 按 ticket **创建时**的 horizon 归因，ready/pending 按保护后的当前
horizon 归因；h2 提升为 h1 不会改写已搬 bytes 的来源。
每个 horizon 的 precision/recall 使用该 horizon 的集合统计，coverage 和
hits/ready 按 actual plans 归一化，不把 snapshot 总数当实际 plan 数。
后续预测在上一层队列副本上继续投影，不会复活更早层已完成的工作。

`cache_transfers.kv_transfer_recent_tickets` 保留最近 256 张有 DMA 的 ticket
明细（bytes、耗时、速率、claim/retire）；累计统计不截断。
耗时从首次提交到观察到最终 event 完成，包含限速、竞争和轮询延迟；它是
readiness 时间尺度，不等于纯 DMA 时间或 PCIe 带宽。report 的
`transfer_per_batch_window > 1` 表示平均传输耗时超过一个 batch 窗口，
本身不是因果证明。

已提供的测量 0.30–0.85s 对 0.11–0.12s，约为 2.5–7.7 个 batch 窗口；
K=3 的 0.33–0.36s 只覆盖较短的搬运。当前修改通过限制 union、减少低
horizon 搬运竞争改善准备机会，并未实现 KV 压缩或增量传输。
真实 ready、TTFT、E2E 收益仍需使用完整的 GPU 对照实验验证。

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
