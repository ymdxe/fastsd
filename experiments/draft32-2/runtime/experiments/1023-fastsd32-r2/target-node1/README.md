# FastSD 32 draft 跨机测试

日期：2026-10-23。32/32 请求成功，512 个输出 token。

## 配置与步骤

沿用 1023-32draft-comm 的源码快照、模型、FP16 精度、CPU/NUMA 分配和 GSM8K 前 32 条题目。node3 使用 32 个独立 draft，node1 使用 8 线程 CPU Target；每题最多输出 16 tokens。

FastSD 参数：`--profile custom --server_sched_mode fastsd --enable_pipeline --enable_proactive_draft --enable_latency_priority --pipeline_gamma_adapt`。初始 γ=4；其余配置见 launch JSON。两端模型为 `/mnt/nfs0/zhangh/models/Qwen3-1.7B`。

加载全部模型后以 barrier 同时释放，然后依次执行各 worker 的原生五探针时钟校准，全部校准后再经第二个 barrier 同时进入 arrival、prefill 和 tail-only verify 流程。实验适配器保留上一轮的相同 non-thinking 模板，新增时钟日志、校准启动顺序控制和完成数检查；未修改原生 FastSD 调度、计算和时间校验。

首次运行见 `../1023-fastsd32/`：并发校准后，10 个 worker 触发负 pull 时长校验，4 个收到 HTTP 422，只完成 18/32。诊断复现了负 upload 校验；本目录为串行校准后的重跑，初次失败记录完整保留。

## 与上一轮比较

| 指标 | Standard SD | FastSD |
| --- | ---: | ---: |
| 请求 E2E 中位数 | 202.89 秒 | 179.51 秒 |
| 请求 E2E P95 | 259.66 秒 | 238.17 秒 |
| 请求 TTFT 中位数 | 173.24 秒 | 150.37 秒 |
| 首轮 draft 中位数 | 135.51 秒 | 139.47 秒 |
| verify HTTP 中位数 | 240.69 ms | 508.80 ms |
| verify HTTP P95 | 409.16 ms | 7169.05 ms |
| Target verify 计算中位数 | 222.37 ms | 273.57 ms |
| HTTP 减 Target 服务耗时中位数 | 4.44 ms | 2.89 ms |
| RPC 总数（不含健康检查与时钟校准） | 192 | 288 |
| 验证轮次 | 128 | 192 |
| 估计总吞吐 | 1.725 tokens/s | 1.964 tokens/s |

本轮输出 token 与上一轮逐题完全一致的请求：**32/32**。FastSD 提前草拟复用命中 192 轮，未命中 0 轮。原生指标中的接受数/候选数为 320/320。

时钟校准不确定度最大 2.263 ms；保持项目默认 10 ms 上限。

Target 正常退出后保存的调度统计确认：FastSD 执行了 111 个调度计划、192 个 verify 切片和 33 个 prefill 切片，其中一次 prefill 被分块。提前预取任务完成 11 次、失败 0 次；这里是 CPU Target 上的逻辑缓存管理计数，不能解释成真实 GPU 传输次数或收益。

本次请求耗时中位数降低约 11.5%，估计总吞吐提高约 13.9%；同时 verify HTTP P95 升到约 7.17 秒，其耗时主要在服务端。此单次结果同时表现出端到端耗时改善和验证请求等待增加，不能只选择其中一项作为结论。

## 限制

- 单次前32题短输出通信测试，不是全量GSM8K准确率或正式速度评测
- 两侧同一1.7B模型，接受率不代表32B Target
- CPU Target不衡量GPU KV搬运收益
- 历史基线与本轮背景负载不同，不能将单次耗时差全部归因于算法
- FastSD自适应gamma开启，4是初始值；基线gamma固定4
- HTTP减服务耗时包含网络与序列化等开销，并非纯网络RTT
- 公共窗口由session/init开始时间加请求E2E估算，排除模型加载但包含校准与prefill
- node1 PyTorch2.6、node3 PyTorch2.2；无warmup

## 文件

- `source/`：与基线相同的源码快照；未修改原仓库推理源码。
- `launch_target.py`、`launch_drafts.py`、`*_launch.json`：启动脚本、命令和环境。
- `run32/edge_metrics_proc*.jsonl`：逐请求输出、耗时、接受率与复用计数。
- `run32/rpc-*.jsonl`、`run32/clock-*.json`：逐次通信与时钟校准记录。
- `communication_summary.json`：汇总与逐题输出一致性结果。
- node1 同名目录 `target/scheduler_metrics.json`：正常退出时保存的 FastSD 调度证据；`target.log`：服务日志。

运行界面：tmux 会话 6，node1 为 6:0.0，node3 为 6:0.1。实验结束后全部 draft 和临时 Target 已正常退出，窗格保留。
