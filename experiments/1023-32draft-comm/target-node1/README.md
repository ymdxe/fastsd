# 32 draft 跨机通信测试

日期：2026-10-23。32/32 请求成功，32 个 draft 进程均已退出。

## 实验内容

node3 启动 32 个独立 Qwen3-1.7B draft；node1 启动一个 Qwen3-1.7B 临时 Target。复用项目 vanilla/Standard SD 的真实 session/init → prefill → verify → 接受/回滚流程，测试通信与并发。

## 配置与步骤

- 两侧模型均为 `/mnt/nfs0/zhangh/models/Qwen3-1.7B`，FP16，greedy，γ=4。
- node3 每 worker 一个独立物理核心、一个计算线程，4 个 NUMA 节点各 8 个，内存绑定同节点。选择和核验记录见 worker_map.json、cpu_selection.json、run32/worker-*.json。
- node1 使用 NUMA 0 的 CPU 0,2,4,6,8,10,12,14，8 个计算线程；HTTP `10.66.0.3:18601`，内网直连，没有 SSH 隧道。
- 数据：用户指定 `/home/hdd3/zhangh/workspace/fastsd/data/gsm8k.jsonl`，SHA256 见 manifest；共 1319 条。本次取前 32 条，每 worker 一条，最多输出 16 tokens。保留原题，使用 Target non-thinking chat 模板，不添加 8-shot 示例。
- 先进行 1 worker、4-token 验收，然后加载 32 份独立 draft 权重，通过 barrier 统一释放。没有跨进程共享权重，也没有正式 warmup。
- 每次 RPC 记录本机单调时钟和 Target 返回的服务耗时；使用同一节点的时钟计算时长，不相减不同主机的 monotonic 时间。
- 单 worker 阶段使用 mt_bench 编码标志调用项目 non-thinking 模板，数据仍为 GSM8K；32-worker 阶段实验适配器显式实现同一模板，dataset 标签为 gsm8k。

## 实验结果

| 指标 | 结果 |
| --- | --- |
| 完成请求 | 32/32 |
| 有效输出 tokens | 512 |
| 验证轮次 | 128 |
| draft 接受率 | 100.00%（同模型临时 Target） |
| 请求 E2E p50 / p95 | 202.89 / 259.66 秒 |
| 首次 draft 耗时 p50 / p95 | 135.51 / 209.66 秒 |
| TTFT p50 / p95 | 173.24 / 230.01 秒 |
| verify HTTP 总耗时 p50 / p95 | 240.69 / 409.16 ms |
| Target verify 计算 p50 / p95 | 222.37 / 242.55 ms |
| HTTP 减 Target 服务耗时 p50 / p95 | 4.44 / 4.73 ms |
| 空载复用连接 HTTP p50 / p95 | 3.18 / 4.81 ms |
| 应用层请求 / 响应 JSON 字节 | 133509 / 29961 |
| 估计公共窗口 / 总吞吐 | 296.88 秒 / 1.725 tokens/s |

## 解释与限制

- 32/1319 samples; truncated 16-token communication test, not accuracy evaluation
- Same checkpoint used on both ends; acceptance not representative of 32B target
- node1 torch2.6 vs node3 torch2.2; do not attribute compute gap only to CPU hardware
- HTTP residual includes serialization, transport and runtime overhead; not pure network RTT
- One-way push/pull fields are unavailable for vanilla unsynchronized clocks; zero placeholders in native metrics must be read as NA
- Shared server; CPU affinity is not exclusive reservation
- No warmup; loading synchronized but model compute cold

本轮可确认 32 个独立 draft 与跨机 CPU Target 的真实请求和验证流程成功，网络、服务计算和本地草拟耗时可追溯。不能据此评价 Qwen3-32B、FastSD 调度收益或 GSM8K 答题准确率。

## 文件

- `manifest.json`、`commands.txt`、`*_launch.json`：配置、原始数据 hash、启动命令。
- `source/`：同一版本代码快照；`edge_comm.py`：实验专用绑核、NUMA、启动屏障、模板和 RPC 记录适配器。
- `workload32.jsonl`：本次 32 条原题和参考答案。
- `run32/edge_metrics_proc*.jsonl`：逐请求输出 token、文本、接受率和耗时。
- `run32/rpc-*.jsonl`：逐次 HTTP 请求日志。
- `communication_summary.json`：从原始记录生成的完整汇总。
- `smoke1/`：单 worker 冒烟记录；`run32.log`：32 worker 完整输出。
- node1 同名目录保存 Target 配置与 `target.log`。

运行界面均在 tmux 会话 `6`：`6:0.0` 为 node1 Target，`6:0.1` 为 node3 draft。用户的 `6:0.2` 保持原状。
