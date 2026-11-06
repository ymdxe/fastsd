
## 2026-10-23 32-draft 跨机通信测试（node1/node3）

- 目的：node3 的 32 个独立 CPU draft 通过内网 HTTP 访问 node1 的临时 Qwen3-1.7B CPU Target，验证真实 Standard SD 会话、预填充和验证流程。
- 配置：两端 FP16，greedy，gamma=4；node3 32×1 线程，四个 NUMA 节点各 8 worker；node1 8 线程。tmux 会话 6。
- 数据范围：指定 GSM8K 的前 32/1319 条，每 worker 一条、最多 16 tokens；另先做一条 4-token 冒烟。
- 结果：32/32 成功，512 tokens；192 次 RPC（32 session/init、32 prefill、128 verify），无 HTTP 错误。verify HTTP p50/p95=240.69/409.16 ms；HTTP 减服务时间 p50/p95=4.44/4.73 ms，包含网络与运行时开销。请求 E2E p50/p95=202.89/259.66 s。
- 边界：同模型接受率 100% 不代表未来 32B Target；短输出不评价准确率。node1 torch2.6 与 node3 torch2.2 不同；不能把 CPU 耗时差全部归因于硬件。
- 归档：两机 /home/zhangh/workspace/fastsd/experiments/1023-32draft-comm/；node3 保存逐 worker/RPC 记录，node1 保存 target.log、启动命令与环境。node2 本地仓库 experiments/1023-32draft-comm 保存结果副本。
- 改动与清理：使用独立 source 快照和实验适配器，未改原推理源码或安装依赖。draft 全部退出，Target 正常停止，临时传输服务关闭，保留 tmux 窗格。
