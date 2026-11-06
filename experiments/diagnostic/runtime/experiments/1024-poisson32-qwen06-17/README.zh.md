# Qwen3 CPU32 / Target1.7B 泊松实验

本轮由用户于 2026-10-24 授权，在 `exp` tmux 会话运行。主调度器位于 `exp:campaign`；`exp:1` 链接原有两个 CPU SSH pane，仅使用 node1/node3 两个 pane。

- CPU node1/node3 各16个独立 Qwen3-0.6B FP16 draft；每实例一个物理核、一个计算线程，NUMA 分配固定。
- gpunode2 物理 GPU1 / UUID `GPU-422501a2-5189-7f5b-5ce6-95464276b27f`：Qwen3-1.7B FP16，权重全部驻 GPU。
- GSM8K test 全部1319题；同一非 thinking 提示模板、greedy、EOS、每题最多1024个输出 token。
- γ={200,500} × 全局 λ={50} 请求/秒 × {标准SD,FastSD,SpecEdge单链}，共6组、7914个正式请求。
- 原生服务 batch_size=8、token_budget=2048、padded KV；SpecEdge batch_size=8、静态 max_len=3072，无同等 token_budget 机制。
- 固定到达种子20261024。三个速率使用同一组指数间隔按 λ 缩放；相同 λ 的所有方法、γ使用完全相同的1319个到达时刻。

所有题按原始索引对32个 draft 做轮转分配。到达后若对应 draft 忙碌，任务留在该 draft 的等待队列，后续到达不等前面的请求完成。`arrival_lag_ms` 是该等待时间；`arrival_ttft_ms` 从计划到达计时，`service_ttft_ms` 从实际开始服务计时，两者都保留。TPOT 从首批验证输出可见到完成摊销，不加入首 token 前的等待。

源代码在实验前 fetch 上游 main；上游仍为 `ee0e334e222593c100b2ee6fc7fc9e4db2cf7cd3`。快照包含本地 kvpre 修复，未 commit/push。`manifest.json` 保存源代码和数据哈希，CPU 启动时逐文件校验。推理快照仅增加缓存搬运的计时包装、验证时预取驻留命中观察，不新增 CUDA 同步或改变调度策略。32个原生客户端各自复用一个 session，以便全量数据集运行时重复利用32份 KV 槽位，防止随1319题累计保留缓存。

GPU 能耗按物理 UUID 以100ms间隔采样板卡功率，积分首个计划到达到最后一个请求完成的区间；排除模型加载，包含低到达率下的 GPU 空闲时间。报告总 J/Wh、平均 W、J/请求、J/输出 token。内置能耗服务固定 NVML GPU0，因此本轮关闭，三种方法统一由外部采样器记录。现有 PyNVML v1 内存字段含驱动保留量，空闲检查同时验证计算进程，不能直接与 nvidia-smi used 字段比较。

KV 记录包括 kvpre 原生计数器、搬运字节与主机墙钟耗时、验证真正开始时的预取驻留命中。`.to()` 本身的阻塞和流等待包含在主机耗时中；不将其当作纯 PCIe 时间或 CUDA 复制计算重叠证据。SpecEdge 保留官方单链推测、主动草拟、验证和 KV 处理，原生异步 KV 预取为不适用。

使用现有 SSH 连接上的 loopback reverse forward：CPU `28634→18634` 控制、`28635→18635` 推理。控制器只监听 gpunode2 的127.0.0.1，带认证；Target也只监听127.0.0.1。此配置不用于宣称 IB 性能。

关键配置校验见 `validation.json`：全1319题两模型输入 token 相同，prompt长35–200 tokens；γ500最大 protobuf约3.1MB；3072上下文足以容纳 prompt、1024输出及两条草拟链。没有独立模型 smoke，直接观察第一组正式请求。

`status.json` 是全局状态，`RESULTS.zh.md`/`results.csv`/`results.json`每分钟和每组结束时更新。每组目录中：

- `node1/requests.jsonl`、`node3/requests.jsonl`：逐题输出和时延指标。
- `target/kv_moves.jsonl`：实际缓存搬运及 demand hit。
- `target/runtime_metrics.json`、结束后的 `target/scheduler_metrics.json`：调度与预取计数。
- `power.jsonl`、`energy.json`：原始GPU功率与最终积分。
- `target/launch.json`、`target/exit.json`、各节点 `status.json`：命令、进程、负载、退出状态。

CPU 端完整逐轮日志保留于 `/home/zhangh/workspace/fastsd/experiments/1024-poisson32-qwen06-17`，每组结果持续上传至本目录。失败时调度器停止推进，保留部分结果；`campaign.py --resume`只跳过已完成组，未完成组需审查后使用新 attempt 目录，绝不覆盖既有结果。

本轮每组的最后一个计划到达约为26秒；完整运行仍需要CPU草拟和队列排空。结果按完成状态解释，部分组的均值不能与完整组直接比较。

当前用户仅要求 γ=200/500、λ=50 的六组实验。旧 γ20/λ1 组已按指令停止，部分结果完整保留，不计入当前六组统计。旧27组配置保存在 subset-g200-g500-l50/backups，原组间切换监视器已取消。
