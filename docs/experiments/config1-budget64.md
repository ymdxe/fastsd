# Config 1：四方法的 64 token forward 预算

本配置沿用 `paper50-1103T162643Z` 的 50 条 HumanEval、任务分配和到达轨迹。
32 个 Draft 在 node3/node1/node2 上分别为 16/8/8 个，gamma 为 8/4/4。
每个 Draft 的到达间隔独立服从均值 30 秒的指数分布。Target 为 Qwen3-14B。
输出上限仍为 400 token，计时从 Draft/本地执行端开始处理任务起，包含连接和输入准备。

`token_budget=64` 限制的是每次 Target 模型 forward 的新增输入位置总数。
bridge、批次内各请求的输入和 CUDA Graph 填充都计入；历史 KV 长度不计入。
预算是上限，不要求每次填满。同样的输入 token 上限不意味着相同 FLOPs、上下文长度或批次行数。

| 方法 | 预算执行方式 |
| --- | --- |
| Target-only | prompt 分片与多会话解码共用每次 forward 的 64 token 上限。 |
| Standard SD | 保留单设备调度，长 prefill/verify 分片到预算内；bridge 仅在第一片计入。 |
| FastSD | 原有多队列、WRR 和 softfill 共用 64 token 计划预算，包含 bridge。 |
| SpecEdge | 保留官方固定宽度 CUDA Graph。每行宽 9，图最多 `min(16, floor(64/9))=7` 行，每次图 forward 实际输入 63 个位置。长 prefill 按 64 分片，续写原位置与 KV。 |

SpecEdge 一次调度可先执行多个 prefill forward，再执行一次图 forward。
这里逐次模型 forward 限额，不把整个调度调用内的 token 加总当成一次 forward。
图中未激活行和 gamma4 行的填充仍计入 63；其额外 KV 使用独立 scratch 位置，不进入返回值或保留的真实 KV。
配置中的 `batch_size=16` 保留为设备上限，SpecEdge 的实际图行数另记为 7。
32 个 Draft 在每个设备最多一个未完成请求的约束下，pending 上限为 32；batch 上限 16 不能替代全队列上限。

## 源码与实验适配器

核心调度修复位于 `src/engine.py`；CLI 默认值为 64。
比较方法与测量适配器属于此前冻结的实验包，其基线身份和五个修改文件的前后 SHA256
见 `config1-budget64-adapters.json`，精确差异见 `config1-budget64-adapters.patch`。
在该基线 package 目录下可用 `git apply --check <patch>` 检查，再应用补丁。
新实验包另外冻结 `config.json`、完整文件 manifest 与 bundle ID，部署到四个独立目录。

四条推理路径统一记录每次 forward 的 `forward_input_tokens`、`token_budget` 和 `input_shape`。
一旦实际输入超过上限，先记录 `target_forward_budget_violation`，随后中止，不退回无预算路径。
模型加载、图捕获和加载后功能检查不属于正式性能测量区间。

## 验证边界

仓库现有 21 项测试、SpecEdge 6 项运行时测试和 6 项链布局测试通过。
CPU 随机小 Qwen3 功能检查覆盖 198/101 token 的 prompt、130 token 的 verify 分片、
跨分片拒绝回滚以及过长 bridge 拒绝；两种原生调度均不超过 64。
SpecEdge 使用官方 Qwen3 模型、固定 KV 引擎的 CPU eager 路径，prefill 为 64/64/64/5，
两次图形状输入均为 63；返回 greedy token 与 215 个保留 KV 位置对照原生模型一致。
Target-only 的真实会话循环在 CPU 检查中完成两条长 prompt 的各 14 token 生成，
每次输入不超过 64，输出对照一致；生产观测器对 65 个输入位置在进入 CUDA 前报错。
这些检查不构成 CUDA Graph 验收、14B 输出等价或新一轮性能结果。

旧 FastSD 数据只有 61/2,118 个计划的 token 成本超过 64。
因此减小预算可能留下更多下一批候选，但不能据此认定多队列或 KV 预取已经提速；
仍需在 GPU 测量中检查实际候选、加载、命中以及吞吐。
