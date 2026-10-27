# 16 个异构 CPU Draft + Qwen3-8B：手动执行方案

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-25
- Verification Status: UNVERIFIED（方案与脚本逻辑已核对，未运行本轮模型实验）
- Version Label: hetero16-manual-v1

## 1. 参数与实验范围

用户已明确：泊松参数针对**每个 Draft**，不是整台机器的总到达率。

| 项目 | cpunode1 | cpunode3 |
|---|---|---|
| Draft 模型 | Qwen3-0.6B | Qwen3-1.7B |
| 实例数 | 8 | 8 |
| 每实例计算线程设置 | 8 | 8 |
| γ | 4 | 8 |
| 每实例独立泊松计划到达率 λ | 5 请求/秒 | 10 请求/秒 |
| 每实例平均计划间隔 1/λ | 0.2 秒 | 0.1 秒 |
| 单节点计划总到达率 | 40 请求/秒 | 80 请求/秒 |
| 精度 | CPU FP32 | CPU FP32 |

Target：gpunode2 单个 Qwen3-8B，BF16，独占/获准使用的 GPU。两端合计计划到达率 120 请求/秒。

本轮补充默认设置（建议的首轮负载验证配置，不是用户新指定的论文正式参数）：

- 沿用已跑通的 greedy、MT-Bench 第一轮 prompt、`max_tokens=32`、EOS 停止；关闭 pipeline、proactive draft、latency priority、能耗采集。
- 每个 Draft 10 条请求，每台 80 条，合计 160 条；最多输出 5120 token。两台各使用同一套 80 条题目，但节点内分给 8 个实例，各题只出现一次。
- 无预热，与上一轮功能检查一致。后续测正式性能需要补预热、扩大请求数、重复实验及共同测量窗口。
- Target 建议 `batch_size=16`（一次计划允许的设备上限）、`token_budget=512`；这两项不同于 Draft 数量，也不保证每次实际 batch 都为 16。
- `request_timeout=600` 秒是单次 HTTP 请求超时，不是整场实验时长上限。

**为何不能直接 `--num_drafts 8 --arrival_rate 5`：**当前源码 `edge/edge.py` 为整台生成一条泊松计划，随后按全局索引轮流分给各进程。传 5 表示整台合计 5；改传 40 也仍是全局计划轮流分片，不能等同于 8 条独立的 λ=5 泊松流。

配套 `run_drafts.py` 因此启动 **8 个独立命令，各自 `--num_drafts 1`**，分别使用独立的指数间隔序列、随机种子、输入文件和结果目录。node1 的 λ 均为 5，node3 均为 10。脚本只包装已有 CLI，不修改 FastSD 源码。

**到达语义：**每个实例在自己的模型加载/健康检查完成后开始计时；处理当前请求时不能同时开始下一条，迟到量记录为 `arrival_lag_ms`。5/10 是计划到达率，不能声称实际每秒已经开始处理 5/10 条。首轮每实例仅 10 条，计划到达窗口平均约 2 秒/1 秒，实际完成可能明显更久；这属于有限负载试跑，不是稳定运行速率测量。

不同实例及不同机器没有统一模型就绪屏障。启动命令接近不代表所有实例同时开始测量；本轮不报告跨机全局 active throughput。

## 2. 已核对与待检查

已在 GPU 机器只读核对：

- Target 主代码 `/home/hdd3/zhangh/workspace/fastsd`，HEAD `ee0e334e222593c100b2ee6fc7fc9e4db2cf7cd3`；有未提交修改，HEAD 不是完整代码版本凭证。
- Target 从每个请求读取 gamma，允许 node1 的 4 和 node3 的 8 共存；云端 `--gamma 8` 只是缺失值时的回退。
- 本机 Qwen3-8B、本机/共享 Qwen3-0.6B、共享 Qwen3-1.7B 的 tokenizer 词表映射、BOS/EOS/PAD 与聊天模板相同；模型配置 vocab_size 均为 151936。
- 共享 1.7B 两个权重分片及索引存在；未执行 1.7B 实际推理。
- 旧 immediate 测试的 active_window 异常已从代码解释：非泊松路径把 actual_arrival_s 记为 0、completion_s 记为单请求 E2E，所以汇总取成最长请求。本方案用 poisson 路径，可保留每个独立实例内的累计时间；8 个实例之间仍不能直接合并这些相对时间原点。

GPU 端参考哈希（CPU 端如不同，先核对差异，不覆盖）：

```text
566d176e043d8b66e019a0256598fb5ac86fae8ff2ea20313d29c4fb8a3b3b26  edge/edge.py
22d613734e1e6ce3090dc2b490753da45d71d45af9f7e698e90c5bd755f2ed5d  src/util.py
aa4bb2b0aa2fec2fa4285f5aee4ebc1c91a94393866690014e8e55054bf42631  src/arrival.py
ce0f142e23e5b8bfaf2ab94c5075cd8778c0da339b08ce2952c9c2185aa6ad73  src/runtime.py
```

资源已核对：node1 为 80 逻辑 CPU / 40 物理核，node3 为 192 逻辑 CPU / 96 物理核。两台均按用户指定设置 8×8 计算线程；node1 会使用超线程，64 线程不代表 64 物理核。两台 PyTorch 分别为 2.6.0+cpu / 2.2.1+cpu，Transformers 均为 4.57.6，保留现有环境。

node1 的 CRLF 换行差异已核对；除 edge.py 多两个时间戳指标外，其余四个文件规范化换行后与 GPU/node3 一致。两条额外指标保留，没有覆盖已有源码。

按 checkpoint 全部已存储张量展开到 FP32、不计权重共享的估算，八份模型分别约 22.4/60.6 GiB；这不是实际 RSS，加载、KV 和激活会增加占用。

## 3. 配置由助手完成

辅助目录：GPU `/home/hdd3/zhangh/workspace/fastsd/manual-hetero16-20261025`；CPU `/home/zhangh/workspace/fastsd/manual-hetero16-20261025`。

- GPU 使用 `start_target.sh`。
- 两台 CPU 使用 `start_drafts.sh` 和 `run_drafts.py`。
- 脚本使用核实过的绝对 Python 路径，不要求手动激活环境或复制文件。
- 只部署与检查脚本；启动模型由用户执行。
- `check --skip-health-check` 可在 Target 未启动时只检查环境；正常 `run` 强制检查 HTTP。
- `prepare` 会创建结果目录，不能先用正式 run-id 做 prepare 再 run。

## 4. 手动启动 Target

在 gpunode2 执行，按提示输入本次分配到的物理 GPU 编号：

```bash
bash /home/hdd3/zhangh/workspace/fastsd/manual-hetero16-20261025/start_target.sh
```

脚本显示 GPU 状态，设置物理/逻辑卡映射，绑定 `219.216.65.42:19635`，保存实际命令、源码状态、GPU 状态与日志。端口已被占用时退出，不自动停止其他服务。保持终端打开，等待模型权重加载完成后再启动 CPU。

Target 使用本地 Qwen3-8B、BF16、batch_size=16、token_budget=512；请求自身的 gamma 决定每个 Draft 的验证长度。

## 6. 第四步：两条隧道与客户端检查

每台 CPU 保持一条原来的 SSH 隧道即可；如果仍可用，不重复建立占用相同端口的新隧道。没有时，在各自终端 A 执行：

```bash
ssh -N -o ConnectTimeout=10 -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L 127.0.0.1:29635:219.216.65.42:19635 \
  zhangh@219.216.65.42
```

各自终端 B 执行：

```bash
curl --noproxy '*' --connect-timeout 5 --max-time 15 http://127.0.0.1:29635/health
```

## 7. 手动启动两台 CPU 的 Draft

Target 权重加载完成、两台 CPU 的 health 返回正常后，在 node1 执行：

```bash
bash /home/zhangh/workspace/fastsd/manual-hetero16-20261025/start_drafts.sh hetero16_r1
```

随后立即在 node3 执行相同命令：

```bash
bash /home/zhangh/workspace/fastsd/manual-hetero16-20261025/start_drafts.sh hetero16_r1
```

脚本自动识别节点并选择环境。两台没有统一模型就绪屏障，实际开始时间取决于各模型加载速度。

脚本自动做的事情：

1. 生成本机 8 份 canonical JSONL，每份 10 条，各自包含独立的计划到达时刻。
2. node1 使用种子 11000–11007；node3 使用 31000–31007，避免两台共用相同随机序列。
3. 启动 8 个已有 `edge.py` 命令，每个 `--num_drafts 1 --edge_threads 8`；node1 为 0.6B/γ4/λ5，node3 为 1.7B/γ8/λ10。
4. 保存实际命令、代码和数据哈希、进程 PID、各 worker 控制台日志、逐请求指标及退出码；每 30 秒显示仍在运行的 worker。
5. 完成后生成 `node_summary.json`，包含接受率、服务延迟、到达迟到量，以及 `arrival_lag + request_e2e` / `arrival_lag + ttft` 的计划到达口径延迟。

每个独立命令内部的日志可能都显示 `[Edge 0]`，这是正常的；跨命令用 `worker0..7` 目录和包含节点/worker 的 sample_id 区分。不能因为都显示 Edge 0 就以为只启动一个模型。

原生 edge 入口还会启动子进程，所以 OS 的 Python 进程数不等于模型实例数。以 8 个 worker 日志、实际模型加载以及 8 份结果文件核对。

## 8. 监控、完成与停止

各自另开终端监控（替换 node1 / node3）：

```bash
free -h
ps -u "$USER" -o pid,ppid,pcpu,pmem,rss,etime,args | grep '[e]dge/edge.py'
tail -n 30 /home/zhangh/workspace/fastsd/exp/hetero16_r1/node1/logs/worker0.log
```

GPU 端可运行 `nvidia-smi` 并查看 Target 控制台。若长时间没有新请求/完成输出，先检查 8 份日志、进程状态、内存和 Target，尤其是 OOM、HTTP 超时、权重加载失败。不要自动重试掩盖失败。

每个 worker 的 HTTP timeout 为 600 秒；模型加载没有自动超时，启动器每 30 秒只报告进程存活，不能把存活当作进度。持续无进度时人工判断后停止。

完成标准：两台各 8 个 worker 全部退出码为 0，每个 10 条，单台共 80 条、全系统 160 条；sample_id 无重复，输出 token 不超过 32，日志无 traceback。未达标保留结果并排错。

结果路径：

```text
/home/zhangh/workspace/fastsd/exp/hetero16_r1/node1/   # 在 node1
/home/zhangh/workspace/fastsd/exp/hetero16_r1/node3/   # 在 node3
  manifest.json           # 参数、数据和代码哈希、八条实际命令
  commands.txt            # 实际执行命令
  preflight.json          # 启动前环境快照
  inputs/worker0.jsonl ... worker7.jsonl
  logs/worker0.log ... worker7.log
  worker0/edge_metrics_proc0.jsonl ... worker7/edge_metrics_proc0.jsonl
  worker0/edge_metrics_summary.json ... worker7/edge_metrics_summary.json
  processes.json
  exit_status.json
  node_summary.json
```

查看节点汇总：

```bash
# node1
cat /home/zhangh/workspace/fastsd/exp/hetero16_r1/node1/node_summary.json
# node3
cat /home/zhangh/workspace/fastsd/exp/hetero16_r1/node3/node_summary.json
```

需要中止时，在各自运行 `run_drafts.py run` 的终端按 Ctrl+C；脚本只清理自己创建的 worker 进程组，包括其子进程，不停止 Target 或其他实验。若异常退出留下进程，依据 `processes.json` 逐个核实，不批量 pkill。正常结束后再停止本轮 Target，最后停止两条隧道。

重跑改成新 run-id，例如 `hetero16_r2`，两台同时改；已有目录会被拒绝。保持同一组到达计划时 `seed-offset` 保持 0；需要新的独立重复时两台一致增加，如 100。先不要改请求数、生成长度或开关，以免混淆变化来源。

## 9. 结论边界与下一轮

- 本轮同时改变模型、γ、到达率且绑定不同机器，不能单独归因“1.7B 比 0.6B 更快/更慢”或某调度机制更好。
- 到达计划很可能超过 CPU 服务能力。重点查看每 worker 的 `arrival_lag_ms`、计划到达口径 TTFT/E2E、失败率；不能只看服务开始后的延迟。
- worker 相对时间原点独立，本方案不把各自吞吐相加来报告全局吞吐。
- 不把 SSH 隧道的网络性能当作 IB 直连性能。
- 资源、代码差异和启动参数已经核对；1.7B 实际推理、16 实例并发效果仍待本轮运行验证。
