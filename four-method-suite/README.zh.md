# Qwen3-8B 三方法手动实验

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: prepare + plan
- Verification Status: PREPARED；静态与无模型预检见 VALIDATION.md，真实模型和性能结果尚未产生。
- Date: 2026-10-26（Asia/Shanghai）

## 已固定的范围

比较 AR、Standard SD、FastSD。Target 为 gpunode2 上一份 Qwen3-8B BF16；node1 为八份 0.6B、每份八线程、γ4；node3 为八份 1.7B、每份八线程、γ8。每条流最多一个在服务请求，其余按预生成计划排队。node2 准备相同代码及状态查询入口，作为协调/备用节点；本矩阵不增加它的 Draft。

本包在独立目录保存 GPU 当前源码快照及本次测量改动；已有仓库代码和结果保留。`manifest.json` 固定全部可执行代码与输入，四端必须相同。包约 2.4 MB，不含模型或环境。GPU 原代码为 kvpre / ee0e334e222593c100b2ee6fc7fc9e4db2cf7cd3，含既有未提交修改；源码快照保留这些内容。

- GPU 包：`/home/hdd3/zhangh/workspace/fastsd/manual-compare-20261026`
- CPU 包：`/home/zhangh/workspace/fastsd/manual-compare-20261026`
- GPU 结果：`/home/hdd3/zhangh/workspace/fastsd/exp/manual_compare_20261026/<RUN_ID>`
- CPU 结果：`/home/zhangh/workspace/fastsd/exp/manual_compare_20261026/<RUN_ID>/<node>`

本机源包：`E:\Codex\artifacts\fastsd-prepare-20261026\manual-compare-20261026`。准备文件在 `E:\Codex\temp\fastsd-prepare-20261026`。

## 0. 连接与预检

SSH 直连端口 22 可达，之前的 19635 HTTP 直连未打通。统一使用既定 SSH 隧道；结果反映该网络条件，不视为 IB 直连性能。每台 CPU 在单独终端执行并保持窗口打开：

```bash
bash /home/zhangh/workspace/fastsd/manual-compare-20261026/launch.sh tunnel
```

node1 和 node3 必须建立。node2 若用于状态查询，也执行一次。按 SSH 提示登录；不要把密码或密钥发到聊天。已有 29635 隧道时不重复建立。

三台 CPU 的其他终端可执行：

```bash
bash /home/zhangh/workspace/fastsd/manual-compare-20261026/launch.sh check
```

GPU：

```bash
bash /home/hdd3/zhangh/workspace/fastsd/manual-compare-20261026/launch.sh check
```

`check` 检查代码清单与到达计划，不启动模型。包已用各机解释器做过预检时，不必为走流程重复执行。

所有 server/oracle 命令先显示 GPU 状态，再询问已分配给本实验的物理 GPU 编号。三方法必须使用同一张卡、相同 CPU 分配与相同隧道。启动前确认当前资源分配；脚本不停止已有服务。不要把 64 个计算线程解释为 64 个独占物理核。

## 1. 先运行输出差异诊断

在 gpunode2 执行：

```bash
bash /home/hdd3/zhangh/workspace/fastsd/manual-compare-20261026/launch.sh oracle
```

按提示填写 GPU。脚本使用同一份 80 题输入、历史输出以及本包算子快照，比较 HF 逐 token AR、varlen 单序列 AR 和历史 node1/node3 输出，保存分歧 logits。日志打印结果目录，形如：

```text
/home/hdd3/zhangh/workspace/fastsd/exp/manual_compare_20261026/oracle_.../results
```

查看这个目录的 `summary.json`、`requests.jsonl`。历史 19 题差异仍是待解释的旧证据。若 `hf_repeat` 或 `varlen_single` 与 HF 不一致，先分析此结果再继续；脚本未证明数值差异根因。诊断完成后模型进程自动退出。

## 2. 三方法各做单客户端链路检查

这里是 node1 和 node3 各一个客户端、各两题，共四条请求，先检查两种 γ、启动协议与状态释放。

先在 gpunode2 执行 AR：

```bash
bash /home/hdd3/zhangh/workspace/fastsd/manual-compare-20261026/launch.sh server ar single single_ar
```

保持 GPU 窗口打开。CPU 终端查询：

```bash
bash /home/zhangh/workspace/fastsd/manual-compare-20261026/launch.sh status
```

只有 health 同时显示 `"status": "ok"` 和 `"model_ready": true` 才启动客户端。在 node1、node3 各执行：

```bash
bash /home/zhangh/workspace/fastsd/manual-compare-20261026/launch.sh clients single_ar
```

两端可先后启动。每个模型/客户端完成加载和一条预热后自动报 READY；协调器在全部 READY 后设置未来十秒的共同 T0，不依靠切换终端的速度。预热不进入正式指标。

两端均完成后，在 GPU 另一终端查看：

```bash
bash /home/hdd3/zhangh/workspace/fastsd/manual-compare-20261026/launch.sh summary single_ar
```

应完成四条请求。检查 `cache-final.json` 的缓存和待处理数量为零。回到 GPU 服务原终端按 Ctrl+C 结束这一轮，再做另外两种方法：

| 方法 | GPU 命令参数 | 两台 CPU 命令参数 |
|---|---|---|
| Standard SD | `server sd single single_sd` | `clients single_sd` |
| FastSD | `server fastsd single single_fastsd` | `clients single_fastsd` |

命令完整形式仍为 `bash <对应节点包目录>/launch.sh <表中参数>`。每轮更换服务实例，运行期间只运行这一种方法；SSH 隧道保持。

## 3. 三方法各做 16 客户端、80 题正确性检查

每台八个客户端，每个十题；两端分别覆盖同一套 80 题，共 160 条。沿用上一节的启动、status、两台 clients、summary、Ctrl+C 顺序。

| 方法 | GPU 命令参数 | 两台 CPU 命令参数 |
|---|---|---|
| AR | `server ar parity parity_ar` | `clients parity_ar` |
| Standard SD | `server sd parity parity_sd` | `clients parity_sd` |
| FastSD | `server fastsd parity parity_fastsd` | `clients parity_fastsd` |

例如 FastSD 的完整命令：

```bash
# gpunode2
bash /home/hdd3/zhangh/workspace/fastsd/manual-compare-20261026/launch.sh server fastsd parity parity_fastsd

# node1 和 node3 各自执行；等 Target model_ready=true 后启动
bash /home/zhangh/workspace/fastsd/manual-compare-20261026/launch.sh clients parity_fastsd
```

每一轮应 `all_requests_completed=true`、160 条请求，并生成 `cache-final.json`。模型加载或 worker 报错时查看 CPU 结果目录的 `worker*/worker.log` 与 GPU `.console.log`；使用新 RUN_ID 重跑，原目录不会覆盖。某端失败会给协调器记录 aborted；其他端已进入生成的请求可能仍需返回或手动 Ctrl+C。

三方法都完成后，在 GPU 执行以下命令。将 `/完整的oracle结果目录/results` 换成第一步真实输出目录：

```bash
bash /home/hdd3/zhangh/workspace/fastsd/manual-compare-20261026/launch.sh gate \
  /完整的oracle结果目录/results parity_ar parity_sd parity_fastsd
```

这个检查要求 HF 重复和单序列算子一致、当前三个方法的全部 480 条输出与 HF 参考逐 token 一致、每轮状态释放干净，且代码包相同。通过后写入 `gate.json`，正式长跑入口才放行。任何不一致都会保留证据并停止放行，需要据此定位问题，不能先看性能再改变标准。

## 4. 60 秒试测

保持上述配置，以 mid_rep1 前 60 秒为同一计划，三方法各一轮：

| 方法 | GPU 命令参数 | 两台 CPU 命令参数 |
|---|---|---|
| AR | `server ar pilot pilot_ar` | `clients pilot_ar` |
| Standard SD | `server sd pilot pilot_sd` | `clients pilot_sd` |
| FastSD | `server fastsd pilot pilot_fastsd` | `clients pilot_fastsd` |

当前固定计划 38 条请求。即使某流在 60 秒内没有请求，它的 Draft 仍加载、预热、参与 READY，并上报空结果，保持 16 个客户端及资源配置。检查完成数量、无失败、缓存释放与 RSS/显存趋势，再运行完整矩阵。

## 5. 正式 36 轮矩阵

速率均为每个 Draft/逻辑客户端的独立泊松计划速率，γ 固定4/8。

| trace 前缀 | node1 每流 λ | node3 每流 λ | 到达窗口/数量 |
|---|---:|---:|---|
| low | 0.01 | 0.02 | 600 秒 |
| mid | 0.025 | 0.05 | 600 秒 |
| high | 0.05 | 0.10 | 600 秒 |
| overload | 5 | 10 | 每流10条，共160条 |

每组 rep1/rep2/rep3 使用不同种子；同一重复的三方法重放完全相同 trace。顺序依次为 AR→SD→FastSD、SD→FastSD→AR、FastSD→AR→SD。完整顺序和建议 RUN_ID 见 `RUNS.tsv`。

例如中负载第一次的 AR：

```bash
# gpunode2
bash /home/hdd3/zhangh/workspace/fastsd/manual-compare-20261026/launch.sh server ar mid_rep1 mid_rep1_ar

# node1 和 node3 分别执行
bash /home/zhangh/workspace/fastsd/manual-compare-20261026/launch.sh clients mid_rep1_ar

# 完成后，在 GPU 另一个终端查看
bash /home/hdd3/zhangh/workspace/fastsd/manual-compare-20261026/launch.sh summary mid_rep1_ar
```

再在 GPU 服务终端 Ctrl+C，按 RUNS.tsv 下一行替换 method/trace/RUN_ID。重复运行时给 RUN_ID 增加 `_r2` 等后缀，三端必须一致。

到达计划独立于服务完成，拥塞请求顺序排队，迟到计入端到端延迟；窗口结束后继续排空。若排空超过30分钟，由你在本轮终端 Ctrl+C，记录未完成数量；不要删除或把该轮当成功结果。

## 方法与计时口径

- AR：一份8B、最多16在服务请求、分块prefill与逐token decode连续批处理、单次HTTP请求流式输出。使用相同varlen算子和每轮CPU KV卸载政策；没有FastSD预取。CPU仅运行逻辑客户端，不加载Draft。
- Standard SD：`vanilla`，关闭pipeline/proactive/延迟优先级/prefill重叠；现有源码vanilla按FCFS逐请求验证，不擅自另加批处理优化。
- FastSD：开启上述优化，使用FastSD调度；关闭动态γ调整。
- 三者上限：batch16、token_budget512、greedy、BF16、max32、Qwen3关闭thinking、EOS包含在输出中。实际批形状属于各方法行为；这是本实现之间的系统比较，不声称对所有优化AR/SD实现最优。
- 每个完成请求释放Target会话状态；普通原项目代码的实验入口不受本包影响。
- 主指标：计划到达到首token/完成的P50/P95，窗口内完整请求完成数量与积压。另报开始处理后的延迟和排空吞吐。
- SD没有每个token的可见时间，因此不报告窗口token吞吐，不把各worker吞吐相加。窗口边界附近的归属受所保存的时钟估计不确定度影响。
- `/clock/sync` 估计本地单调钟与GPU时间的偏差，所有请求时长在同机单调钟上计算；barrier最大不确定度20ms，FastSD原计时协议限制保留10ms。超限需检查网络，不把等待时间截成零掩盖问题。
- 首轮只有32token和MT-Bench第一轮80题；反复使用提示词是重复负载，不增加独立语义样本。低/中/高为待实测档位。CPU环境版本差异保留并记录。

## 结果文件

GPU 每轮的 `config.json`、`target-flags.json`（SD）、`launch.json`、`commands.txt`、`coordinator.json`、`resources.jsonl`、`reports/*.json`、`cache-final.json`、`summary.json` 与相邻 `.console.log`。AR另保存 `ar-batches.jsonl`。CPU保存分片、实际命令、worker日志、barrier时钟校准和退出码。结果通过实验HTTP连接自动回传GPU，无须你手动复制日志。

node2 只需隧道和 `launch.sh status` 即可查看健康、READY名单和完成名单。它不参与原16 Draft的测量流，也不需要为本轮复制模型。