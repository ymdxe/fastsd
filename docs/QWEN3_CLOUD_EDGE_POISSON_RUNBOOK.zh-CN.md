# Qwen3 云边 Poisson 对比实验运行手册

本手册对应 `main` 中的统一实验接口。它比较三种方法：FastSD、严格
FCFS 的 Vanilla，以及**固定提交**的 Official SpecEdge。所有实验均使用：

- Edge（node1）：`Qwen3-0.6B`，两个获授权的 A5000；
- Cloud（node2）：`Qwen3-8B`，一个获授权的 A6000；
- IB：node1 `10.66.0.4`，node2 `10.66.0.5`；
- FastSD HTTP：`10.66.0.5:1597`；SpecEdge gRPC：`10.66.0.5:18000`；
- 基准配置：[configs/experiments/mtbench_poisson.yaml](../configs/experiments/mtbench_poisson.yaml)。

不要复用 `run_id`，不要手动清理旧结果。所有新产物写到
`/home/hdd/zhangh/results/fastsd`，脚本会拒绝已有文件、已有目录、通配监听
地址和非 IB SpecEdge 地址。

## 0. 同步代码与资源准入

在 **node1** 与 **node2** 分别执行。任何 `git status --short` 输出、SHA
不一致、SpecEdge 未初始化或预检失败都是停止条件；不要使用 `reset`、`clean`
或停止他人进程。

```bash
cd /home/hdd/zhangh/workspace/fastsd
git status --short
git fetch origin --prune
git switch main
git pull --ff-only origin main
git submodule sync -- baselines/specedge/official
git submodule update --init baselines/specedge/official
git rev-parse HEAD
git -C baselines/specedge/official rev-parse HEAD
```

每轮先查看实时状态并在获得 GPU 独占授权后设置**实际物理编号**，示例中的
编号不是固定配置：

```bash
# node1
export EDGE_PHYSICAL_GPUS="0,1"
export RUN_ROOT=/home/hdd/zhangh/results/fastsd
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
df -h /home/hdd
bash scripts/experiments/preflight.sh \
  --role edge --physical-gpus "$EDGE_PHYSICAL_GPUS" \
  --run-root "$RUN_ROOT" --min-free-gib 40 \
  --peer-ip 10.66.0.5 --require-exclusive

# node2
export CLOUD_PHYSICAL_GPU="0"
export RUN_ROOT=/home/hdd/zhangh/results/fastsd
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
df -h /home/hdd
bash scripts/experiments/preflight.sh \
  --role cloud --physical-gpus "$CLOUD_PHYSICAL_GPU" \
  --run-root "$RUN_ROOT" --min-free-gib 80 \
  --bind-host 10.66.0.5 --ports 1597,18000 --require-exclusive
```

`preflight.sh` 只读取状态：校验工作树、固定子模块、空间、模型、CUDA 映射、
IB 路由/ping、端口和选中 GPU 的 compute process；它从不创建结果、删除文件或
终止进程。`CUDA_VISIBLE_DEVICES=<物理编号>` 会使进程内所选 GPU 映射为
`cuda:0`（node1 的第二张为 `cuda:1`）。

空间采用三层、且不由脚本擅自清理：node1 只保留 0.6B、edge 环境、代码和结果，
node2 保留 8B、cloud 环境和临时 cloud 结果；结果根固定为
`/home/hdd/zhangh/results/fastsd`。node1 少于 40 GiB 或 node2 少于 80 GiB
可用空间时预检会停止。当前 node1 如仍保有未用于 edge 的 8B 副本，应由拥有者在
确认路径后**手工**释放；本仓库命令绝不会删除模型、缓存或历史结果。

## 1. 预生成一次 Poisson workload

只在 **node1** 为每个 `(lambda, seed)` 生成一次。生成器使用独立
`random.Random(seed)`，指数间隔累积成绝对 `scheduled_offset_s`；它不在请求
完成后 sleep。因此三个方法复用相同 SHA 的输入轨迹和渲染后的 prompt 字节。

```bash
cd /home/hdd/zhangh/workspace/fastsd
source /home/hdd/zhangh/envs/fastsd/bin/activate
export RUN_ROOT=/home/hdd/zhangh/results/fastsd
export LAMBDA=<rps>
export SEED=<20260812|20260813|20260814>
export TRACE_ID="mtbench80-l${LAMBDA}-s${SEED}"

bash scripts/experiments/build_workload.sh \
  --dataset mt_bench --data data/mt_bench.jsonl \
  --tokenizer-model /home/hdd/zhangh/models/Qwen3-0.6B \
  --max-requests 80 --arrival-mode poisson \
  --request-rate-rps "$LAMBDA" --arrival-seed "$SEED" \
  --num-clients 2 --output "$RUN_ROOT/workloads/$TRACE_ID"
```

结果目录包含 `arrival_trace.jsonl`、`prompts.jsonl` 和 `manifest.json`。
Edge 会在启动前验证 trace/prompt SHA，拒绝重渲染或混入不同 workload。

## 1.5 Pilot 与 2 请求 smoke

先做 Vanilla 16 请求 closed-loop pilot。它没有 Poisson 到达轨迹，因而只用于得到
`mu0` 与低负载 P95，不进入最终 open-loop 矩阵；脚本仍保存 data SHA、静态
round-robin 数据分配、完整 argv、provenance 与请求指标。先按下一节的 node2 命令
启动 `--method vanilla` cloud（`RUN_ID=vanilla-pilot-...`），再在 node1 执行：

```bash
EDGE_PHYSICAL_GPUS="$EDGE_PHYSICAL_GPUS" SERVER_URL=http://10.66.0.5:1597 \
bash scripts/experiments/run_fastsd_replay.sh \
  --config configs/experiments/mtbench_poisson.yaml --method vanilla \
  --arrival-mode closed_loop --max-requests 16 --max-tokens 256 \
  --dataset mt_bench --data data/mt_bench.jsonl \
  --run-id "$RUN_ID" --run-dir "$RUN_ROOT/$RUN_ID/edge"
```

从该 pilot 的 `edge/metrics/summary.json` 读取 `system_req_per_s=mu0` 与
`task_e2e_ms_p95`。每个 Vanilla Poisson 候选完成并 finalize 后，用新文件保存容量
判定（候选依次传入，工具选出最高合格 offered rate）：

```bash
python scripts/experiments/derive_vanilla_capacity.py \
  --pilot-edge-run "$RUN_ROOT/vanilla-pilot-<id>/edge" \
  --candidate-run "$RUN_ROOT/vanilla-mtbench-<candidate-1>" "$RUN_ROOT/vanilla-mtbench-<candidate-2>" \
  --output "$RUN_ROOT/vanilla-capacity-<new-id>.json"
```

每个方法在正式 80 请求之前都须完成 2 请求、32 token smoke。为三种方法只生成一次
新的短 trace；FastSD/Vanilla edge 用 `--max-tokens 32 --max-requests 2`，SpecEdge
server 也加 `--max-tokens 32`，SpecEdge edge 加 `--max-requests 2`：

```bash
export SMOKE_TRACE_ID="mtbench2-smoke-s20260812"
bash scripts/experiments/build_workload.sh \
  --dataset mt_bench --data data/mt_bench.jsonl \
  --tokenizer-model /home/hdd/zhangh/models/Qwen3-0.6B \
  --max-requests 2 --arrival-mode poisson --request-rate-rps 1 \
  --arrival-seed 20260812 --num-clients 2 \
  --output "$RUN_ROOT/workloads/$SMOKE_TRACE_ID"
```

将该 trace 替换进第 2/3 节命令并使用全新 smoke `RUN_ID`；smoke 仍须通过
`finalize_run.sh` 与 `validate_run.sh`，否则不得开始正式负载。

## 2. FastSD 与 Vanilla

每种方法使用不同的全新 `RUN_ID`。先在 **node2 的独立终端** 前台启动 Cloud，
再在 **node1** 回放。Cloud 前台退出后再进行下一种方法；不要在一个 Cloud
服务上混跑两种 profile。

```bash
# node2 terminal — FastSD Cloud
cd /home/hdd/zhangh/workspace/fastsd
source /home/hdd/zhangh/envs/fastsd/bin/activate
export RUN_ROOT=/home/hdd/zhangh/results/fastsd
export RUN_ID="fastsd-mtbench-l${LAMBDA}-s${SEED}"
CLOUD_PHYSICAL_GPU="$CLOUD_PHYSICAL_GPU" \
bash scripts/experiments/start_fastsd_cloud.sh \
  --config configs/experiments/mtbench_poisson.yaml --method fastsd \
  --run-id "$RUN_ID" --run-dir "$RUN_ROOT/$RUN_ID/cloud" \
  --bind-host 10.66.0.5 --port 1597
```

```bash
# node1 terminal — FastSD Edge
cd /home/hdd/zhangh/workspace/fastsd
source /home/hdd/zhangh/envs/fastsd/bin/activate
curl --fail --retry 30 http://10.66.0.5:1597/health
EDGE_PHYSICAL_GPUS="$EDGE_PHYSICAL_GPUS" SERVER_URL=http://10.66.0.5:1597 \
bash scripts/experiments/run_fastsd_replay.sh \
  --config configs/experiments/mtbench_poisson.yaml --method fastsd \
  --trace "$RUN_ROOT/workloads/$TRACE_ID/arrival_trace.jsonl" \
  --run-id "$RUN_ID" --run-dir "$RUN_ROOT/$RUN_ID/edge"
```

Vanilla 完全复用 trace、模型和 GPU，仅将上述两个命令的 `RUN_ID` 改为
`vanilla-mtbench-l${LAMBDA}-s${SEED}`，并把 `--method fastsd` 改为
`--method vanilla`。该 profile 强制严格 FCFS、关闭 FastSD scheduler、
proactive draft 和 pipeline。

## 3. Official SpecEdge

Official tree 不被修改。服务适配器只将监听地址参数化为 IB 地址，并把统一配置
渲染为该 run 私有的 Official YAML；客户端适配器只负责 trace replay、prompt
输入和记录。对真实 Poisson 到达，它使用 Official 已支持的 `dynamic` batch 配置，
而不修改树状推测算法或 Official 源码；先准备专用 `specedge` 环境，并确认其依赖与 pinned Official tree
兼容，再运行：

Official 当前锁定 `Python 3.14`（`pyproject.toml` 与 `uv.lock` 均会被脚本核对）。
先在 **node1**、**node2** 分别执行下列命令一次；它是唯一允许创建
`/home/hdd/zhangh/envs/specedge` 的入口。脚本会在任何写入前拒绝：没有 `uv`、
没有显式绝对路径的 Python 3.14、Official 子模块未初始化/脏、锁文件不匹配、父目录
不可写或目标环境已存在。它使用 `uv sync --frozen --no-dev`，绝不升级锁文件、覆盖
已有环境或清理失败残留；若创建中断，请由环境拥有者检查明确路径，而不是重跑覆盖。

```bash
# node1 和 node2：先只检查命令可用性；不要把系统的 python3.10 传入
cd /home/hdd/zhangh/workspace/fastsd
command -v uv
command -v python3.14
python3.14 --version
test ! -e /home/hdd/zhangh/envs/specedge
git -C baselines/specedge/official status --short

# 确认以上均通过后，显式传入 Python 3.14 的绝对路径；此命令才会安装锁定依赖
bash scripts/experiments/setup_specedge_locked_env.sh \
  --python "$(command -v python3.14)" \
  --uv "$(command -v uv)"

# 创建成功后（两台机器各自执行）
/home/hdd/zhangh/envs/specedge/bin/python -c \
  'import sys, grpc, torch, transformers; print(sys.version)'
```

当前任一节点缺少 `uv` 或 `python3.14`，或者已有不明来源的 `specedge` 环境时，
这是明确停止条件，不能用 FastSD 的 Python 3.10 环境替代，也不能由实验脚本删除或
重建该目录。先由环境拥有者安装/指定 Python 3.14 与 uv，或确认旧环境的处置方式，
再从上述门禁重新开始。

```bash
# node2 terminal — Official SpecEdge server
cd /home/hdd/zhangh/workspace/fastsd
source /home/hdd/zhangh/envs/specedge/bin/activate
export RUN_ROOT=/home/hdd/zhangh/results/fastsd
export RUN_ID="specedge-mtbench-l${LAMBDA}-s${SEED}"
CLOUD_PHYSICAL_GPU="$CLOUD_PHYSICAL_GPU" \
bash scripts/experiments/start_specedge_server.sh \
  --config configs/experiments/mtbench_poisson.yaml \
  --run-id "$RUN_ID" --run-dir "$RUN_ROOT/$RUN_ID/cloud" \
  --bind-host 10.66.0.5 --port 18000
```

```bash
# node1 terminal — Official SpecEdge Edge replay
cd /home/hdd/zhangh/workspace/fastsd
source /home/hdd/zhangh/envs/specedge/bin/activate
EDGE_PHYSICAL_GPUS="$EDGE_PHYSICAL_GPUS" \
bash scripts/experiments/run_specedge_replay.sh \
  --config configs/experiments/mtbench_poisson.yaml \
  --trace "$RUN_ROOT/workloads/$TRACE_ID/arrival_trace.jsonl" \
  --grpc-address 10.66.0.5:18000 \
  --run-id "$RUN_ID" --run-dir "$RUN_ROOT/$RUN_ID/edge"
```

对于只验证 trace/元数据而不访问 GPU 或服务的 smoke，可在最后一条命令追加
`--dry-run`。正式运行不允许 dry-run，也不允许改为未知 factory 或端口。

## 4. 收集、校验与分析

在 Edge 结束、Cloud 正常退出后，在 **node1** 执行：

```bash
bash scripts/experiments/finalize_run.sh \
  --run-dir "$RUN_ROOT/$RUN_ID" --cloud-host node2 \
  --cloud-dir "$RUN_ROOT/$RUN_ID/cloud"

bash scripts/experiments/validate_run.sh --run-dir "$RUN_ROOT/$RUN_ID"
```

`finalize_run.sh` 使用 `rsync --ignore-existing` 与 `cp --no-clobber`，并验证
远端/本地 Cloud manifest SHA；不覆盖、不删除历史结果。矩阵完成后只用新输出目录：

```bash
bash scripts/experiments/analyze_matrix.sh \
  --run-root "$RUN_ROOT" --output "$RUN_ROOT/analysis-<new-id>" \
  --slo-e2e-ms <2x-pilot-P95-ms>
```

每个成功 run 至少包含：

```text
manifest.json                    config/resolved.yaml
workload/arrival_trace.jsonl     workload/manifest.json
provenance/node1.json            provenance/node2.json
logs/                            metrics/requests.jsonl
metrics/gpu_samples.csv          metrics/summary.json
outputs/completions.jsonl
```

`manifest.json` 保存完整 argv、root/SpecEdge SHA、模型文件清单、配置与 trace
SHA、物理/逻辑 GPU 映射、IB route/ping、白名单环境变量、起止时间、状态和退出码；
不保存 SSH 私钥、令牌或完整环境变量。

每个 component 还保留 `metrics/gpu_samples_raw.csv`（含启动/预热证据）。只有按 edge
实际 arrival/completion 或 cloud `Validate` RPC 边界裁剪后的 canonical
`metrics/gpu_samples.csv` 参与能耗、J/token 和最终分析；若边界没有被采样覆盖，run
失败而不是估算能耗。

## 5. 固定 32 条质量守门

性能 replay 完成后，分别以 **新的** `run_id` 对 HumanEval 和 GSM8K 的原始
JSONL 做 32 请求确定性 replay；不要把 80 条 MT-Bench 性能输出拿来评分。收集
完成后，质量工具只读取该 run 的 canonical
`outputs/completions.jsonl`，严格要求其恰有前 32 条数据各一条、无重复、无缺失，
并且拒绝 prompt-inclusive 文本或已有 `quality/` 目录。

先在 node1 为每个质量数据集生成一次固定 seed 的 32 请求 trace；同一 trace 必须被
FastSD、Vanilla、SpecEdge 三种方法共同使用。下面以 GSM8K 为例，HumanEval 只需把
`QUALITY_BENCHMARK` 和 `QUALITY_DATA` 分别替换为 `humaneval` 和
`data/humaneval.jsonl`：

```bash
# node1 — 只执行一次，输出目录必须全新
cd /home/hdd/zhangh/workspace/fastsd
export QUALITY_BENCHMARK=gsm8k
export QUALITY_DATA=data/gsm8k.jsonl
export QUALITY_TRACE_ID="${QUALITY_BENCHMARK}32-s20260812"
bash scripts/experiments/build_workload.sh \
  --dataset "$QUALITY_BENCHMARK" --data "$QUALITY_DATA" \
  --tokenizer-model /home/hdd/zhangh/models/Qwen3-0.6B \
  --max-requests 32 --arrival-mode poisson --request-rate-rps 1 \
  --arrival-seed 20260812 --num-clients 2 \
  --output "$RUN_ROOT/workloads/$QUALITY_TRACE_ID"
```

随后按第 2 节或第 3 节先启动对应的 cloud，并把 `RUN_ID` 改为全新的
`<method>-<benchmark>-quality32-...`。FastSD/Vanilla 在 node1 的 edge 命令如下；
`METHOD` 依次设为 `fastsd`、`vanilla`。SpecEdge 则使用第 3 节 edge 命令，替换 trace
并加 `--max-requests 32`。每次 replay 后都运行第 4 节的 `finalize_run.sh` 与
`validate_run.sh`，通过后才可以调用下方评分工具。

```bash
# node1 — FastSD 或 Vanilla quality replay
export METHOD=fastsd  # 第二次设为 vanilla；cloud 端也必须用同一 method
EDGE_PHYSICAL_GPUS="$EDGE_PHYSICAL_GPUS" SERVER_URL=http://10.66.0.5:1597 \
bash scripts/experiments/run_fastsd_replay.sh \
  --config configs/experiments/mtbench_poisson.yaml --method "$METHOD" \
  --trace "$RUN_ROOT/workloads/$QUALITY_TRACE_ID/arrival_trace.jsonl" \
  --max-requests 32 --max-tokens 256 \
  --dataset "$QUALITY_BENCHMARK" --data "$QUALITY_DATA" \
  --run-id "$RUN_ID" --run-dir "$RUN_ROOT/$RUN_ID/edge"

# node1 — Official SpecEdge quality replay（其 cloud 用第 3 节命令启动）
EDGE_PHYSICAL_GPUS="$EDGE_PHYSICAL_GPUS" \
bash scripts/experiments/run_specedge_replay.sh \
  --config configs/experiments/mtbench_poisson.yaml \
  --trace "$RUN_ROOT/workloads/$QUALITY_TRACE_ID/arrival_trace.jsonl" \
  --max-requests 32 --grpc-address 10.66.0.5:18000 \
  --run-id "$RUN_ID" --run-dir "$RUN_ROOT/$RUN_ID/edge"
```

```bash
# node1：两个变量分别是已完成的独立 32 请求质量 replay，均为全新 run_id
cd /home/hdd/zhangh/workspace/fastsd
export GSM8K_QUALITY_RUN="$RUN_ROOT/<new-gsm8k-quality-run-id>"
export HUMANEVAL_QUALITY_RUN="$RUN_ROOT/<new-humaneval-quality-run-id>"

python scripts/experiments/quality_guard.py \
  --run-dir "$GSM8K_QUALITY_RUN" --benchmark gsm8k \
  --dataset /home/hdd/zhangh/workspace/fastsd/data/gsm8k.jsonl

# HumanEval 生成供官方 evaluator 消费的 sample JSONL，但此命令绝不执行生成代码。
python scripts/experiments/quality_guard.py \
  --run-dir "$HUMANEVAL_QUALITY_RUN" --benchmark humaneval \
  --dataset /home/hdd/zhangh/workspace/fastsd/data/humaneval.jsonl
```

产物为 `quality/summary.json`、`quality/scored_completions.jsonl`、
`quality/manifest.json` 和 `quality/command_hints.txt`；HumanEval 另有只含
`task_id`/`completion` 的 `quality/humaneval_samples.jsonl`。GSM8K 从全文提取
最右侧有效数字、去千分位/前导正号、规范化零和小数后做精确字符串比较；完整规则
写入 summary/manifest。HumanEval 的 `command_hints.txt` 给出
`evaluate_functional_correctness` 命令，但只能在隔离、可丢弃且设定 timeout 的环境
中由实验者显式运行。

## 6. 矩阵与结果纳入规则

1. 先做 16 请求 closed-loop Vanilla pilot，得到 `mu0` 和低负载 P95 E2E。
2. 从 `0.5mu0` 开始倍增、二分，定义 Vanilla 容量 `mu`：100% 完成且 P95 E2E
   不高于 pilot 的两倍。
3. 正式负载为 `0.5mu`、`0.8mu`、`1.0mu`、`1.2mu`，每点 80 条 MT-Bench、3 个
   arrival seed；每个 seed 轮换三方法顺序。
4. 先做 2 请求 / 32 token smoke。再对固定 32 条 HumanEval 和 32 条 GSM8K
   进行确定性质量复跑；质量集不改变性能 trace。

主指标为完成率、系统 tok/s、req/s、SLO goodput、P50/P95/P99 TTFT/E2E/ITL、
edge/server queue、prefill/verify 时间、接受率、GPU 利用率/显存/功耗和 J/token。
测量窗口是首条**实际到达**至最后一条完成；服务启动、模型加载和 warmup 另记。
最后一个计划到达后最多排空 10 分钟。trace SHA 不同、GPU 非独占、端口冲突、
provenance 缺失、错误/超时/丢请求均标记失败，不能进入最终对比图表。

FastSD 与 Vanilla 会直接记录上述 token 级 TTFT/ITL、prefill/verify 与接受率。
Pinned Official SpecEdge 可公平比较完成率、到达 lag、edge queue、E2E、req/s、显式
token 计数/tok/s 与裁剪后能耗；它没有稳定公开的逐 token 时间戳或与 FastSD 同构的
server prefill/verify/acceptance 字段。`analyze_matrix.py` 对这些 SpecEdge 字段保留
缺失值（不是伪造为 0）；报告中必须标作 `N/A`，不要把它们宣称为三方法可比指标。
