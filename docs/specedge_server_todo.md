# SpecEdge 服务器侧待办

SpecEdge 是 FastSD 四组 runall 之外的独立基线。下面的步骤需要实验服务器操作者完成；Codex 不下载模型、数据或 Python 包，也不申请 SSH 权限。

## 1. Python 3.14 和依赖

在官方 SpecEdge 源码所在服务器创建 Python 3.14 环境。官方仓库如果带有 `uv.lock`，建议使用：

```bash
cd /path/to/specedge
uv venv --python 3.14
source .venv/bin/activate
uv sync
```

如果服务器没有 `uv`，由服务器管理员按官方 `pyproject.toml` 建立 Python 3.14 虚拟环境，再安装官方锁定依赖。通常需要 PyTorch、Transformers、gRPC、CUDA/NCCL、tokenizer 依赖和官方项目列出的其他包；具体版本以官方 `pyproject.toml`/`uv.lock` 为准。不要把 FastSD 的 offload、prefetch 或 TP wrapper 注入官方 SpecEdge。

## 2. 官方源码、模型和数据

先准备官方 SpecEdge 源码（建议固定论文对应提交），然后在 FastSD 工作树执行：

```bash
export SPECEDGE_OFFICIAL_SOURCE=/srv/specedge
export SPECEDGE_REPLACE_LINK=1
bash baselines/specedge/link_official.sh
```

用户需要自行下载或挂载以下模型：

- A5000 24GB：`Qwen3-8B` target 与 `Qwen3-0.6B` draft。
- A100 80GB：`Qwen3-32B` target 与 `Qwen3-1.7B` draft。

用户还要准备 HumanEval 164、GSM8K 1319、MGSM 的完整本地版本（先确认是 110 还是 250）、MT-Bench 单轮 80 或双轮 160。数据文件放在 `data/<dataset>.jsonl`，或通过 `DATA_PATH` 指定目录。运行脚本会转换为 SpecEdge 输入 JSONL，不会替用户下载数据。

## 3. SSH、主机和配置

设置：

```bash
export SPECEDGE_SSH_KEY=$HOME/.ssh/id_ed25519
export SPECEDGE_HOST=user@server
export SPECEDGE_SERVER_HOST=server
export SPECEDGE_CLIENT_HOST=edge
export SPECEDGE_TARGET_MODEL_8B=/models/Qwen3-8B
export SPECEDGE_DRAFT_MODEL_0_6B=/models/Qwen3-0.6B
export SPECEDGE_TARGET_MODEL_32B=/models/Qwen3-32B
export SPECEDGE_DRAFT_MODEL_1_7B=/models/Qwen3-1.7B
```

8B 配置是 `baselines/specedge/configs/final/specedge_qwen3_8b_0.6b.yaml`，目标 GPU 为单张 A5000 24GB；32B 配置是 `baselines/specedge/configs/final/specedge_qwen3_32b_1.7b.yaml`，目标 GPU 为单张 A100 80GB。配置中的 `official_tp: false` 和 `offload: false` 是刻意保留的能力边界。

官方 SpecEdge 没有 TP。把 32B 放到 2×A6000 时预期是 OOM 或 unsupported；请保存启动日志并在结果中标记 `oom`/`unsupported`，不要把两张卡写成 TP=2，也不要添加 offload 后冒充官方结果。

这条限制只针对 SpecEdge。FastSD 自己的 32B 入口另有显式 TP 开关：
`TARGET_TP_SIZE=2 TARGET_TP_GPUS=0,1`；未设置 TP=2 时，32B 会 fail-fast。

## 4. Smoke 和完整运行

先做不执行官方程序的转换/配置 smoke：

```bash
DATASET=humaneval DATA_PATH=/path/to/data \
  bash scripts/zjh_ab/run_specedge.sh
```

8B A5000：

```bash
SPECEDGE_CONFIG=$PWD/baselines/specedge/configs/final/specedge_qwen3_8b_0.6b.yaml \
SPECEDGE_MODE=official SPECEDGE_EXECUTE=1 \
DATASET=humaneval DATA_PATH=/path/to/data \
  bash scripts/zjh_ab/run_specedge.sh
```

32B A100 80GB：

```bash
SPECEDGE_CONFIG=$PWD/baselines/specedge/configs/final/specedge_qwen3_32b_1.7b.yaml \
SPECEDGE_MODE=official SPECEDGE_EXECUTE=1 \
DATASET=gsm8k DATA_PATH=/path/to/data \
  bash scripts/zjh_ab/run_specedge.sh
```

MGSM 和 MT-Bench：

```bash
DATASET=mgsm DATA_PATH=/path/to/data bash scripts/zjh_ab/run_specedge.sh
DATASET=mt_bench MT_BENCH_TURNS=2 DATA_PATH=/path/to/data bash scripts/zjh_ab/run_specedge.sh
```

脚本只负责转换数据、检查官方目录和调用官方入口。不同官方提交的参数名可能不同；服务器操作者需要按官方 README 调整 `batch_server.sh`/`client_host.sh` 的实际命令。

## 5. 结果收集

把官方 server/client 输出和可选的 `energy_metrics.json` 放到同一个目录，例如：

```text
runtime/specedge/<run>/raw/**/*.jsonl
runtime/specedge/<run>/raw/energy_metrics.json
```

统一收集：

```bash
python3 scripts/zjh_ab/collect_specedge.py \
  --input runtime/specedge/<run>/raw \
  --output runtime/specedge/<run>/summary.json \
  --dataset humaneval --expected-requests 164 \
  --wallclock-s 120
```

输出包含 `TTFT`、`TPOT`、`E2E`、`tok/s`、逐请求 SLO、`energy`，并带有 `mode`、`official_tp=false`、`offload=false`。CPU-adapted 路径必须用 `SPECEDGE_MODE=cpu-adapted`，结果名称为 `specedge_cpu_adapted`，不能冒充官方 GPU 基线。

## 6. 不能由 Codex 自动化的步骤

- 创建服务器 Python 3.14 环境并安装官方依赖。
- 下载/挂载 Qwen3 模型和四套数据集，确认 MGSM 110/250 与 MT-Bench 单/双轮版本。
- 提供 SSH key、host、远端绝对路径和 GPU 设备权限。
- 在 A5000/A100 上启动官方 SpecEdge、处理其版本差异和 CUDA OOM。
- 采集 NVML 能耗数据、确认官方日志字段和最终质量 judge。
- 对 2×A6000 32B 的 OOM/unsupported 结果作如实归档。
