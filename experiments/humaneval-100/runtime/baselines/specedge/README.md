# SpecEdge 官方基线复现

本目录把论文 *SpecEdge: Scalable Edge-Assisted Serving Framework for Interactive LLMs*
的官方实现作为 FastSD 的独立基线接入。官方源码位于 `official/`，以 Git submodule
固定到：

```text
https://github.com/kaist-ina/specedge.git
1edcaf02ffc41a7b57726450c5357ed216a3b9bc
```

这里保留两条清晰边界：

- `official/` 是论文作者发布的实现，不在 FastSD 中复制或改写其核心算法。
- 本目录其余文件只提供版本检查、论文参数配置和复现说明，不把 FastSD 内部推理路径称为 SpecEdge 官方结果。

面向当前 FastSD 数据集和四方法对比的运行时适配器位于 `integration/`。它负责读取
统一 canonical JSONL、按多个边缘 client 分片、复现泊松到达、精确记录 TTFT/TPOT 和
导出生成文本；官方 `official/` 子模块保持不变。完整流程见
[`docs/FOUR_METHOD_EVALUATION.md`](../../docs/FOUR_METHOD_EVALUATION.md)。

当 draft device 是 CPU 时，`integration/cpu_adapter.py` 提供
`CPUCompatibleSpecEdgeEngine` 和 CPU-safe timing boundary；它复用官方模型、KV cache、
Tree、SpecExec 和 proactive draft 调用协议，不构造一次性 `/generate` worker。网络侧的
`integration/wire_codec.py` 显式负责 bf16 mask 的 uint16 raw-byte cast，目标端仍使用官方
gRPC 服务。该路径的结果名称固定为 `specedge_cpu_adapted`，不应写成 untouched official
SpecEdge。

## 当前 Windows 本地状态

Windows 本地可以完成源码获取、版本固定、配置检查和 Python 语法编译；不能据此声称
论文端到端系统或性能已经复现。官方启动路径使用 Linux Bash、SSH、CUDA Graph、gRPC
以及多 GPU：服务端是 A100 40GB（32B 模型使用 A100 80GB），边缘侧按并发请求数配置
`batch size x 2` 张消费级 GPU，论文主实验使用 RTX 4090，平均网络 RTT 为 14.07 ms。

先运行本地检查：

```powershell
python baselines/specedge/repro.py doctor
python baselines/specedge/repro.py paper-matrix
python baselines/specedge/repro.py recommend-depth --verify-ms 94.2 --draft-ms 11 --rtt-ms 15
python -m unittest tests.test_specedge_repro -v
python scripts/run_tests.py
```

`doctor` 在 Windows 上会给出运行时警告，但只要官方源码、固定提交和六份配置正确，
本地集成检查应通过。统一入口在 FastSD 的 Python 3.10 环境中会跳过官方源码编译，
因为官方源码使用 Python 3.14 语法；GitHub CI 使用
`python scripts/run_tests.py --strict-official` 在 Python 3.14 下执行严格检查。
`doctor --strict-runtime` 用于完整 Linux GPU 运行环境，会把缺少的运行时条件视为失败。

## 获取子模块

新克隆 FastSD 时：

```bash
git clone --recurse-submodules <fastsd-repository-url>
```

已有克隆中：

```bash
git submodule update --init --recursive
```

## 论文对齐配置

`configs/` 包含三组主实验模型对，每组同时提供 SpecEdge 和 server-only tree speculative
decoding 配置：

| Target / Draft | SpecEdge | Server-only |
|---|---|---|
| Qwen3-14B / Qwen3-1.7B | `specedge_qwen3_14b_1.7b.yaml` | `server_only_qwen3_14b_1.7b.yaml` |
| Qwen3-14B / Qwen3-0.6B | `specedge_qwen3_14b_0.6b.yaml` | `server_only_qwen3_14b_0.6b.yaml` |
| Qwen3-32B / Qwen3-1.7B | `specedge_qwen3_32b_1.7b.yaml` | `server_only_qwen3_32b_1.7b.yaml` |

配置固定了论文明确给出的主实验条件：SpecBench、temperature 0.7、batch size 1、
tree budget 32、每个请求最多生成 256 个 token。`max_beam_len` 的默认值沿用官方示例为 4；
论文要求 server-only 通过穷举选择最优深度，而 SpecEdge 根据下式按实测延迟校准：

```text
server verification time ≈ edge drafting time + network RTT
```

可用 `recommend-depth` 计算初始候选值，然后在真实硬件上做邻近值 sweep。论文给出的
32B/1.7B 示例中，verification=94.2 ms、draft forward=11 ms 时，RTT 为 15/40/50 ms
对应深度约为 7/5/4。

配置中的 `server-node`、`edge-node` 和 SSH key 是占位值，部署前必须替换；官方
`client_host.sh` 还要求所有边缘节点上的仓库绝对路径完全相同。

## Linux GPU 正式运行

以下命令应在各 Linux 节点的同一绝对路径执行。官方项目声明 Python 3.14，并使用
`uv.lock` 固定依赖：

```bash
cd baselines/specedge/official
uv sync
```

把所选配置复制到官方目录并按实际主机名、SSH key、GPU 编号修改。例如：

```bash
cp ../configs/specedge_qwen3_14b_1.7b.yaml config/fastsd_specedge.yaml
```

服务端：

```bash
./script/batch_server.sh -f config/fastsd_specedge.yaml
```

边缘控制节点：

```bash
./script/client_host.sh -f config/fastsd_specedge.yaml
```

server-only 对照：

```bash
cp ../configs/server_only_qwen3_14b_1.7b.yaml config/fastsd_server_only.yaml
./script/server_only.sh -f config/fastsd_server_only.yaml
```

收集服务端和边缘端 JSONL 到同一个实验目录后：

```bash
. .venv/bin/activate
python src/metric/specedge.py -d result/paper/specedge_qwen3_14b_1.7b --gpu A100-40
python src/metric/server_only.py -d result/paper/server_only_qwen3_14b_1.7b --gpu A100-40
```

32B target 对应 A100 80GB，指标命令的 GPU 参数改为 `A100-80`。

## 许可证

官方 `pyproject.toml` 写有 `license = "MIT"`，但仓库根目录 `LICENSE` 实际限制为研究、
评估和教育用途，商业部署需要另行许可。集成保留官方子模块及其许可证；使用时应以
`official/LICENSE` 的具体条款为准。
