# SpecEdge baseline 复现说明

本目录提供一个可追溯、可比较的 SpecEdge baseline：核心推理代码保持作者官方实现不变，外层只补充论文主实验配置、环境检查、draft-depth 校准、指标标准化和公平性检查。

## 1. 复现边界

| 项目 | 本目录采用的实现 |
|---|---|
| SpecEdge 算法 | 作者官方仓库固定提交 `1edcaf02ffc41a7b57726450c5357ed216a3b9bc` |
| Tree drafting | 官方 SpecExec tree 实现 |
| Proactive edge drafting | 官方单 expansion-head 扩展与 complete-alignment 复用 |
| Pipeline-aware scheduling | 官方 gRPC 多客户端 batch server |
| Server-only baseline | 官方同仓库的 tree-based speculative decoding |
| 评测配置 | 本目录 `configs/` 中按论文附录校正后的配置 |
| 指标 | 官方公式兼容的 JSON 规范化结果，加配置一致性检查 |

这里没有把 FastSD 的调度器包装成 SpecEdge，也没有重新实现一个近似版本。后续对比时，`official/` 是 baseline 算法，`tools/repro.py` 只是复现与统计控制层。

## 2. 论文机制到代码的映射

| 论文机制 | 官方代码位置 |
|---|---|
| Draft tree 构建、裁剪、验证后更新 | `official/src/specedge/client/specexec.py` |
| 单个最高累计概率 expansion head | `official/src/specedge/client/proactive.py` |
| Complete draft alignment 与 KV/tree 复用 | `official/src/specedge/client/specexec.py` |
| 多 edge 请求的 pipeline batching | `official/src/strategy/server_verify/specexec/grpc.py` |
| 异长请求 attention mask 与 KV padding | `official/src/strategy/server_verify/specexec/grpc.py`、`official/src/specedge/engine/graph.py` |
| gRPC token/tree 传输 | `official/src/specedge/network/grpc.py`、`official/specedge.proto` |
| 官方指标公式 | `official/src/metric/specedge.py`、`official/src/metric/server_only.py` |

## 3. 主实验口径

论文主实验使用：

- Server：A100 40GB；Qwen3-32B 使用 A100 80GB。
- Edge：RTX 4090；每个 server batch slot 对应两个 edge client。
- 实测 edge-cloud RTT：14.07 ms。
- Target/draft：Qwen3-14B/1.7B、Qwen3-14B/0.6B、Qwen3-32B/1.7B。
- 数据：SpecBench 六类任务。
- Temperature：0.7。
- Draft tree budget：32。
- 每个请求最多生成 256 个 output token。
- 模型为原始 checkpoint，不额外微调。
- Server-only 的 draft depth 通过穷举选择；SpecEdge 按
  `verify_time ~= draft_depth * draft_forward_time + RTT` 校准。

本目录配置修复了上游示例与论文之间两个重要差异：上游示例的 `max_new_tokens` 是 64，而论文为 256；上游 server-only 示例的 `max_budget` 是 64，而论文主实验为 32。

## 4. 运行环境

完整 GPU 实验应在 Linux/CUDA 环境运行。作者公开环境为 Ubuntu、NVIDIA driver 580；官方项目要求 Python `~=3.14.0`、PyTorch `~=2.9.0`。当前 Windows 工作区只适合静态检查和指标工具单测，不能据此声称复现了论文性能。

在 server 和所有 edge 节点上，将整个仓库放到相同绝对路径。进入官方实现并安装锁定依赖：

```bash
cd baselines/specedge/official
uv sync --frozen
```

先编辑目标配置中的：

- `base.ssh_key`
- `client.host`
- `node.edge-a`、`node.edge-b` SSH alias
- 各节点 `device`

然后运行环境检查：

```bash
python ../tools/repro.py doctor \
  --method specedge \
  --config ../configs/qwen3-14b-1p7b-specedge.yaml \
  --output ../results/doctor-specedge.json
```

只有 `ready: true` 后再开始正式实验。`tc` 只被列为可选项；若不是实际 WAN，应自行记录网卡、带宽、已有基础 RTT 和 netem 规则，不能只保留默认的 `127.0.0.1`。

## 5. 校准 draft depth

先用配置中的 pilot depth 跑一个小实验，记录真实 RTT，再计算 SpecEdge 推荐深度：

```bash
python ../tools/repro.py recommend-depth \
  --data result/paper/specedge-qwen3-14b-1p7b \
  --rtt-ms 14.07 \
  --output ../results/qwen3-14b-1p7b-depth.json
```

将输出中的 `recommended_max_beam_len` 写回正式配置并重新运行。论文给出的一个可核对点是：Qwen3-32B/1.7B 在约 15 ms RTT 下，server verify 约 94.2 ms、单次 draft forward 约 11 ms，对应 depth 7。

Server-only 不能直接复用这个深度。按论文做法生成 2-8 的穷举配置：

```bash
python ../tools/repro.py prepare-sweep \
  --config ../configs/qwen3-14b-1p7b-server-only.yaml \
  --depths 2,3,4,5,6,7,8 \
  --output-dir ../generated/server-only-14b-1p7b
```

逐个运行后，以相同质量约束下的最佳 server throughput/ITL 点作为 server-only baseline，并保留所有 sweep 原始日志，避免只报告最佳结果而丢失选择过程。

## 6. 启动实验

### SpecEdge

Server 节点：

```bash
cd baselines/specedge/official
./script/batch_server.sh -f ../configs/qwen3-14b-1p7b-specedge.yaml
```

能够 SSH 到全部 edge 节点的 client host：

```bash
cd baselines/specedge/official
./script/client_host.sh -f ../configs/qwen3-14b-1p7b-specedge.yaml
```

### Server-only

```bash
cd baselines/specedge/official
./script/server_only.sh -f ../configs/qwen3-14b-1p7b-server-only.yaml
```

每次运行必须使用新的 `base.exp_name`，并同时保存 server、所有 client 日志、配置副本、doctor manifest 和 RTT 测量结果。

## 7. 规范化和比较

SpecEdge：

```bash
python ../tools/repro.py normalize \
  --method specedge \
  --data result/paper/specedge-qwen3-14b-1p7b \
  --config ../configs/qwen3-14b-1p7b-specedge.yaml \
  --gpu A100-40 \
  --rtt-ms 14.07 \
  --output ../results/specedge-qwen3-14b-1p7b.json
```

Server-only：

```bash
python ../tools/repro.py normalize \
  --method server_only \
  --data result/paper/server-only-qwen3-14b-1p7b \
  --config ../configs/qwen3-14b-1p7b-server-only.yaml \
  --gpu A100-40 \
  --output ../results/server-only-qwen3-14b-1p7b.json
```

比较：

```bash
python ../tools/repro.py compare \
  --specedge ../results/specedge-qwen3-14b-1p7b.json \
  --server-only ../results/server-only-qwen3-14b-1p7b.json \
  --output-json ../results/compare-qwen3-14b-1p7b.json \
  --output-md ../results/compare-qwen3-14b-1p7b.md
```

比较命令会阻止 target/draft、数据集、temperature、batch size、tree budget、max tokens、seed 或 dtype 不一致的结果进入同一张表。

## 8. 论文结果锚点

这些数字是结果核对点，不是任何机器都必须逐位相同的硬断言：

| 指标 | 论文报告 |
|---|---:|
| 平均 server throughput gain | 2.22x |
| 平均 cost-efficiency gain | 1.91x |
| 相对 server-only speculative decoding 的平均 ITL 降低 | 11.24% |
| Proactive drafting 带来的 tokens/verification 平均提高 | 13.21% |
| 14B/1.7B 完整 SpecEdge component 结果 | 67.89 tok/s |

不同 GPU 时钟、driver、网络抖动、Hugging Face checkpoint revision、请求采样和 draft-depth 选择都会造成偏差。正式报告至少应给出三次独立运行的均值、标准差及原始结果目录。

## 9. 已知限制

- 上游默认 `client.host=127.0.0.1` 不能代表论文的 14.07 ms WAN。公开仓库目前也有一个关于网络条件记录不足的 open issue，因此本复现强制单独记录 RTT。
- 上游代码把 draft depth 作为静态配置读取；论文描述的“动态校准”需要本目录的 pilot/calibration 流程在正式运行前完成。
- 上游 client timing 将网络等待与 server wait 合并，无法从单一字段精确分离网络时间；规范化结果只把外部实测 RTT作为元数据记录。
- `cache_prefill: true` 是 benchmark 优化，会预先构建数据集 prompt 的 server KV cache；在线请求实验应改为 `false`，且不能与主表结果混用。
- `official/pyproject.toml` 标为 MIT，但实际 `official/LICENSE` 是非商业研究/评测/教学许可。应以 LICENSE 的更具体条款为准，不得直接商用部署。

## 10. 测试

指标标准化和深度公式可在不加载模型的机器上测试：

```bash
python -m unittest tests.test_specedge_repro -v
```

该测试不等于 GPU 端到端复现；它只验证复现控制层的计算与错误检查。
