# Qwen3-8B 云端 / Qwen3-0.6B 本地实验安排

## 已确认环境

- 云端服务器：`gpunode2-R8424-G12`
- GPU：3 张 RTX A6000 48GB
- 当前选择：物理 GPU 2；检查时约 48.5GB 空闲，GPU 0/1 正在高负载运行其他任务
- Cloud Target：`/home/hdd/zhangh/models/Qwen3-8B`，BF16，约 31GB 落盘
- Draft tokenizer：`/home/hdd/zhangh/models/Qwen3-0.6B`
- Cloud Conda 前缀：`/home/hdd/zhangh/envs/fastsd`
- pip/Hugging Face/PyTorch 缓存：`/home/hdd/zhangh/cache/`
- 安装临时目录：`/home/hdd/zhangh/tmp/fastsd`
- 实验输出：`/home/hdd/zhangh/workspace/fastsd/results`
- 公网入口：`http://39.102.209.27:1597`
- 校园服务器上游：`http://115.190.90.101:1597`

Qwen3-0.6B 与 Qwen3-8B 的 `tokenizer.json`、`tokenizer_config.json`、`vocab.json` 和 `merges.txt` 已逐项核对一致。

## Cloud 环境

当前 Conda 环境最初只包含 Python 3.10 和 pip。Cloud 最小运行依赖安装方式：

```bash
cd /home/hdd/zhangh/workspace/fastsd
bash scripts/setup_qwen3_cloud_env.sh
```

安装脚本遵循 `/home/hdd/zhangh/DIRECTORY_LAYOUT.md`：环境、缓存、临时文件和
实验输出分别写入上述目录，并将这些路径持久化为该 Conda 环境的变量。脚本可重复执行；
在最后输出 `FastSD cloud environment is ready` 前，不应启动正式实验。

Cloud 不加载 Draft 权重，只使用 0.6B 目录中的 tokenizer。Target 使用普通 BF16 Transformers 权重，不要求 AutoGPTQ。

## 本地 Edge 环境

本地记录的 GPU 是 RTX 3050 Laptop 4GB。Qwen3-0.6B BF16 权重约 1.5GB，建议先只运行一个 Draft 进程：

```bash
conda create -n fastsd-local python=3.10 pip -y
conda activate fastsd-local
python -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  transformers==4.57.6 accelerate==1.12.0 numpy==1.26.4 \
  requests==2.32.5 fastapi==0.128.0 uvicorn==0.40.0
```

将 Qwen3-0.6B 放到本地目录，并通过 `FASTSD_DRAFT_MODEL` 指定。Edge 已支持普通 BF16 和带 `quantize_config.json` 的 GPTQ checkpoint。

## 第一阶段：公网连通与单请求冒烟

云端服务器：

```bash
cd /home/hdd/zhangh/workspace/fastsd
CUDA_VISIBLE_DEVICES=2 /home/hdd/zhangh/envs/fastsd/bin/python -c \
  'import torch; print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())'
bash scripts/run_qwen3_8b_cloud_server.sh
```

本地检查：

```bash
curl http://39.102.209.27:1597/health
```

本地运行：

```bash
cd /path/to/fastsd
export FASTSD_DRAFT_MODEL=/path/to/Qwen3-0.6B
export FASTSD_PYTHON=/path/to/conda/env/bin/python
bash scripts/run_qwen3_0p6b_local_edge.sh
```

冒烟参数固定为 `num_drafts=1`、`max_tasks_per_draft=1`、`max_tokens=32`、`max_num_seqs=1`，先验证输出/token parity，不用于评价 batching 吞吐。

## 第二阶段：单客户端稳定性

Cloud 保持运行，本地增加任务数和生成长度：

```bash
bash scripts/run_qwen3_0p6b_local_edge.sh \
  --max_tasks_per_draft 10 \
  --max_tokens 128 \
  --exp_name qwen3_single_client_10tasks
```

关注 EOS、连续 Verify、KV cache 长度、`accept_rate` 和是否出现 `verify_exceeds_token_budget`。

## 第三阶段：真正的 batching 实验

一次只有一个活跃 Edge session 时，Cloud 不会形成多请求 batch。实验必须提供至少 2 个并发 Draft session，并令 `num_drafts >= max_num_seqs`。

当前本地 RTX 3050 4GB 不建议直接加载多份 0.6B BF16。可选方案：

1. 使用两台或更多 Edge 机器，每台运行一个 0.6B Draft；
2. 准备兼容的 0.6B GPTQ checkpoint，再逐步尝试 2/4 个本地 Draft 进程；
3. 后续实现共享单份 Draft 权重的多 session worker。

有足够 Draft 并发后，先重启 Cloud：

```bash
bash scripts/run_qwen3_8b_cloud_server.sh \
  --token_budget 256 \
  --max_num_seqs 4 \
  --exp_name qwen3_8b_cloud_batch4
```

Edge 示例：

```bash
bash scripts/run_qwen3_0p6b_local_edge.sh \
  --num_drafts 4 \
  --max_num_seqs 4 \
  --token_budget 256 \
  --max_tasks_per_draft 10 \
  --max_tokens 128 \
  --exp_name qwen3_batch4_budget256
```

建议矩阵：

| 阶段 | `num_drafts` | `max_num_seqs` | `token_budget` | `max_tokens` |
| --- | ---: | ---: | ---: | ---: |
| 冒烟 | 1 | 1 | 128 | 32 |
| 并发起步 | 2 | 2 | 128 | 128 |
| Batch 规模 | 4 | 1/2/4 | 256 | 128 |
| Budget 扫描 | 4 | 4 | 128/256/512 | 128 |

每组至少记录 `system_tok_per_s`、任务延迟 P50/P95、Prefill queue/service、Prefill chunk 数、Verify latency、接受率和失败/OOM 数量。

## 运行注意事项

- `CUDA_VISIBLE_DEVICES=2` 后，物理 GPU 2 在进程内会变成 `cuda:0`。
- 1597 当前未发现监听进程；启动前仍应再次检查端口。
- 公网 API 没有认证，只应在实验窗口内运行，并用安全组限制来源地址。
- 启动 Cloud 前再次运行 `nvidia-smi`，如果 GPU 2 已被占用，应停止或显式选择另一张有足够显存的 GPU。
- `token_budget` 是有效 token 准入预算，不是实际 padding/FLOPs 上限。
