# 异构 CPU Draft + Llama-2-7B Target 运行方案 (模型异构 + Token 数量异构)

> 历史方案：本文的 `pre-kvcache` 分支和 `padded` 命令不对应当前无 padding 的 Llama 主流程。当前代码与命令见 [README](../README.md) 和 [Llama 运行命令](../scripts/command.md)；文中服务器路径、GPU 状态和原始 Llama checkpoint 的可用性需重新核实。

本文档记录经实地核查 node2 与 node3 环境后的完整异构运行方案。
包含**模型异构（Model Heterogeneity）**与**投机生成 Token 数量异构（Gamma Heterogeneity）**。

---

## 1. 运行拓扑与异构配置

| 节点 | 资源分配 | 模型配置 | 投机步数 ($\gamma$) | 角色定位 |
| --- | --- | --- | --- | --- |
| **node3** | CPU `56-63` (8核8线程, NUMA 2) | `TinyLlama-1.1B-Chat-v1.0` | **$\gamma = 4$** | 轻量级对话模型，推测短序列 |
| **node3** | CPU `64-71` (8核8线程, NUMA 2) | `TinyLlama-1.1B-1T-OpenOrca` | **$\gamma = 6$** | 指令增强模型，中等推测深度 |
| **node3** | CPU `80-95` (16核16线程, NUMA 3) | `TinyLlama-1.1B-python-v0.1` | **$\gamma = 8$** | 算力加倍的代码特化模型，深度长序列推测 |
| **node2** | GPU 2 (`CUDA_VISIBLE_DEVICES=2`) | `Llama-2-7B` Target | - | 云端大模型批处理验证 (Padded Batching) |

### 节点实机核查确认：
1. **工作目录 (Workspace)**：
   - node2 路径：`/home/hdd/zhangh/workspace/new_fastsd`
   - node3 路径：`/home/zhangh/workspace/new_fastsd-pre-kvcache-20260818`
2. **GPU 显存 (node2)**：
   - GPU 2 空闲显存为 **42.8 GB**（总共 49 GB），运行 bfloat16 Llama-2-7B（~13.5 GB）非常充裕。
3. **网络连通性 (node3 $\to$ node2)**：
   - 内部高速网络 IP 为 `10.66.0.5`，实测往返 Ping 时延仅 **0.17 ms**。
4. **模型文件**：
   - node3 上的三个 TinyLlama 模型全量保存在 `/home/zhangh/models/`，词表均为 LLaMA 32000 兼容。
   - node2 上现有 `/home/hdd/zhangh/models/Llama-2-7b` 为 Meta 原始 `.pth` 格式，首次执行前需按 Step 0 转换为 Transformers 格式。
5. **代码机制与参数说明**：
   - 当前部署代码为 `pre-kvcache` 分支（commit `6f2d746`），底层天然支持每个客户端携带各自独立的 `req.gamma`，云端批处理时按各请求自己的 `gamma` 截断验证。
   - 当前分支使用原生贪心匹配验证，无需指定 `--verify_method` 与 `--draft_top_k`（避免触发参数未知报错）。

---

## 2. 部署与执行步骤

### Step 0. [node2] 一次性准备：转换 Llama-2-7B 为 HuggingFace 格式

在 **node2** 执行该命令（若已存在 `/home/hdd/zhangh/models/Llama-2-7b-hf/config.json` 则跳过）：

```bash
/home/hdd/zhangh/envs/new_fastsd/bin/python \
  /home/hdd/sanzo/software/miniconda3/envs/ayo124/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
  --input_dir /home/hdd/zhangh/models/Llama-2-7b \
  --output_dir /home/hdd/zhangh/models/Llama-2-7b-hf \
  --llama_version 2 \
  --num_shards 1 \
  --safe_serialization
```

检查转换完成：
```bash
ls -l /home/hdd/zhangh/models/Llama-2-7b-hf/config.json
```

---

### Step 1. [node2] 启动 Cloud Target 服务

在 **node2** 终端执行：

```bash
cd /home/hdd/zhangh/workspace/new_fastsd

CUDA_VISIBLE_DEVICES=2 \
CLOUD_SERVICE_HOST=0.0.0.0 \
CLOUD_SERVICE_PORT=18001 \
/home/hdd/zhangh/envs/new_fastsd/bin/python cloud/cloud_service.py \
  --exp_name llama7b_hetero_target \
  --draft_model /home/hdd/zhangh/models/TinyLlama-1.1B-Chat-v1.0 \
  --target_model /home/hdd/zhangh/models/Llama-2-7b-hf \
  --target_device cuda:0 \
  --target_dtype bfloat16 \
  --server_sched_mode fastsd \
  --kv_batch_mode padded \
  --batch_size 3 \
  --token_budget 512 \
  --no-enable_pipeline \
  --temp 0 \
  --top_k 0 \
  --top_p 1.0 \
  --max_tokens 256
```

启动完成后，在 node2 另开终端验证服务健康状态：
```bash
curl http://127.0.0.1:18001/health
```

---

### Step 2. [node3] 启动模型 + Gamma 双重异构 Draft 进程

在 **node3** 终端执行以下脚本：

```bash
(
set -euo pipefail
cd /home/zhangh/workspace/new_fastsd-pre-kvcache-20260818

TARGET_URL="http://10.66.0.5:18001"
export PYTHON_BIN=/home/zhangh/envs/fastsd_cpu_py314/bin/python314_glibc

RUN="tiny_hetero_gamma_$(date +%Y%m%d_%H%M%S)"
OUT="$PWD/exp/$RUN"
mkdir -p "$OUT"

# 均匀划分请求数据集
awk '(NR-1)%3 == 0' data/humaneval.jsonl > "$OUT/chat.jsonl"
awk '(NR-1)%3 == 1' data/humaneval.jsonl > "$OUT/orca.jsonl"
awk '(NR-1)%3 == 2' data/humaneval.jsonl > "$OUT/python.jsonl"

launch_draft() {
  local name="$1" cores="$2" numa="$3" threads="$4" model="$5" gamma="$6"

  echo "[LAUNCH] 启动 $name: 模型=$model, 绑核=$cores, 线程=$threads, Gamma=$gamma"
  CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS="$threads" \
  MKL_NUM_THREADS="$threads" \
  OPENBLAS_NUM_THREADS="$threads" \
  nice -n 5 numactl --physcpubind="$cores" --membind="$numa" \
    "$PYTHON_BIN" edge/edge.py \
      --server_url "$TARGET_URL" \
      --exp_name "$RUN/$name" \
      --draft_model "/home/zhangh/models/$model" \
      --target_model /home/hdd/zhangh/models/Llama-2-7b-hf \
      --edge_use_cpu \
      --edge_threads "$threads" \
      --draft_dtype float32 \
      --num_drafts 1 \
      --profile custom \
      --server_sched_mode fastsd \
      --enable_latency_priority \
      --enable_proactive_draft \
      --no-enable_pipeline \
      --dataset humaneval \
      --data_path "$OUT/$name.jsonl" \
      --max_tasks_per_draft 0 \
      --arrival_distribution immediate \
      --max_tokens 256 \
      --gamma "$gamma" \
      --temp 0 \
      --top_k 0 \
      --top_p 1.0 \
      --seed 42 \
      --stop_policy eos \
      --request_timeout 300
}

# 1. Chat 模型: 8 线程 (NUMA 2), gamma = 4
launch_draft chat 56-63 2 8 TinyLlama-1.1B-Chat-v1.0 4 \
  > "$OUT/chat.log" 2>&1 &
pid_chat=$!

# 2. Orca 模型: 8 线程 (NUMA 2), gamma = 6
launch_draft orca 64-71 2 8 TinyLlama-1.1B-1T-OpenOrca 6 \
  > "$OUT/orca.log" 2>&1 &
pid_orca=$!

# 3. Python 模型: 16 线程 (NUMA 3), gamma = 8
launch_draft python 80-95 3 16 TinyLlama-1.1B-python-v0.1 8 \
  > "$OUT/python.log" 2>&1 &
pid_python=$!

echo "所有 Draft 进程已在后台启动，输出目录: $OUT"
status=0
wait "$pid_chat" || status=1
wait "$pid_orca" || status=1
wait "$pid_python" || status=1
echo "所有异构 Draft 进程执行完毕，退出状态码: $status"
exit "$status"
)
```

---

## 3. 运行监控与结果检查

### 检查 CPU 亲和性与运行状态（node3）：
```bash
ps -u "$USER" -o pid,psr,pcpu,pmem,args | grep edge.py
```

### 查看实时输出日志（node3）：
```bash
tail -f exp/tiny_hetero_gamma_*/chat.log
tail -f exp/tiny_hetero_gamma_*/orca.log
tail -f exp/tiny_hetero_gamma_*/python.log
```

### 查看 Target GPU 状态（node2）：
```bash
nvidia-smi
curl http://127.0.0.1:18001/health
```

### 实验指标结果（node3）：
执行完成后，各进程的详细吞吐、接受率和延迟指标记录在：
```text
exp/tiny_hetero_gamma_<timestamp>/
├── chat.log
├── orca.log
├── python.log
├── chat/
│   ├── edge_metrics_proc0.jsonl
│   └── edge_metrics_summary.json
├── orca/
│   ├── edge_metrics_proc0.jsonl
│   └── edge_metrics_summary.json
└── python/
    ├── edge_metrics_proc0.jsonl
    └── edge_metrics_summary.json
```

可通过以下命令快速打印各异构实例的总结结果：
```bash
python3 -c "
import json, glob
for path in sorted(glob.glob('exp/tiny_hetero_gamma_*/edge_metrics_summary.json')):
    data = json.load(open(path))
    print(path, 'Tokens/s:', data.get('tokens_per_second'), 'Accept Rate:', data.get('acceptance_rate'))
"
```
