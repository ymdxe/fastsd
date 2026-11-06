# 两台 CPU + 一台 GPU：手动跑通测试

本目录已准备好独立的代码快照与入口，没有启动模型或修改现有实验。已有 checkpoint 是 **Qwen3-0.6B** 和 **Qwen3-1.7B**；未找到你提到的 0.7B 目录。以下按 0.6B 编写。如果你指的是另一个 0.7B checkpoint，需要先确认其准确路径和与 Target 的 token ID 兼容性，不能仅按参数量替换。

## 1. 这次只验证什么

| 项目 | 设置 |
| --- | --- |
| cpunode1 | 1 个独立 Qwen3-0.6B，FP32，4 个 CPU 计算线程 |
| cpunode3 | 1 个独立 Qwen3-0.6B，FP32，4 个 CPU 计算线程 |
| gpunode2 | 1 个 Qwen3-1.7B，FP32，空闲 GPU，关闭 TF32 |
| γ / 长度 | 固定 γ=4，每条最多 32 个输出 token，EOS 提前停止 |
| 输入 | 两条固定短提示词，每台 CPU 各执行一遍；非 thinking、greedy |
| 到达方式 | 每台 CPU 连续处理两条；不使用泊松到达和全数据集 |
| 方法 | 先标准 SD，再 FastSD；每种方法共 4 条，总共 8 条 |
| Target | padded KV，batch_size=2，token_budget=256 |
| 网络 | GPU 监听 127.0.0.1:19635；两台 CPU 分别通过 SSH 转发访问本机 29635 |

4 个线程不等同于独占 4 个物理核。默认由系统调度；需要固定绑核时先看 `lscpu -e=CPU,CORE,SOCKET,NODE`，选 4 个不同 CORE、同一 NUMA 的可用 CPU，再在 edge 命令追加 `--cpus 选中的四个CPU编号`。本次只有每节点一个模型，线程配置不会按原来 16 个实例展开。

这里的 TTFT/TPOT 只用于确认指标正常生成。短请求、没有 warmup、没有同步起跑，不用于判定方法优劣，也不测 GPU 能耗。KV 预取在 FastSD 保持启用；少量请求不保证出现命中，命中数为 0 不能据此判断预取故障。

现有完整实验仍在运行时，额外 CPU 推理会干扰其计时。建议等它结束后操作；本方案不包含取消旧实验的命令。操作安排在 `exp` 的三个节点窗口，必须先确认对应 pane 已回到 shell，不能向正在运行的程序粘贴命令。

## 2. 同步这份独立测试包

在 gpunode2 的 shell 执行。下列 `zhangh@cpunode1`、`zhangh@cpunode3` 使用你平时的 SSH 登录地址；如实际是 IP 或别名，请相应替换。CPU 上无需覆盖现有项目代码。

```bash
cd /home/hdd3/zhangh/workspace/fastsd
scp experiments/manual-smoke.tar.gz zhangh@cpunode1:/tmp/fastsd-manual-smoke.tar.gz
scp experiments/manual-smoke.tar.gz zhangh@cpunode3:/tmp/fastsd-manual-smoke.tar.gz
```

两台 CPU 分别执行一次：

```bash
mkdir -p /home/zhangh/workspace/fastsd/experiments
tar -xzf /tmp/fastsd-manual-smoke.tar.gz -C /home/zhangh/workspace/fastsd/experiments
cd /home/zhangh/workspace/fastsd/experiments/manual-smoke
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('manifest.json').read_text())
for name, digest in manifest['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
print('测试包校验通过')
PY
```

该包包含准备时 `kvpre` 工作区的 `cloud/edge/src` 快照，包括未提交的修改；`manifest.json` 记录文件哈希。GPU 和 CPU 都使用包内代码，避免三端代码版本不同。不要在已经产生结果的目录里反复解包；只改 `--run` 可以开始新一轮测试。

## 3. 建立两条 SSH 转发

从 gpunode2 登录 CPU 的两个窗口，分别使用以下命令。`-R` 让 CPU 的 29635 转发到 gpunode2 的 19635；这两个 29635 在不同机器上，不冲突。连接要在测试期间保持打开。

cpunode1 窗口（这条 ssh 命令从 gpunode2 的 shell 发起）：

```bash
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -R 127.0.0.1:29635:127.0.0.1:19635 zhangh@cpunode1
```

cpunode3 窗口（同样从 gpunode2 的 shell 发起）：

```bash
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -R 127.0.0.1:29635:127.0.0.1:19635 zhangh@cpunode3
```

如果窗口已经 SSH 到 CPU，先正常退出该 SSH 回到 gpunode2，再用以上方式连接。不要从 CPU 再运行这些反向转发命令；转发方向会变。

## 4. 先跑标准 SD

gpunode2：先运行 `nvidia-smi`，查看空闲卡。下面 `GPU_ID` 必须填写你实际选中的物理 GPU 编号；不要照搬旧实验正在占用的 GPU1。

```bash
cd /home/hdd3/zhangh/workspace/fastsd/experiments/manual-smoke
nvidia-smi
read -r -p '空闲物理 GPU 编号: ' GPU_ID
CUDA_VISIBLE_DEVICES="$GPU_ID" \
  /home/hdd/zhangh/envs/new_fastsd/bin/python -u run.py cloud \
  --method standard --run trial1
```

等待日志出现 **`MANUAL_SMOKE_TARGET_READY`**。它在模型加载并同步 CUDA 后输出；仅看到 HTTP `/health` 成功不足以证明权重加载完毕。

cpunode1（SSH 已建立后，在 CPU 的 shell 执行）：

```bash
cd /home/zhangh/workspace/fastsd/experiments/manual-smoke
export PYTHONPATH=/home/zhangh/venvs/fastsd_cpu/lib/python3.11/site-packages:/home/zhangh/miniconda3/envs/fastsd/lib/python3.11/site-packages
/home/zhangh/venvs/fastsd_cpu/bin/python -u run.py edge \
  --method standard --run trial1 --node node1 --threads 4
```

cpunode3：

```bash
cd /home/zhangh/workspace/fastsd/experiments/manual-smoke
/home/zhangh/envs/fastsd_cpu/bin/python -u run.py edge \
  --method standard --run trial1 --node node3 --threads 4
```

两台 CPU 在云端 ready 后依次启动即可；每台启动后不要等待它跑完才切到另一台。每台仅加载一个 draft，完成两条后打印结果并回到 shell。

## 5. 停掉本次标准 SD 服务，再跑 FastSD

确认两台 CPU 都完成两条。随后在任一 CPU 窗口执行：

```bash
curl --fail --noproxy '*' -X POST http://127.0.0.1:29635/exit
```

这只关闭 **19635/29635 这套手动测试服务**。等待 gpunode2 的服务退出并回到 shell；FastSD 的调度统计也在这个正常退出过程中写入文件。不要使用 `pkill python` 等全局结束命令。

三端重新执行各自上一节的启动命令，**统一把 `--method standard` 改为 `--method fastsd`**。GPU 仍先启动，看到 ready 后启动两个 CPU。`--run trial1` 保持相同，两种方法写入不同目录。CPU 的 `PYTHONPATH` 和 GPU 的 `GPU_ID` 在原 shell 中仍可复用。

FastSD 两端配置自动一起开启 pipeline、主动草拟和 latency priority，固定 γ=4；标准 SD 则关闭这些开关。两种方法都使用相同输入、精度和长度限制。FastSD 客户端复用已验证的时钟校准修复，Target 的 HTTP keepalive 为 7200 秒，避免 CPU 草拟间隔导致连接过期。

FastSD 两台 CPU 全部结束后，同样执行一次上述 `/exit`，让 Target 正常退出并落盘调度统计。

## 6. 看结果

CPU 每个方法结束后会自动打印两条的输出和均值。也可以在各自 CPU 的 `manual-smoke` 目录运行：

```bash
python3 run.py summary --method standard --run trial1
python3 run.py summary --method fastsd --run trial1
```

本机文件：

- CPU：`runs/trial1/<standard或fastsd>/<node1或node3>/edge_metrics_proc0.jsonl`。
- CPU：同目录 `edge_metrics_summary.json`，含 TTFT、TPOT、输出 token 数和接受率。
- GPU：`runs/trial1/<standard或fastsd>/cloud/target_model_ready.json`。
- GPU：正常 `/exit` 后 `runs/trial1/fastsd/cloud/scheduler_metrics.json`，关注 `prefetch_async_failed` 和预取相关计数。
- FastSD 客户端时钟诊断：CPU 同目录 `timing-runtime*.json*`。不可靠单向通信估计不应解释为实测通信时间；烟测主要查看 `avg_http_total_ms`。

通过标准：两个方法中每台 CPU 各有 **2 条完整记录**，输出非空、token 数不超过 32，TTFT/TPOT 有效且非负（只生成 1 token 时 TPOT 无定义），没有请求错误、OOM 或 traceback。FastSD 正常关闭后预取失败计数应为 0。允许因为达到 32 token 上限而文本不完整。

本次已检查 Python 语法、命令参数配置和两种模型的 tokenizer 文件一致性；**尚未启动模型进行实际推理验证**。如果要重跑，把三端的 `--run trial1` 同步改成 `--run trial2`，入口会拒绝覆盖旧运行目录。
