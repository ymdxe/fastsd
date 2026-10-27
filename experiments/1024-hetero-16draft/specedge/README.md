# 本轮 SpecEdge 单链实现与执行

必须在本轮 Standard SD / FastSD 完成并关闭其 Target 后运行。本目录尚未执行真实32B模型加载或正式服务；已完成以下代码与CPU验证。

## 保留与适配范围

`vendor/`是固定提交`1edcaf02ffc41a7b57726450c5357ed216a3b9bc`的39个源码/协议/许可证文件，逐文件SHA256见`vendor_manifest.json`，原始字节未修改。`bootstrap.py`在内存中等价转换一条日志f-string，并提供Python3.10的`typing.Self`和异步日志兼容。

CPU客户端保留官方`SpecExecClient`、Tree、proactive及自定义Qwen3/KV实现；每个父节点只有一个子节点。主深度和budget分别固定为node1的100、node3的20，主动草拟为bridge+同样长度的单链，最终输出按256/EOS截断。候选块不随剩余输出额度缩短。官方主动草拟同步完成后一个cycle才结束，因此请求E2E/TPOT包含它的全部计算；完成通知另由公共coordinator记录。

服务端保留官方gRPC消息/servicer、Qwen3实现、KVCache和BatchGraphEngine；实验适配器用实际长度padding替换固定宽度入队/回滚逻辑。真实chain在GPU计算，padding放入预留cache位置且不进入接受前缀，响应按每请求真实宽度返回。最终验证选择仍由官方greedy采样和客户端SpecExec逻辑决定。服务端只将已接受的连续前缀KV保留到CPU，供下轮加载，记录所有batch/接受数。

固定设置：16client，node1/3各8个独立CPU FP16 Qwen3-1.7B；Target Qwen3-32B FP16，仅物理GPU3，GPU/CPU权重预算34/96GiB；活动batch=8，动态入队，max_len=768，CUDA Graph关闭，prefill缓存关闭，gRPC超时7200秒。最大prompt123 + 输出256 + 两块100 + bridge/检查余量小于584，768足够；8slot GPU静态KV约1.5GiB，16client历史KV按实际长度保存在CPU。输入、绑核、NUMA和non-thinking模板复用本轮，wire mask明确为FP16。SpecEdge没有与native token_budget=512等价的参数，本实现记录token_budget=null，不声称该调度限制相同。

## 云端：tmux8

云端现有`new_fastsd`环境有必要的gRPC、protobuf、rich等依赖，其torch/Transformers/Accelerate与本轮相同。不安装新依赖。

```bash
/home/hdd/zhangh/envs/new_fastsd/bin/python experiments/1024-hetero-16draft/specedge/launch_server.py --smoke-first
```

此方式仅加载一次权重，持久服务输出始终在`specedge/target/`。初始阶段为smoke，期望一个Sync客户端和gamma2。首个真实冒烟采用node1/global client0、输出4tokens，CPU结果写到独立的`smoke-specedge/node1/`。

云端检查真正模型就绪：

```bash
curl --fail --silent http://127.0.0.1:18626/health
```

需要`ready=true`且`worker_alive=true`；加载失败见`worker_error.json`。仅gRPC端口监听不代表模型就绪。冒烟client完全结束后，云端切换正式阶段：

```bash
curl --fail --silent -X POST http://127.0.0.1:18626/phase -H 'Content-Type: application/json' -d '{"phase":"formal"}'
```

该操作拒绝仍有in-flight请求或Sync等待的情况；在worker中保存` scheduler_metrics-smoke.json`，清空全部会话KV和计数，再将γ设置为100/20、Sync人数设为16，保留同一模型和worker PID。响应应包含`ok=true, model_reloaded=false, client_kv_cleared=true`。首次正式请求的prefill会为client0..15重新建立cache，不继承冒烟client0状态。仅允许一次smoke→formal转换，防止清空正式结果。先收到成功响应，再启动两节点正式client。

gRPC监听`127.0.0.1:18625`，复用CPU已有`28625`转发；管理HTTP只监听node2的`127.0.0.1:18626`。`launch_server.py`会先检查两个端口空闲；不会停止其他任务。两阶段batch日志分别为`batches-smoke.jsonl`、`batches-formal.jsonl`，模型及stdout日志共享同一次服务归档。

全部正式请求结束后优雅关闭：

```bash
curl --fail --silent -X POST http://127.0.0.1:18626/shutdown
```

也可对`specedge/target/target.pid`记录的本次服务PID发送SIGINT。服务通知worker退出并等待进程结束，保存正式`scheduler_metrics.json`和launcher的`exit.json`。核对`service.json`中worker PID已退出、`exit_status=0`、GPU3上对应实验进程消失，才能确认GPU释放；不能只以关闭HTTP管理端口判断。超过退出等待时，官方cleanup仅终止其自身创建的worker。

如需直接正式运行，省略`--smoke-first`即可。额外混合宽度冒烟需使用独立归档及新的coordinator状态，不要覆盖已有冒烟目录。

## CPU：tmux6

把`specedge/`源码目录分发到两CPU节点同一实验根目录；排除`.coordinator-token`，CPU节点已经有该文件。同时排除本地`target/`、`node1/`、`node3/`结果、`__pycache__`和日志。依赖使用现有环境；node1 launcher自动设置与本轮`launch_cpu.py`相同的venv/Conda PYTHONPATH组合。

明确的代码allowlist打包命令（在gpunode2的tmux8执行）：

```bash
tar --exclude='__pycache__' -czf /tmp/fastsd-specedge-code.tar.gz -C /home/hdd3/zhangh/workspace/fastsd/experiments/1024-hetero-16draft specedge/bootstrap.py specedge/client_adapter.py specedge/launch_clients.py specedge/server_adapter.py specedge/launch_server.py specedge/vendor specedge/vendor_manifest.json specedge/README.md
```

CPU已有的`node1-worker-map.json`/`node3-worker-map.json`、`nodeN-workload.jsonl`和`.coordinator-token`留在原根目录使用，不加入代码压缩包。通过本轮已建立的文件传输机制在tmux6下载并解压到对应实验根目录即可。

node1冒烟：

```bash
/home/zhangh/venvs/fastsd_cpu/bin/python specedge/launch_clients.py node1 --run smoke-specedge --gamma 2 --max-new-tokens 4
```

正式运行两节点：

```bash
# node1
/home/zhangh/venvs/fastsd_cpu/bin/python specedge/launch_clients.py node1
# node3：使用本轮已验证的激活环境python
python specedge/launch_clients.py node3
```

默认每node8worker、client_idx分别0–7与8–15，公共barrier人数16。CPU本地端口`28624`访问coordinator、`28625`访问gRPC。请求完成后上传到公共实验目录的`specedge/node1|node3`，包含`run/edge_metrics_proc*.jsonl`、RPC/cycle、worker/completion和prompt token记录，可由已扩展的`../summarize.py`读取。

## 已完成验证与剩余验收

- Python3.10语法及真实依赖import通过；vendor hash逐文件保持一致。
- Python3.10异步日志配置及输出通过。
- 客户端官方grow/verify/proactive在gamma2/20/100下的全接受、拒绝、主动复用、输出cap模拟通过；FP16 wire、7200秒超时、响应宽度和分支拒绝通过。
- 服务端混合gamma100/20模拟覆盖padding隔离、拒绝、后续轮KV连续性和响应裁剪，通过。
- 用CPU随机初始化的微型真实Qwen3和官方KV执行混合gamma100/20、连续两轮共4请求，与HuggingFace参考全前缀前向的greedy选择完全一致。此检查未加载checkpoint或使用GPU。
- smoke→formal清理状态、保存冒烟统计和复用同一模型的控制路径验证通过。
- 剩余：两个CPU节点实际SpecEdge依赖import；GPU3+CPU offload下32B真实冒烟；跨机gRPC及16worker正式运行。需在前两种方法结束后执行。

本路径结果标为`specedge_cpu_adapted_single_chain`，不能标为未适配官方性能。gRPC与HTTP延迟不能当作同协议网络开销对比；主要比较同口径TTFT、TPOT、E2E和coordinator公共窗口吞吐。
