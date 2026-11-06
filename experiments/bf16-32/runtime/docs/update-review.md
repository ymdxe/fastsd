# 审阅方案落实记录

基于 `kvpre` 的 `2db0d070ed0639a2a41698862b525a2de075df7f`，在独立工作目录 `/home/hdd3/zhangh/workspace/fastsd-update` 的 `update` 分支修改公共 `src/`、`edge/`、`cloud/`。历史实验 runtime 快照保持原样，复现实验时应明确选择新版公共入口。

## 已实现

| 审阅问题 | 修改与可验证行为 |
| --- | --- |
| 全历史概率常驻 | 公共 Target 和 HTTP Edge 启用带绝对位置的概率窗口。Target Prefill 只保存最后一行，Verify 保留当前片段与边界，回滚后只留确认边界。Draft 按当前轮与 ahead 保留概率。克隆裁剪视图，避免小窗口继续引用完整 logits 存储。拒绝采样仍使用完整 FP32 Target 分布。 |
| 会话泄漏 | 新增 `/session/release`；Edge 样本及 warmup 使用 finally 释放；取消、HTTP 等待超时、空闲超时进入同一回收路径。正在执行/保留的缓存延后回收，同时清理概率、RNG、接受统计、时延估计和版本状态。 |
| Prefill 每块往返 | 未完成的 Prefill 和内部 Verify 切片在显存余量允许时保留 GPU KV。结束逻辑片段后再执行卸载策略，避免先搬运尚需回滚的完整 KV。 |
| 完成结果不能唤醒 | 入站中继队列与 executor 结果队列共享 Event。控制线程先清事件再 drain，避免漏唤醒；空闲超时只用于会话维护。 |
| Prefill 预算浪费 | 仅一个可执行 Prefill 时扩展已选择的 slice，使用剩余 token budget，支持 batch_size=1；存在 Verify 或其他竞争任务时保留 quantum 限制。 |
| logical gamma 混淆 | 切片保留 logical gamma；草拟速度优先使用直接测量，否则用整轮 lag/整轮 gamma。整轮响应汇总各切片 Verify 时间。 |
| 首轮没有真正重叠 | FastSD 时延路径立即上传 prompt，Target Prefill 与本地首轮草拟独立进行；两者完成后才 Verify。上一样本的本地测量用于下一次 Prefill 估计，当前轮真实测量用于 Verify。旧 arrival 协议继续兼容。 |
| ahead 命中仍重算 | 先检查全接受及 bridge 命中，再决定 KV 裁剪位置。保留下一轮实际使用的 ahead 前缀，gamma 缩短时裁掉多余后缀。 |
| ahead 与 gamma 绑定 | `--ahead_max_tokens` 独立控制提前生成上限（含 bridge，默认 4）。结合上一轮等待窗口、单步成本和最近净收益停止提前计算。 |
| 缺少 ahead 成本统计 | 每请求新增 generated/reused/discarded 数量、ahead 前向耗时、响应就绪至消费延迟、Draft 实际 forward token 数及进程 CPU 时间。reused 只计下一轮真正送去验证的复用候选，bridge 不计入该数。 |
| 预取强引用与残余预测 | 一次只持有正在搬运的缓存；驱逐 DMA 完成后再加载。lookahead 在只读副本上模拟当前切片完成，保留剩余工作；实际安装仍检查 active/reserved、版本、对象和长度。 |
| CUDA 搬运生命周期 | 独立 copy stream、pinned CPU 缓冲池、ready event、计算流等待及存储存活保护。兼容 Transformers 4.57 首次 DynamicCache.update 也会复制的行为，避免未完成的 CPU DMA 被提前读取。 |
| KV 和 GQA 开销 | varlen 使用按容量预分配的连续 KV 缓冲区，默认每 128 tokens 扩容；支持时使用原生 SDPA GQA，旧 PyTorch 保留展开回退。修复 greedy 归一化的跨行索引问题并批量归一化。 |
| 质量分母 | 以有效数字 reference 为分母，输出缺失或无法解析数字计错。 |
| GPU 计时 | 独立返回 `verify_gpu_ms`，使用 CUDA Events；`verify_ms` 保留主机墙钟时间含义。 |

## 可选方法与对照

`--server_sched_mode fastsd` 下增加 `--scheduler_policy legacy|fcfs|progress`。三种选择共享执行后端、batch、切片与 KV 管理，默认 `legacy` 保留原排序。`fcfs` 是同后端的批处理对照；`progress` 使用接受长度生存概率、在线 batch 成本表、上下文与非驻留 KV 搬运估计，超时请求优先按等待时长服务。

`--gamma_policy progress` 为默认自适应长度策略：用整轮接受统计估计新增提交 token，比较候选 gamma 的进展/成本，保留步幅限制、迟滞与拥塞保护。`--gamma_policy legacy` 可复现旧窗口填充策略；`--no-pipeline_gamma_adapt` 固定 gamma。新成本模型是启发式起点，尚未做真实模型负载校准，不能据此宣称方法提升。

建议以下消融固定模型、采样、种子、输入、batch_size、token_budget、KV 后端和缓存策略：

| 对照 | scheduler_policy | prefill_chunk_quantum | kv_prefetch | proactive | gamma_adapt |
| --- | --- | --- | --- | --- | --- |
| 批处理 FCFS | fcfs | token_budget | 关闭 | 关闭 | 关闭 |
| 加 chunking | fcfs | 128 | 关闭 | 关闭 | 关闭 |
| 加原优先级 | legacy | 128 | 关闭 | 关闭 | 关闭 |
| 加预取 | legacy | 128 | 开启 | 关闭 | 关闭 |
| 加 ahead | legacy | 128 | 开启 | 开启 | 关闭 |
| 加收益 gamma | legacy | 128 | 开启 | 开启 | 开启，progress |
| 新调度估计 | progress | 128 | 开启 | 开启 | 开启，progress |

所有配置都使用新的正确性修复。比较“关闭 ahead”请用 `--no-enable_proactive_draft`；关闭预取用 `--no-kv_prefetch`。第一行的长 Prefill 仍受总 token budget 约束，不是无限长的不可中断请求。

## 新配置与测量边界

- `--session_idle_timeout 300`：无工作、无 active/reserved 所有权的会话超时回收；0 关闭。HTTP 请求超时单独由 `FASTSD_REQUEST_TIMEOUT_S` 控制，默认 300 秒。
- `--kv_headroom_mb 512`：剩余 CUDA 内存低于该值时卸载继续任务的 KV。
- `--async_kv_transfer` / `--no-async_kv_transfer`：异步与同步搬运对照。
- `--pinned_pool_mb 128`：可复用空闲 pinned 缓冲池容量；活跃会话及在途传输的存储另计。
- `FASTSD_KV_BLOCK_TOKENS=128`：连续 KV 扩容粒度；`FASTSD_NATIVE_GQA=0` 强制旧展开路径。
- `--scheduler_max_wait_s 1`：progress 策略中请求转为紧迫状态的阈值，不是过载下的硬 SLO 保证。
- `--kv_transfer_gbps 10`：progress 策略的初始传输估计，单位 GB/s，应按实际主机校准。

当前显存策略采用剩余容量阈值，**尚未实现覆盖所有 logits、attention workspace 和并发 DMA 的硬峰值显存准入预算**。KV 为连续预分配结构，**尚未接入 paged attention 专用后端**。每序列 SDPA 调用仍存在。原生 GQA 是否减少实际临时展开取决于选中的 PyTorch kernel。

本次测试验证功能、并发协调、分布与存储生命周期；没有运行正式模型的多并发 GPU 压测，没有验证 DMA 与 forward 的实际重叠比例、长期 RSS/GPU 曲线或 SLO goodput。独立开放式 Target 到达压测器仍需另行建设，现有 Edge 串行 worker 结果不能当作纯 Target serving capacity。

## 验证方式

在服务器已有环境中运行：

```bash
cd /home/hdd3/zhangh/workspace/fastsd-update
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/hdd/zhangh/envs/specedge/bin/python -m unittest discover -s tests -v
```

新增测试覆盖有限概率窗口的物理存储、微型 Llama 分布对齐、分块整轮接受、重复会话回收、事件唤醒、部分任务预测、gamma 速度估计、成本控制、CUDA 跨流复制及真实 GPU worker 的异步卸载与重载。

2026-10-27 最终运行 235 项：232 项通过、3 项因未配置指定 Llama 检查点而跳过，0 失败、0 错误，耗时 37.146 秒。测试使用现有环境和微型随机模型，CUDA_VISIBLE_DEVICES=2；不包含正式模型压测。保留 CRLF 源文件，并以 `cr-at-eol` 检查行尾后的 `git diff --check` 通过。
