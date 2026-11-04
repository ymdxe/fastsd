# 从初始提交实现论文算法

本分支以 `39fdab402c659dc189dc672e4460349d9559c866` 为唯一代码基线，
根据用户提供的 `1020/FastSD.pdf` 第 3.2–3.4 节及 Algorithm 1 实现。
没有合并、移植或参照后续开发分支的实现。

论文运行入口是 `edge/edge.py` 与 `cloud/cloud_service.py`，
云端继续使用原有 `Decoding.run_target_process_batching` 主循环和原生模型 forward。
原有 benchmark 中的本地多进程 draft 路径不是本分支的论文运行入口。

## 方法与代码对应

| 论文要求 | 实现位置及行为 |
| --- | --- |
| 六队列、动态长度阈值 | `src/fastsd_scheduler.py`、`src/engine.py`：分别记录最近 100 个新 prefill/verify 任务的原始长度；默认阈值为 128/512 和 20/100；窗口满后使用 70%/90% 分位数。队列重建保留到达时间与分片进度，分片不会重复进入长度历史。 |
| 式 (3)、(4) 的优先级 | `compute_priority_score`：初始 prefill 的接受率为 1；verify 使用 `(accepted + 1)/(drafted + 1)` 平滑历史接受率。等待项为 `exp(0.01 * W)`，所有时长单位为秒。 |
| 实际端云计时 | `edge/edge.py`、`cloud/cloud_service.py`：先登记云端到达时间，再测量本次本地 prefill（包含首 token）和随后解码；复用已生成的首轮草稿。上传采用本次发送时间与时钟偏移估计，下载采用上一轮验证回复的测量值。 |
| WRR 与 soft fill | `plan_batch`：保留初始代码的 6:3:1 访问配置；类内 verify 优先，prefill 连续两个完整周期未获服务后提升优先级；当前类无可行任务时，按短、中、长类顺序搜索其他队列。soft fill 不额外消耗访问次数。 |
| 资源限制和分片 | 一个计划每设备最多一次；两阶段共用 `token_budget` 和 `batch_size`。预算计算实际新 forward token，包括尚未计算的 bridge token。长任务保留 offset，只在完成或发生拒绝时回复。 |
| Algorithm 1 的只读规划 | `BatchPlan` 记录任务、offset、token 分配与提议的 WRR 状态；规划不更新任务进度。实际计算完成后提交 WRR 状态和 prefill 等待周期。 |
| verify 先于 prefill | 同一计划拆为两个子批次，按 verify、prefill 顺序执行。两者不会分别获得一份完整预算。 |
| 式 (8) 的预测 | 使用同一份当前队列与当前 WRR 状态，再调用同一个 planner；规划前排除当前已选设备。预测计划的状态被丢弃，不投影当前批次执行后的结果。 |
| 异步 KV 预取 | `src/kv_preload.py`：单后台线程与独立 CUDA stream，从预测计划中提取 verify 设备并预取其 CPU 缓存；主推理线程检查原缓存对象和长度，再等待 CUDA event 并安装；未命中时按需加载。 |
| 无 padding 的变长执行 | `src/kvcache_batching.py`：拼接有效 token 和历史 KV，使用各设备独立的位置编号与块因果 attention mask；计算后分回独立 KV，支持逐设备回滚。 |

保留初始代码的贪心接受/拒绝规则。一次 verify 跨多个分片时，
全部接受的中间分片继续执行，不提前生成 bonus token；拒绝或最后一个分片完成后才提交本轮接受率。
`src/util.py` 同时修复了多行贪心概率计算的索引问题，防止一行得到其他行的最大值位置。

`server_sched_mode=fastsd` 关闭初始框架的主动续写和 gamma 自适应。
固定 gamma、完整前缀上传和 tail-only 上传均可使用；tail-only 是传输格式，
不改变论文的调度算法。没有增加收益控制、预取校准、配额或限速策略。

## 运行

端云应使用本分支的同一版本，HTTP 请求增加了论文所需的时延字段。
沿用原仓库的环境安装和模型路径配置；`install.sh` 固定的 Transformers 版本为 4.57.6。

云端示例：

```bash
python cloud/cloud_service.py \
  --profile custom --server_sched_mode fastsd \
  --draft_model TinyLlama-1.1B-Chat-v1.0-GPTQ --target_model llama-2-7b \
  --token_budget 64 --batch_size 4 --gamma 6 --exp_name paper_cloud
```

边缘端示例：

```bash
SERVER_URL=http://CLOUD_HOST:8001 bash scripts/run_fastsd_profile.sh paper_edge \
  --draft_model TinyLlama-1.1B-Chat-v1.0-GPTQ \
  --num_drafts 4 --max_tasks_per_draft 10 --gamma 6
```

`token_budget` 必须能容纳 bridge 与至少一个 draft token，正常续轮至少需要 2。

默认 `token_budget=64`。`vanilla` Standard SD 同样按该预算分片，继续采用单设备处理；
bridge 只在一个 verify 请求的第一片计算。输出长度仍由 `max_tokens` 控制。
四方法 Config 1 的预算口径与 SpecEdge 图形状见
[`experiments/config1-budget64.md`](experiments/config1-budget64.md)。
端侧按 `max_tasks_per_draft` 处理样本，不再受初始代码硬编码的单样本限制；完成后释放会话缓存。

时钟同步使用 5 次交换中 RTT 最小的一次估计偏移。单向通信时间受链路不对称影响，
属于测量估计；等待时间 `W` 始终使用云端单调时钟，不依赖端侧时钟。

## 验证与实现边界

2026-11-03 在 CPU 环境完成以下验证：

- 原仓库 21 项 unittest 全部通过，Python 编译与差异空白检查通过。
- PyTorch 2.6.0 下，Transformers 4.51.3 和 4.57.6 的小型随机 Llama、Qwen3 模型，
  在 eager/SDPA 后端通过不等长序列、不同 KV 长度、续写、回滚和设备隔离的数值对照；
  最大概率绝对误差约 `3.73e-9`。
- 使用真实小型模型的 HTTP 与调度集成验证覆盖长 prefill/verify 分片、bridge 预算、
  全接受、跨分片拒绝、缓存释放及错误请求后的继续服务；混合计划验证 verify 先执行且预算共享。
- 端侧完整循环验证首轮草稿只生成一次、连续多样本运行，以及完整前缀和 tail-only 两种上传格式。
- 临时集成验证脚本位于仓库之外的验证目录，不加入仓库测试；未新增单元测试。

当前无 padding 后端是原生模型加稠密块 mask，仍有缓存拼接和被 mask 掉的注意力计算，
不是融合的 ragged FlashAttention kernel。要求 full attention、eager/SDPA、默认或 linear RoPE。
因此不能仅凭本次功能验证宣称吞吐提升或复现论文性能。

本次使用 CPU 验证环境；GPU 上的真实模型、GPTQ 内核、异步拷贝重叠及峰值显存尚未实测。
CPU 验证覆盖执行语义，CUDA 预取路径需要在实际实验服务器上进一步测量。
