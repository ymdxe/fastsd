# 固定 γ=20：Standard SD 与 FastSD

日期：2026-10-23。两种方法各 32/32 请求完成，使用同一批 GSM8K 前 32 条题目。

## 配置

- node3：32 个独立 Qwen3-1.7B FP16 draft，每 worker 一个物理核心、一个计算线程；4 个 NUMA 节点各 8 个 worker，绑定节点内存。
- node1：Qwen3-1.7B FP16 CPU Target，8 个计算线程；batch_size=8，token_budget=256。
- 两种方法固定 gamma=20，greedy，正常 EOS，max_new_tokens=64，无 warmup；依次执行，源代码、CPU/NUMA 映射、问题和 non-thinking 模板相同。
- FastSD 使用 custom/fastsd、pipeline、proactive drafting、latency priority；关闭 gamma adaptation，min/max 同设为20。逐次 RPC 记录确认实际 verify gamma 均为20。
- FastSD 沿用上轮验证通过的顺序时钟校准和校准后统一放行；保留所有原生时间校验。
- 上一轮上限16小于本次gamma20，因此本轮将两侧上限统一提高为64。不得将本轮与旧的16-token测试直接比较。

## 结果

| 指标 | Standard SD | FastSD |
| --- | ---: | ---: |
| 请求 E2E 中位数 | 324.44 秒 | 311.27 秒 |
| 请求 TTFT 中位数 | 197.70 秒 | 179.40 秒 |
| 首轮 draft 中位数 | 168.58 秒 | 166.71 秒 |
| prefill HTTP 中位数 | 32659.08 ms | 5348.10 ms |
| verify HTTP 中位数 | 692.78 ms | 1985.99 ms |
| Target verify 计算中位数 | 664.33 ms | 1327.47 ms |
| HTTP 减 Target 服务时间中位数 | 4.45 ms | 2.84 ms |
| 请求 E2E P95（秒） | 394.36 | 371.60 |
| verify HTTP P95（ms） | 1416.35 | 12461.54 |
| 输出 tokens | 2048 | 2048 |
| 协议 RPC 数 | 192 | 224 |
| 提前草拟复用命中 | 0 | 125 |
| verify 次数 | 128 | 128 |
| 草稿接受率 | 99.69% | 98.95% |
| 估计最后输出完成窗口（秒） | 450.000 | 398.845 |
| 估计输出吞吐（tokens/s） | 4.551 | 5.135 |
| 全部 worker 完成窗口（秒） | 450.126 | 398.849 |
| 包含 worker 收尾的吞吐（tokens/s） | 4.550 | 5.135 |
| 最后输出到全部 worker 完成间隔（秒） | 0.126 | 0.004 |

逐题完整输出 token 一致：**32/32**。差异明细（若有）保存在 comparison.json。

## 统计口径与限制

- 输出窗口从全部模型就绪时的 barrier 起算，结束时间由每个 session/init 开始时刻加原生请求 E2E 估算；包括启动校准、prefill 和 decode，排除模型加载。
- worker 窗口使用进程内完成时间戳，包含原生执行器等待尚未完成工作的收尾时间。两种口径均报告，避免隐藏提前草拟的额外工作。
- 候选块保持固定gamma20，最后一块可能超过剩余输出额度；输出按64-token/EOS截断，因此接受的候选数可能超过最终保留的输出数。
- HTTP总耗时包含Target排队和计算；HTTP减服务耗时也包含序列化等开销，不是纯网络RTT。
- 两侧为同一1.7B模型，接受率不代表未来32B Target；CPU预取计数不能代表GPU传输收益。
- 共享服务器，单次32题短输出测试，后台负载随时间变化；不能作为正式加速结论或GSM8K准确率。

## 复现与文件

两个目录均包含 launch_target.py、launch_drafts.py、*_launch.json、commands.txt、manifest.json、source/、workload32.jsonl、worker_map.json、preflight.txt、progress.jsonl、run32.log 和逐 worker/RPC/输出记录。

对照目录：`../1023-sd-g20/` 与 `../1023-fastsd-g20/`。node1 对应目录保存 target.log、Target 启动环境及正常关闭时的调度统计。运行界面均使用 tmux 会话6；实验结束后关闭临时进程，保留窗格。

## 调度与负载证据

FastSD 执行 85 个调度计划、128 个 verify 切片和 35 个 prefill 切片，其中 3 次分块。提前草拟复用命中125轮；预取完成4次、失败0次，这些CPU缓存计数不代表GPU传输收益。

两轮主机负载存在差异，progress.jsonl 保存了连续快照；负载包含本实验工作，不能直接当作后台负载。当前单次结果只描述观察值。原项目推理源码与基线一致。
