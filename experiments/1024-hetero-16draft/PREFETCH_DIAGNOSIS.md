# 本轮KV预取为何没有发生

## 已确认的结论

本轮没有记录到预取执行失败。实际情况是预取目标集合始终为空，后台线程没有收到任何搬运任务。

| Target生命周期计数（含冒烟） | 值 |
| --- | ---: |
| prefetch_plans | 135528 |
| prefetch_misses / hits / evictions | 0 / 0 / 0 |
| prefetch_async_submitted / dropped | 0 / 0 |
| prefetch_async_completed / moved / failed | 0 / 0 / 0 |
| prefetch_async_last_error | null |
| 实际执行的调度计划 plans | 308 |

`target.log`没有`KV-PREFETCH-FAILED`记录。当前`src/engine.py`、`src/fastsd_scheduler.py`与实验快照逐字节相同。

这不是仅凭零完成数推测：`prefetch_next_plan()`在禁用缓存卸载时会直接返回，不增加`prefetch_plans`。本轮该计数非零，说明规划路径运行了。该函数对非空`desired`必然增加`prefetch_hits`或`prefetch_misses`；两者始终为0，因此这条路径从未选出非空的下一批验证集合。任务还没有进入CUDA搬移阶段，不能据此归因于显存、拷贝异常或后台线程执行失败。

## 触发窗口的限制

实际执行顺序是：

```text
HTTP进程 -> request_queue
                 |
        Target主线程接收入站请求
                 v
              task_queues
                 |
          选出 current_plan
                 |
      排除 current_plan 的任务后再规划
                 |
       next_plan.verify_proc_ids
                 |
         非空且需搬运才提交预取
                 |
          执行本轮模型前向
```

1. 预取只看已进入`task_queues`、且未被本轮选中的verify。它不预测客户端尚在本地草拟的下一次请求，也不预取prefill的空缓存。
2. 入站接收和模型前向在同一个Target主循环中串行执行。模型计算过程中到达的请求停在`request_queue`，不会自动进入预取器读取的`task_queues`。
3. 等当前计算结束，这些新请求被接收并可能全部选入下一次`current_plan`，随后又被lookahead排除。因此，即便存在HTTP排队，也不意味着预测时刻有可见的后续verify。

源码位置：`source/src/engine.py`的`_plan_queues()`、`prefetch_next_plan()`及主循环约1900–2000行。主循环在执行前向之前调用预取，传入`reserved_work_ids=current_plan.selected_work_ids`。

## 本轮负载的证据

本轮共有16个draft实例，但它们大部分时间在CPU本地计算，不等于Target一直有16个排队verify。

从362条正式verify的相同`verify_ms`及经时钟偏移校准的相近响应结束时刻，估计得到297个计算批次：

| 估计批大小 | 批次数 |
| --- | ---: |
| 1 | 247 |
| 2 | 39 |
| 3 | 8 |
| 4 | 2 |
| 5 | 1 |

约83.2%的批次只有1个请求，估计最大批为5；HTTP verify同时在途峰值为6。配置容量为`batch_size=8, token_budget=512`。这些记录支持“通常没有跨批积压verify”的解释。

以上批处理分组来自响应元数据，**不是原始调度队列快照**。在途HTTP数量还包含执行、入站等待和传输，不能直接等同于`task_queues`长度。原实验未记录每轮live队列、reserved集合和desired集合，因此无法精确重放每次空预测的队列组成。

## 用原调度代码复现

`prefetch_diagnosis.py`直接提取并执行实验快照中的`prefetch_next_plan()`、`sort_task_queues()`，调用同一份`plan_iteration()`。不加载模型、不申请CUDA显存。

| 合成队列场景 | 本轮选中 | 提交的预取对象数 |
| --- | ---: | ---: |
| 1个γ100 | 1 | 0 |
| 8个γ20 | 8 | 0 |
| 4个γ100＋4个γ20 | 8 | 0 |
| 8个γ100同时积压 | 6（含一个切片） | 2 |
| 8个γ100＋8个γ20同时积压 | 6 | 8 |

有足够的已入队积压时，同一提交逻辑确实能产生任务。另一个复现中，新请求只位于入站队列时，预取提交为0；将它提前放入调度队列、保留当前批次占位后，预取器立即选中它。如果按现有串行顺序等到下一轮再接收，它成为新current_plan，预取又为空。

这些是触发/提交路径的正反例，不是原实验实际队列的回放，也不证明CUDA拷贝已经验证成功或真正与计算重叠。

## 独立发现：完成任务没有从调度队列清理

`commit_admission_plan()`将任务标记为finished，但`sort_task_queues()`仍把它放回队列；正常完成路径也没有从`work_items`移除它。主循环用物理队列是否非空决定是否规划，planner却会过滤finished项，于是重复排序完成项并产生空计划。

复现：队列仅放1个finished任务，连续执行3次原队列排序和规划后，队列仍有1项、`prefetch_plans=3`、提交数为0。日志中135528次iteration对比308个实际计划，存在大量这样的无效规划，不能将13.5万解释为有效预取机会。

该问题会增加维护开销并使计数失真；planner仍能跳过完成项选择新的有效任务，所以它不是“有有效候选却被旧任务硬性挡住”的证据，不能单独解释所有零提交。

## 实际KV搬运与后续修正方向

后台预取计数为0不代表完全没有KV搬运。验证前，主线程仍会同步调用`move_dynamic_cache_to(cache, target_model.device)`；验证后按策略把KV移回CPU。请求能正常完成，但没有获得后台提前搬运的收益。

建议按顺序处理：

1. 清理finished WorkItem及相关队列引用，区分空队列、空计划、有效预测的计数。
2. 增加每轮live verify数、reserved数、desired数、入站队列可见性以及无目标原因的日志。
3. 若希望覆盖“当前计算期间到达”的请求，需要将接收/预取规划与阻塞前向解耦，并保持缓存所有权与版本保护；之后再用GPU trace验证实际搬运/计算重叠和端到端收益。

原始统计和复现结果见[prefetch_diagnosis.json](prefetch_diagnosis.json)，可执行诊断见[prefetch_diagnosis.py](prefetch_diagnosis.py)。本次检查未修改推理源码或重新运行模型实验。
