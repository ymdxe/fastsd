# kvpre 修改与审查记录

日期：2026-10-24。分支：`kvpre`。基线：`ee0e334e222593c100b2ee6fc7fc9e4db2cf7cd3`。

用户要求由 tmux `dscode:0.0` 中的 Claude Code 实现，由 Codex 制定约束并独立审查。实现和审查已完成，改动尚未 commit/push。

## 行为变化

- 完成、拒绝完成及取消的 WorkItem 从分类队列和索引移除；分块任务提交进度后会重新分类。会话KV和跨轮接受率状态仍由原生命周期管理。
- FastSD使用一个控制所有者、一个模型执行线程和一个KV搬运线程。控制线程在模型前向期间继续接收入站请求，刷新排除当前计划后的预取候选。
- Verify响应在其阶段完成后返回，不等待同一计划中的Prefill结束。vanilla/pipeline保持原有串行FCFS执行路径。
- 缓存使用active/reserved保护与版本校验；过期意图不能安装旧缓存，同目标重试与待处理计数有明确状态。
- None关闭时处理已接收任务；致命错误可中止后续阶段；退出等待当前不可中断调用真正结束后才释放资源。HTTP admission与退出哨兵使用同一临界区，关闭后新工作返回503。
- 模型执行线程显式禁用梯度，并设置目标CUDA设备上下文。

涉及实现：[engine.py](../src/engine.py)、[scheduler_runtime.py](../src/scheduler_runtime.py)、[cloud_service.py](../cloud/cloud_service.py)。新增测试覆盖运行时协调、执行上下文和HTTP关闭入口。

## 审查中发现并修复的问题

1. None哨兵可能丢弃已接收但尚未提交的任务。
2. 阶段确认ack可能导致停机线程不退出，或停止后继续进入下一前向阶段。
3. 持续入站流量可能饿死已有ready任务的提交。
4. 消费者在生产者增加计数前完成预取，造成pending永远为1。
5. 部分Prefill提交后未置dirty，剩余472 tokens仍保留在long桶。
6. HTTP退出与普通请求发布缺少原子顺序，导致请求可能排到None后。

以上边界已分别以确定性回归或实际入口检查核对。未发现当前审查范围内尚未处理的阻塞问题。

## 验证证据

- 修改前独立副本：完整CPU测试为1 failed、156 passed、3 skipped。
- Codex最终聚焦回归：**83 passed、2 skipped、3 subtests passed**。跳过项为CUDA上下文检查，独立审查限定CPU。
- 最后关闭入口变更：直接回归 **9 passed**（6个关闭入口用例、3个IPC检查）。未再次运行全量测试。
- 两项独立边界复查通过：预取完成后pending=false；实际engine进度提交后600→472 tokens，dirty=true、category=mid。
- 已完成的一次关键CPU入口检查：实际Target控制/KV路径中，A前向尚被Event阻塞时，B通过预取路径；A/B输出正确，所有前向在同一禁梯度执行线程，退出后无残留工作线程。未重复运行该smoke。
- `git diff --check`通过，测试前后所审查源码hash一致。

已知基线失败是：

`tests/test_speculative_sampling.py::SparseRejectionSamplingTests::test_first_middle_and_last_rejections_stop_at_the_first_row`

错误为`target probability rows must include one correction row`，在原始提交的独立副本中已复现。本分支没有改动该采样逻辑，不能将全量测试宣称为全绿。

详细命令、真实退出码、源码hash及各轮反馈保存在 `tmp/kvpre-handoff/`：

- `TASK.md`、`IMPLEMENTATION.md`
- `REVIEW_ROUND1.md`、`REVIEW_ROUND2.md`、`REVIEW_FINAL_GATE.md`
- `baseline-full.json` / `.log`
- `review-final-tests.json`、`review-final-tests.log`
- `review-gate-final.json` / `.log`
- `review-probe-final-results.json`、`engine-smoke-results.json`

## 验证边界

这里只验证队列清理、前向期间的预取触发、响应/退出顺序与缓存并发保护。CPU检查中的搬运是逻辑路径验证，不能据此宣称已证明真实CUDA拷贝与计算重叠或吞吐提升。未重跑32B实验；按照用户要求只保留关键smoke，后续检查限直接相关的单元回归。
