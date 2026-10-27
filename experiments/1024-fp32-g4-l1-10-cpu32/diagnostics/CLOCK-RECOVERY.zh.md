# 第二次中断：时钟测量误差

第二次FastSD γ=4、λ=1完成156/1319，node3 worker-06首先因`synchronized clocks produced a negative pull duration`退出。此时Target已返回有效验证结果；客户端将跨机器epoch时钟换算出的负单向耗时当成致命错误。随后总调度停止Target，node1的503/连接拒绝是连带错误。本次不是HTTP保活断连重现；7200秒保活配置继续保留。

修复限定在实验客户端适配器：每60秒用只读GET重新校准时钟；出现负pull时重新校准并使用原始接收时间计算，禁止重发已经完成的验证。若仍无法可靠测量，保留原始负值及不确定度，标记`pull_timing_valid=false`和`measured_pull_ms=null`。调度只可沿用上次有效估计，没有历史时保持未测得；不会把负值当作0秒实测值。

服务端明确返回422、说明upload时间为负时，该错误发生于入队和KV状态推进之前。只有这一类明确拒绝允许校准后重发，最多两次。连接中断、其他HTTP错误和已完成验证均不自动重发。

单向时钟记录位于CPU各worker的`timing-runtime.jsonl`及`timing-runtime-status.json`；校准、有效/不可靠计时及入队前重试计数持续上传到节点status.json。原始avg_pull_ms可能包含调度估计，不用于精确单向通信对比；TTFT/TPOT由客户端本机单调时钟统计，HTTP通信差值同样使用单调计时。

关键故障注入检查覆盖有效时延、负值校准、仍不可测的标记、没有历史估计、明确入队前拒绝、其他错误禁止重试，以及运行时校准不进入启动屏障。均不加载模型。

该失败尝试将完整保留在attempts/fastsd-g4-l1-clock-calibration。正式实验从头重跑，每组仍为1319题；不同失败尝试不合并成正式结果。
