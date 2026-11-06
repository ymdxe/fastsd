# 本轮有限时钟恢复规则

2026-10-26（America/Denver；服务器为2026-10-27）。修复位于隔离的 four-method-suite，不修改原仓库模型代码、模型、方法开关、γ或100ms时钟不确定度门槛。

负 pull：保存首次已经收到的响应和首次 receipt epoch，只重新校准并重算该响应的单向估计，不重发生成或验证 RPC。每个异常最多校准3次；每次仍须通过原不确定度门槛，失败或3次后仍负则明确终止。

负 upload：只有 `/verify` 在调用 `_enqueue_and_wait` 之前返回的结构化 `negative_upload_pre_enqueue`、`route=/verify`、`enqueued=false` 三项同时匹配，才允许有限重校准并重发。session、task、prefix、token、γ、seed和其他模型字段保持相同，只刷新发送 timestamp 与 offset。最多3次校准、4次上传；其它422、worker错误、超时和断连均不重试。若同一次逻辑 Verify 先后出现 upload 与 pull 两种异常，两种恢复各最多3次，因此理论上合计最多6次校准。

所有校准 GET 纳入应用层 RPC/响应字节计数；GET请求体为空，因此该口径下请求体字节数为0。校准失败响应也计数；epoch时钟倒退形成的负RTT直接失败并保留样本。拒绝的上传 POST 也计入原HTTP计数。

请求 E2E、首token等待和轮次等待都包含真实恢复开销。用于 transport RTT 的 HTTP时长只保留最后成功一次，失败POST和校准成本另外记录，不伪装成正常传输时延。每请求记录 clock_sync_rpc_count、clock_sync_ms、clock_recovery_count、clock_pull_recovery_count、clock_upload_recovery_count、clock_recovery_ms、clock_rejected_upload_http_ms；summary按节点和总组汇总。预热后只清零正式统计，预热恢复事件仍保留。

每个worker的 `clock-recovery.jsonl` 保存成功与失败事件、原始负值、offset、不确定度、原始 receipt/send 时间、所有校准样本和耗时。worker日志打印有界 CLOCK_RECOVERY 摘要。恢复后的 pull 是事后校准且以非负为停止条件的估计，存在选择偏差，不作为无偏或精确单向时延结论。禁止把负值截断成0，禁止使用RTT/2替代单向时延。

真实模型和性能是否成功以新run的实际记录为准。旧失败或中断目录保留，不与新run拼接。
