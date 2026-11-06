# 独立只读审查

审查者：协作子代理 `/root/code_semantics_8b`；范围为 FrameDecoder、Runtime.pump、Hub.authenticate、Agent.slot 与后续修正。

最终意见：未发现阻断性背压或单向关闭错误；EOF顺序、组合缓冲预算、DATA尺寸、EOF后数据拒绝、Hub/Agent raw/channel方向与版本绑定符合设计。

审查提出的两项问题已处理并测试：

- 认证 JSON 必须是 object。Hub/Agent 均显式检查，避免 Agent.slot 因 AttributeError 退出并永久缩小池；新增非object challenge线程保持测试。
- raw.shutdown 只允许 ENOTCONN 并记录已经关闭；其他错误记录 raw_shutdown_errors 并进入失败路径，不能静默宣称成功。

审查确认 lifetime 默认48h、最大21天+1h与父任务新的生命周期要求一致。审查不是生产负载性能证明；部署依据还包括 Linux 测试与四节点探针，见 validation.json。
