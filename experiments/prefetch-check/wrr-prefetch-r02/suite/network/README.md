# 服务器反向 TCP 中转：framed-v2

本目录是新的隔离实现。已归档和正在运行的 raw a/b 包未修改。只有 GPU 主动连接各 CPU 的 Hub，数据不经过桌面或 SSH 转发。此次协议加入显式 DATA/EOF，避免把跨节点 TCP FIN 是否被观察到作为逻辑请求结束的唯一依据；历史 FIN_WAIT2 根因仍未完全定位。

## 固定端点

| 位置 | 地址/端口 | 用途 |
| --- | --- | --- |
| CPU1 / CPU2 / CPU3 Hub | 10.66.0.3 / 10.66.0.1 / 10.66.0.2，端口 19661 | GPU 发起认证通道 |
| 每台 CPU 本地 | 127.0.0.1:29659 | HTTP/coordinator：FastSD、标准SD、协调与计时 |
| 每台 CPU 本地 | 127.0.0.1:29660 | model：SpecEdge gRPC |
| GPU 本地后端 | 127.0.0.1:19659 / 19660 | HTTP / gRPC |

CLI 不提供任意后端目标或端口覆盖。程序内部端口覆盖只用于独立 port0 测试。生产配置仍是每 Agent `--channels 32 --control-channels 64`，池上限各 64，Hub 每服务队列上限 128。

## 启动接口

父运行器提供 `PY`、`SUITE`、`STATE`、`AUTH` 的绝对路径；`AUTH` 只包含本轮认证文件路径。Linux 认证文件必须是当前 UID 所有的普通文件，禁止符号链接与 group/other 权限（新文件 mode 0600）；去掉首尾空白后 32–256 字节，文件读取上限 4096 字节。程序不创建认证文件、不修改权限、不打印认证内容。

每个 CPU 先启动 Hub，`CPU_IB` 分别选 10.66.0.3、10.66.0.1、10.66.0.2：

```sh
"$PY" -B -u "$SUITE/network/reverse_relay.py" hub --bind-ip "$CPU_IB" --auth-file "$AUTH" --status-file "$STATE/hub-status.json" --lifetime 1818000 --idle-timeout 900
```

GPU 再为每个 CPU 启动一个 Agent，`NODE` 分别为 node1/node2/node3：

```sh
"$PY" -B -u "$SUITE/network/reverse_relay.py" agent --hub "$CPU_IB" --channels 32 --control-channels 64 --auth-file "$AUTH" --status-file "$STATE/agent-$NODE-status.json" --lifetime 1818000 --idle-timeout 900
```

`--lifetime` 默认 172800 秒（48小时），允许 30–1818000 秒；显式最大值为父运行器授权的 21 天 + 1 小时。结束/失败由父运行器清理已登记的本任务 PID，生命周期上限只作为最后保护。`--idle-timeout` 默认 900 秒，允许 10–3600 秒。

所有 6 个进程必须使用同一 framed-v2 实现。raw v1 与 framed-v2 明确拒绝混用；不可只替换一侧。认证保持 HMAC-SHA256，并绑定 framed-v2 版本前缀、nonce、service。该认证不是传输加密。

## 生命周期与观察项

放行任务前确认每个 Agent：`protocol_version=2`，`model_ready_channels=32`，`coordinator_ready_channels=64`，`active_connections=0`。成功方法结束后应恢复这些数值，并确认 Hub 同样 active 0、队列完整；不要把残留连接当作已回收。

状态每约 2 秒原子更新。关键计数为 `protocol_errors`、`relay_errors`、`raw_shutdown_errors`、`authentication_failures`、`channel_acquire_failures`、`backend_connect_failures`、`idle_timeouts`、`completed_connections`、`eof_frames_sent/received`、`active_threads`、`threads_reaped`、`inbound_buffer_peak/outbound_buffer_peak`。`raw_shutdown_already_closed` 仅记录本地 raw socket 的 ENOTCONN，其他 shutdown 错误显式失败。启动前健康探测造成的 backend 失败须单列基线，不能删除。

清理顺序：停止请求并等待收尾，按登记 PID 校验 UID、启动时间、源码路径和 role/IP，先停 GPU Agent，再停 CPU Hub。只操作本任务进程；不使用宽泛 pkill，不删除历史证据。SIGTERM/SIGINT/SIGHUP 或 lifetime 到期会关闭登记 socket、等待线程、写最终状态。

## 已验证范围

- GPU Linux：12 项 framed 协议测试 + 6 项固定端点、私密文件、生命周期、线程回收和源码来源检查全部通过。
- 四节点独立 port0 探针：每 CPU 两种服务各 4 个请求，共 24 个，大小 1B/4KiB/64KiB/256KiB；响应完整，所有 Hub/Agent active 归零、通道池恢复，协议/relay 错误 0，全部探针进程退出。
- 跨节点探针使用缩小池 model 2/control 4，属于功能与回收验证；不是生产 32/64 池的满负载性能测试，也未加载模型。
- 独立子代理只读审查未发现阻断问题，见 `REVIEW.md`。完整协议见 `PROTOCOL.md`，证据摘要见 `validation.json`。

结果必须注明“使用 IB 地址的服务器反向 TCP 中转（framed-v2）”；不能标成原生直连 IB，也不能用短探针证明中转开销可忽略。
