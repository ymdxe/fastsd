# 本轮服务器反向 TCP 中转

用途：三台 CPU 各启动一个 Hub，GPU 各启动一个对应 Agent；只有 GPU 主动建立跨节点 TCP。数据不经过桌面或 SSH 转发。用户已授权本轮服务器反向 TCP 中转。称为“使用 IB 地址的服务器反向 TCP 中转”，不能把结果标成原生直连 IB 性能。

## 固定端点和接口

| 端点 | 地址与端口 | 用途 |
| --- | --- | --- |
| CPU1 Hub | 10.66.0.3:19661 | GPU 建立认证通道 |
| CPU2 Hub | 10.66.0.1:19661 | GPU 建立认证通道 |
| CPU3 Hub | 10.66.0.2:19661 | GPU 建立认证通道 |
| 每台 CPU 本地 HTTP | 127.0.0.1:29659 | FastSD/标准 SD HTTP RPC、协调与计时 |
| 每台 CPU 本地 gRPC | 127.0.0.1:29660 | SpecEdge 推理 |
| GPU 固定 HTTP 后端 | 127.0.0.1:19659 | 映射 coordinator 服务 |
| GPU 固定 gRPC 后端 | 127.0.0.1:19660 | 映射 model 服务 |

程序入口：`python -u network/reverse_relay.py {hub,agent}`。

- 必需 `--auth-file PATH --status-file PATH`；Hub 必需 `--bind-ip CPU_IB`，Agent 必需 `--hub CPU_IB`。
- `--lifetime 172800`：默认及上限均 48 小时，下限 30 秒。生命周期从进程启动计时，并非每方法重置；若整轮可能超过 48 小时，应由套件在方法之间有序重建。不得在仍处理请求时重建。
- `--idle-timeout 900`：默认 900 秒，允许 10–3600 秒；有数据转发的连接会更新活动时间。
- `--channels 32`：每 Agent 的 gRPC/model 通道池；默认 32，允许 1–64。
- `--control-channels 64`：每 Agent 的 HTTP/coordinator 通道池；默认 64，允许 1–64。本轮三台 CPU 一律使用 64。HTTP 池从原脚本上限 32 扩至 64，是为了覆盖 CPU3 的 16 Draft × 主/计时 Session 外的协调连接，避免中转池人为成为瓶颈。
- 无可修改后端目标/服务端口的 CLI；GPU 后端只连 loopback，CPU 推理/协调入口也只监听 loopback。
- Hub 每服务空闲通道队列仍限制 128，socket listen backlog 仍为 128。

## 认证文件

操作者或父套件为本轮提供认证文件；程序不会创建凭据、修改权限或打印认证内容。各 Hub 与其 Agent 必须读取相同字节（去掉首尾空白后 32–256 字节）。Linux 上必须为当前 UID 所有的普通文件，不得是符号链接，也不得有 group/other 权限（例如新建文件 mode 0600）。整个文件读取上限 4096 字节。

认证保留原随机 nonce + HMAC-SHA256 + constant-time 比较。认证内容不在命令参数、状态 JSON 或日志中；命令参数仅包含文件路径。认证过程不提供传输加密，仍只应用于获准的服务器网络和本轮任务。不得把认证文件加入源包、结果包或提交。

## 启动命令

以下为远端 shell 命令模板：`SUITE` 是父套件部署的绝对目录，`STATE` 是该节点本轮专用状态目录，`AUTH` 是预先提供的私密认证文件绝对路径。三者由父套件设定；不从源码或文档生成 secret。每个后台进程的 PID 应由父运行器登记，并保留对应 role/IP、启动时间、完整源码路径，以便只清理本轮进程。

先启动三个 CPU Hub（在对应节点运行一行）：

```sh
# cpunode1
/home/zhangh/miniconda3/envs/fastsd/bin/python -u "$SUITE/network/reverse_relay.py" hub --bind-ip 10.66.0.3 --auth-file "$AUTH" --status-file "$STATE/hub-status.json" --lifetime 172800 --idle-timeout 900
# cpunode2
/home/zhangh/venvs/fastsd_cpu/bin/python -u "$SUITE/network/reverse_relay.py" hub --bind-ip 10.66.0.1 --auth-file "$AUTH" --status-file "$STATE/hub-status.json" --lifetime 172800 --idle-timeout 900
# cpunode3
/home/zhangh/envs/fastsd_cpu_py314/bin/python314_glibc -u "$SUITE/network/reverse_relay.py" hub --bind-ip 10.66.0.2 --auth-file "$AUTH" --status-file "$STATE/hub-status.json" --lifetime 172800 --idle-timeout 900
```

再在 gpunode2 分别启动三个 Agent（父运行器以独立后台进程托管各行）：

```sh
/home/hdd/zhangh/envs/specedge/bin/python -u "$SUITE/network/reverse_relay.py" agent --hub 10.66.0.3 --channels 32 --control-channels 64 --auth-file "$AUTH" --status-file "$STATE/agent-cpu1-status.json" --lifetime 172800 --idle-timeout 900
/home/hdd/zhangh/envs/specedge/bin/python -u "$SUITE/network/reverse_relay.py" agent --hub 10.66.0.1 --channels 32 --control-channels 64 --auth-file "$AUTH" --status-file "$STATE/agent-cpu2-status.json" --lifetime 172800 --idle-timeout 900
/home/hdd/zhangh/envs/specedge/bin/python -u "$SUITE/network/reverse_relay.py" agent --hub 10.66.0.2 --channels 32 --control-channels 64 --auth-file "$AUTH" --status-file "$STATE/agent-cpu3-status.json" --lifetime 172800 --idle-timeout 900
```

以上假定本轮所有配对使用同一私密 AUTH；若采用每 CPU 一个认证文件，三个 GPU Agent 的 `--auth-file` 分别选对应文件即可。正式运行前必须先确认后端服务健康与通道池就绪，再放行边缘任务。

## 监控与结束

状态 JSON 每约 2 秒原子替换：Hub 有 `listeners`、`available_channels`；Agent 有 `pool_sizes`、`model_ready_channels`、`coordinator_ready_channels`；通用计数有 `open_sockets`、`active_threads`、`threads_reaped`，运行后按事件出现 `active_connections`、`relay_errors`、`channel_acquire_failures`、`backend_connect_failures`、`authentication_failures` 等。

- 放行任务前，确认 Hub 地址正确，认证失败为 0，各 Agent 空闲池可达到 model 32 / coordinator 64。服务池在请求处理时减少属于正常现象。
- 运行中记录队列耗尽、连接获取失败、后端失败、idle timeout 和转发错误；这些会影响方法比较，不能忽略。
- `Runtime.threads` 在每次 spawn、快照与状态刷新时清理已退出引用；活跃线程保留并在 shutdown 时等待（总等待上限 5 秒）。
- SIGTERM/SIGINT/SIGHUP 或 lifetime 到期会设置停止标记，关闭已登记 socket、等待线程、写最终状态。启动日志额外含 PID 和 lifetime；认证内容从未包含在日志中。
- 清理顺序：等待本轮请求全部收尾，停止 GPU Agent，再停止 CPU Hub。由父运行器对已登记 PID 校验当前 UID、启动时间、源码绝对路径与 role/IP 后仅发送 SIGTERM；等待退出并确认本轮监听端口释放。不得按宽泛进程名 pkill，不得处理其他用户或历史任务的进程。不要删除认证/日志/结果以“清理”证据。

## 测量限制与来源

每个方向的待发缓冲阈值仍为 4 MiB，单次 recv 最多 256 KiB，因此瞬时可略超过阈值。两方向合计每连接最多约 8.5 MiB 待发 payload，32 个并发连接在每个 relay 进程中理论可接近 272 MiB（不是预先分配）。同一任务可能同时有主 RPC、计时和协调连接；连接数不等于 Draft 数。池扩容有上限但并不保证 relay 无性能影响，需要正式 smoke 和进程 CPU/RSS/失败计数作证。

原脚本来源、原 SHA256、新 SHA256、逐个保留函数的 AST SHA256 位于 `relay-provenance.json`。原完整文件保留于本地 `E:/Codex/temp/fastsd-8b-full-poisson15-20261030/network/reverse_relay.previous.py`，没有覆盖。

`validate_relay.py` 只使用公开内存 fixture，不监听网络、不创建认证文件、不生成 secret。执行：

```sh
python -B network/validate_relay.py
```

此次本地六项验证全部通过，包含固定端点、48h/连接池边界、认证文件模式与属主模拟、256 个并发短线程加一个存活线程的引用回收、日志快照不含认证内容，以及原认证/转发关键函数 AST 完全一致。未启动正式 relay，未完成正式端到端或长时间网络性能验证。
