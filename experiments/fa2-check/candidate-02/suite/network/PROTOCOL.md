# framed-v2 协议与关闭语义

## 认证与开始转发

Hub 给出 JSON challenge `version=2` 和随机 64 字符 nonce。Agent 回答 `version=2`、白名单 service（model/coordinator）和 HMAC-SHA256。认证消息为 ASCII `framed-v2:<nonce>:<service>`，使用私密文件中的共享字节作为 key；比较为 constant-time。非 JSON object、错误版本、服务或 HMAC 均拒绝。版本同时出现在双向握手并绑定 HMAC，防止 raw/framed 混用。

认证后的 R、请求分配 G、GPU 后端连接成功 C 标记保持不变；在 C 之后，跨节点连接才进入以下 framing。固定后端仍只连 GPU loopback，不允许对端指定任意地址。

## 帧格式

每帧 5 字节 header：1 字节 type，加 4 字节网络字节序 unsigned length (`!BI`)。

| type | length | 含义 |
| --- | --- | --- |
| 1 / DATA | 1..262144 | 原样应用 payload |
| 2 / EOF | 必须 0 | 此逻辑方向不会再发送 payload |

任意 header/payload 分片、一次读取多个帧均可解析。先验证类型和长度，再等待对应 payload；拒绝未知类型、空 DATA、超限 DATA、带 payload 的 EOF、重复 EOF 和 EOF 后的任意字节。合法 EOF 前物理通道关闭或中途截断为协议失败，不能记为成功完成。

## 双向状态

每侧 pump 连接一个 raw 本地 socket 和一个 framed 跨节点 socket。

1. raw `recv()` 返回 EOF 后，把逻辑 EOF 放在本方向已读 DATA 的后面；它仍须实际从发送缓冲发出。
2. 收到对端逻辑 EOF 后，先把该方向已解码 payload 全部写入 raw，之后才对 raw 执行 `shutdown(SHUT_WR)`。另一方向保持可读写，允许“请求端先半关闭，服务端随后返回大响应”。
3. 逻辑 EOF 后继续监控跨节点读取，直到双方完成，以识别迟到的非法 DATA/重复 EOF。合法 EOF 后出现物理读 EOF 只关闭该运输读方向；它不会替代另一方向的逻辑结束条件。
4. 只有本地 EOF 帧已发完、对端逻辑 EOF 已收到、待写 raw 数据和解析缓冲都为空时成功退出。此时发送缓冲也已为空。无需等待对端 TCP FIN。
5. 帧错误、异常断链、零字节 send 和非 ENOTCONN 的 raw shutdown 错误显式计错并释放 socket。正常单向 EOF 本身不触发提前关闭。

这个机制解决的是“逻辑结束不依赖跨节点 FIN”；并未断言已找到历史网络 FIN/RST 状态不同步的系统原因。

## 缓冲与资源边界

outbound（含 header）最多 4MiB；raw recv 按剩余空间扣除 header 后读取，单次 DATA 不超过 256KiB。inbound 的未解析帧字节与待写 raw payload 合计最多 4MiB，channel recv 同样按剩余空间读取。达到上限时暂停对应读方向，依赖 socket 背压。

这两个上限是逻辑字节预算，不是 RSS 绝对上限：bytearray 扩容、单帧解析拷贝（至多 256KiB）、Python 对象与内核 socket 缓冲另占内存。状态记录两个逻辑 buffer peak，测试确认均未超过 4MiB。线程引用继续按 spawn/status 回收；已完成连接不应永久占用 Agent slot。

## 测试和兼容性

`python -B network/test_framed_relay.py` 覆盖逐字节分片、多帧、边界与畸形长度、截断、EOF 后数据、双向大于4MiB慢读、单EOF后大响应、对端不发送TCP FIN仍能回收、短HTTP连接恢复、旧版本/错误认证/非object认证拒绝和Agent池线程保持。

`python -B network/validate_relay.py` 验证生产固定端口、池范围、48h默认及1818000s显式上限、私密认证文件条件、线程引用回收、日志不包含认证内容，以及跨Python版本稳定的保留源码段hash。历史AST hash只作为记录，不再用于跨版本比较。

本版本与 raw-v1 线协议不兼容。必须由父运行器在方法/任务边界整体替换 3 Hub + 3 Agent，不能修改当前仍在运行的包文件。
