# 首组HTTP连接断开与恢复

首轮FastSD γ=4、λ=1在完成162/1319后停止。cpunode1 worker-08首先在POST /verify中出现RemoteDisconnected，随后总调度关闭Target，node3的503是关闭后的连带错误。Target正常退出，日志没有GPU异常或显存溢出。

worker-08最后成功响应时间为1792834480.2231085，异常日志更新时间为1792834486.3947182，间隔约6.17秒。当前安装的Uvicorn默认HTTP空闲保活为5秒，到期会关闭连接。这与CPU草拟间隔接近连接过期时的竞态相符；没有网络抓包，因此不把它表述为已逐包确认的唯一原因。

实验入口通过transport_config.py显式设置HTTP空闲保活7200秒，标准SD与FastSD共用。保留状态性POST不自动重试的行为，避免在响应丢失时重复推进KV状态。推理算法、gamma、模型、线程和到达时间表不变。

仅增加一个关键传输检查：真实Uvicorn服务使用同一TCP连接，空闲6.2秒后继续POST；确认连接与客户端端口一致，检查通过。没有加载模型或追加模型smoke。

失败运行完整保存在attempts/fastsd-g4-l1-http-disconnect；该目录含原始请求、功率采样、目标日志，以及原配置/打包文件。CPU节点保留同名归档及更新前代码。新的正式运行从第0题开始，所有方法仍各测完整1319题，不拼接失败样本。

diagnostics/node1.json、node3.json保存错误证据；keepalive-check.json保存检查结果。当前status.json及RESULTS.zh.md只报告重新开始后的正式运行。
