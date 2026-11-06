# FP32、多核、γ100 / λ10 的32草稿模型实验

本轮是独立新实验，旧FP16运行的部分结果保留在原目录，不混入本轮统计。

- cpunode1/cpunode3 各16个独立Qwen3-0.6B CPU模型，全部FP32；每个实例使用多个物理核，线程数不超过其CPU集合大小。CPU集合互不重叠，位于单个NUMA节点内，实际分配保存于各节点worker-map和运行元数据。
- gpunode2空闲GPU运行Qwen3-1.7B FP32，完整驻GPU，关闭TF32。实际物理GPU/UUID以manifest.json为准。
- 标准SD、FastSD、SpecEdge单链三组，每组完整GSM8K test1319题，共3957个正式请求。非thinking模板、greedy、EOS、输出上限1024tokens。
- 固定γ=100，关闭gamma自适应；全局泊松λ=10请求/秒，种子20261024，与旧实验lambda10使用相同题目、分片和时间表。最后计划到达约130秒。
- batch_size=8、原生token_budget=2048、paddedKV；SpecEdge静态max_len=3072，主链与主动草拟均为单链，FP32模型和FP32 mask传输。

FP32用于CPU和GPU两端。只调整精度、CPU资源分配和实验参数，保留kvpre推理/预取逻辑及既有概率历史处理方式。线程与CPU集合由实际允许的CPU、cgroup配额、物理核和NUMA拓扑计算；固定32个模型同时存在，不通过减少模型数量加速。

到达时间为全局时间表，32个实例静态轮转领取题目。实例忙碌时任务等待；arrival_ttft_ms包含draft排队，service_ttft_ms从实际开始服务计时，两者分别报告。TPOT是首批验证输出可见后至完成的摊销间隔。输入时间表不等同于云端Verify请求的发送时间表。

记录KV预取提交、完成、实际搬运、失败、验证时驻留命中、搬运字节与主机墙钟耗时。此耗时包括原有阻塞复制/流等待，不作为纯PCIe时间或CUDA重叠证据。

按选定物理GPU UUID以100ms采样板卡功率，积分首个计划到达到最后完成的区间，排除模型加载。保存总J/Wh、平均W、J/请求及J/输出token。三种方法统一使用同一采样口径。

tmux exp中的cpunode1、cpunode3、gpunode2三个窗口分别运行两端CPU控制器及GPU总调度器。CPU原始逐轮日志保留在各自/home/zhangh/workspace/fastsd/experiments/1024-fp32-g100-l10-cpu32，结果持续上传至GPU端本目录。控制和推理均只监听loopback，复用本任务SSH反向转发28634/28635，控制端有认证。

只做必要配置、资源边界、FP32消息编解码与能耗积分检查；不追加模型smoke。直接观察正式请求确认两端FP32和多核配置。

status.json是实时状态；RESULTS.zh.md/results.json/results.csv按组和周期更新。三组全部完成后自动核对全量覆盖、输出一致性并生成comparison.png/pdf。失败会保留现有结果并停止推进。
