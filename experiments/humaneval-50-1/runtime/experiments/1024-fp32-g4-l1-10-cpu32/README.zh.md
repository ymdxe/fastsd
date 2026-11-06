# FP32、多核、γ=4 / λ=1与10 的32草稿模型实验

两台CPU各16个独立Qwen3-0.6B模型：cpunode1每个2个物理核，cpunode3每个4个物理核；模型均为FP32。Target是gpunode2空闲GPU1上的Qwen3-1.7B FP32，关闭TF32，完整驻GPU。

固定γ=4，全局泊松到达率λ∈{1,10}请求/秒。每个到达率分别运行FastSD、标准SD、SpecEdge单链，共6组。每组完整GSM8K test 1319题，每题EOS或最多1024个输出token，共7914个请求。非thinking模板，greedy，种子20261024。

SpecEdge主草拟及主动草拟都限定为单链：beams=1、branch_width=1、主链draft预算4，主动链预算包含额外bridge。保持官方SpecEdge生成/验证核心和现有CPU适配器。

采用上一轮FP32、多核实验的同一份代码快照与CPU分配，只改变gamma和到达率矩阵。各方法共享相同的题目、分片和全局到达时间表。λ=1的最后计划到达约1298秒，λ=10约130秒。32个worker静态轮转领取题目，忙碌时排队；服务TTFT与包含排队的到达TTFT分别统计。

记录TTFT、TPOT的均值及p50/p95/p99、通信、KV预取搬运/失败/验证需求命中和GPU板卡能耗。能耗以GPU UUID每100ms采样，从首个计划到达到最后完成积分，排除模型加载。KV搬运耗时为包含复制/等待的主机墙钟时间。

CPU原始逐轮日志位于各自/home/zhangh/workspace/fastsd/experiments/1024-fp32-g4-l1-10-cpu32，结果持续汇总到本目录。tmux exp的cpunode1、cpunode3、gpunode2窗口分别运行CPU控制器和GPU总调度器。

旧γ=100实验按用户要求取消，其部分结果保留于原目录，不纳入本轮统计。只做必要配置、消息和资源检查，不额外运行模型smoke。

status.json为实时状态，RESULTS.zh.md和results.json/csv周期更新；6组完成后自动验证7914个请求的覆盖、比较输出并生成图表。失败保留部分记录并停止推进。
