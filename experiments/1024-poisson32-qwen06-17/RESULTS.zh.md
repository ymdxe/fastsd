# 实验进度与结果

Qwen3-0.6B CPU draft：node1/node3 各16个；Qwen3-1.7B FP16 Target：gpunode2 GPU1；完整 GSM8K test 1319题，EOS/1024-token 上限。
全局泊松 λ=50 请求/秒，γ=200/500，三种方法共6组。尚未完成的组不作为最终结果。

| Run | 状态 | 完成 | 含排队 TTFT(s) | 服务 TTFT(s) | TPOT(ms) | GPU能耗(J) | 预取搬运 |
|---|---|---:|---:|---:|---:|---:|---:|
| fastsd-g200-l50 | failed | 21/1319 | 239.091 | 139.694 | 10128.587 | — | 43 |
| standard-g200-l50 | pending | 0/1319 | — | — | — | — | — |
| specedge-g200-l50 | pending | 0/1319 | — | — | — | — | — |
| fastsd-g500-l50 | pending | 0/1319 | — | — | — | — | — |
| standard-g500-l50 | pending | 0/1319 | — | — | — | — | — |
| specedge-g500-l50 | pending | 0/1319 | — | — | — | — | — |

能耗只积分首个计划到达至最后一个请求完成的区间，排除模型加载。KV 搬运时间为原有阻塞式 `.to()` 的主机墙钟耗时，不能当作纯 PCIe 传输或 CUDA 重叠证据。
SpecEdge 无原生异步 KV 预取机制，相关指标标为不适用；同步缓存搬运与整卡能耗仍计入其流程。标准SD的预取计数也不代表另加了预取实现。
TTFT 是首批验证输出可见时间；TPOT 是首批可见输出之后的摊销 token 间隔。到达后的 draft 排队时间单独保存，并加入 arrival_ttft_ms。
每个参数组合一次完整运行；共享 CPU 的负载记录保留在节点 status/preflight 中。SSH隧道结果不代表IB网络性能。
