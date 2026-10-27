
## 2026-10-23 FastSD 32 draft 跨机通信复测（node1/node3）

- 沿用 Standard SD 的 Qwen3-1.7B FP16、GSM8K 前32条、16-token 上限和 CPU/NUMA 映射。开启 FastSD custom 调度、流水线、提前草拟、延迟优先级与自适应gamma（初始4）。
- 首轮 1023-fastsd32 仅18/32成功：10个负pull时长错误、4个HTTP422；主进程退出1。保留完整失败日志和负upload校验诊断。
- 在实验适配器中依次执行原生五探针校准，再以屏障统一开始请求；未改原仓库推理源码、未关闭FastSD功能或时间校验。第二轮 1023-fastsd32-r2 全部32/32成功，512输出token与Standard SD逐题一致，288次协议RPC无错误，提前草拟复用命中192轮。
- 请求E2E中位数179.51s（基线202.89s）；TTFT中位数150.37s（基线173.24s）；估计总吞吐1.964 tokens/s（基线1.725）。verify HTTP中位数508.80ms、P95 7169.05ms，高于基线；不将短输出、不同背景负载的单次结果作为正式加速比。
- Target调度证据：111个计划、192个verify切片、33个prefill切片；CPU上的预取计数不代表GPU传输收益。最大校准不确定度2.263ms。
- 两机目录：/home/zhangh/workspace/fastsd/experiments/1023-fastsd32{,-r2}/；node2仓库experiments同名目录保留结果副本。启动命令、环境、源码SHA、原始日志、输出、汇总及实验适配器diff均已归档。
- 所有draft已退出，Target通过/exit保存调度统计后正常停止（退出0）；tmux会话6保留。未安装依赖或修改网络、权限。
