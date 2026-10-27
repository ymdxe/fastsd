# 16-draft 异构实验结果

三种方法均完成 **16/16** 请求，合计48条正式请求，无RPC错误。

node1、node3各8个独立Qwen3-1.7B FP16 CPU draft，每个绑定一个物理核、一个计算线程；node1固定γ=100，node3固定γ=20。
Target为gpunode2的Qwen3-32B FP16，物理GPU3（48GB A6000）+CPU权重卸载，GPU权重预算34GiB。三种方法的权重设备映射相同。
同一批GSM8K前16题，偶数索引分配node1，奇数索引分配node3；non-thinking模板、greedy、EOS、256-token上限。每方法一次，正式请求不含加载和冒烟。

## 核心结果

以下均为逐请求均值；完整中位数、P95、E2E见[RESULTS.md](RESULTS.md)及[CSV](results.csv)。

| 方法 | 分组 | TTFT均值（秒） | TPOT均值（毫秒/token） | 输出tokens |
| --- | --- | ---: | ---: | ---: |
| Standard SD | 整体 | 163.79 | 5257.37 | 3723 |
| Standard SD | node1 / γ100 | 110.77 | 5284.69 | 1997 |
| Standard SD | node3 / γ20 | 216.81 | 5230.05 | 1726 |
| FastSD | 整体 | 218.01 | 7461.86 | 3723 |
| FastSD | node1 / γ100 | 93.64 | 5349.17 | 1997 |
| FastSD | node3 / γ20 | 342.38 | 9574.55 | 1726 |
| SpecEdge 单链 | 整体 | 278.29 | 10832.31 | 3714 |
| SpecEdge 单链 | node1 / γ100 | 125.17 | 8228.71 | 1997 |
| SpecEdge 单链 | node3 / γ20 | 431.42 | 13435.91 | 1717 |

| 方法 | 整体吞吐（tokens/s） | 统一计时窗口（秒） | 候选接受率 |
| --- | ---: | ---: | ---: |
| Standard SD | 1.670 | 2229.62 | 17.18% |
| FastSD | 0.891 | 4178.87 | 17.20% |
| SpecEdge 单链 | 0.856 | 4338.13 | 15.79% |

TTFT为请求开始至首个已验证输出块可见；TPOT=(E2E−TTFT)/(输出tokens−1)，是块输出的摊销间隔，包含剩余proactive计算的收尾。整体吞吐使用协调器单一时钟，从模型就绪屏障至全部worker完成，包含后续启动/校准，不跨主机相减monotonic。

## 输出核验

- FastSD与Standard SD：**16/16**逐token完全相同。
- SpecEdge与Standard SD：**15/16**逐token完全相同。node3的`gsm8k-0003`从第25个输出token起不同，输出152/143 tokens；两者都算出540，但措辞不同。根因尚未定位，不能声称全部输出完全对齐。
- 该题完整输出和参考答案保存在[output_review.json](output_review.json)。本实验并未进行全量GSM8K准确率评测。

## 解释范围

node3三轮的平均load1分别为：Standard SD 4.26、FastSD 111.55、SpecEdge 33.68。负载包含实验与其他任务，三轮顺序运行且负载变化明显，**本轮耗时差不能单独归因于算法**。
node1负载较接近：FastSD平均TTFT为93.64秒，低于Standard SD的110.77秒；平均TPOT为5349.17毫秒，略高于5284.69毫秒。该组同样只有8个请求、一次运行。
FastSD记录了308个调度计划，但异步KV预取提交/完成/失败均为0，本轮没有触发该机制。Native调度计数涵盖Target整个生命周期（含独立冒烟），正式延迟和RPC统计已排除冒烟。
SpecEdge使用官方固定版本的CPU单链适配，保留主草拟、提前草拟、gRPC传输、Target验证、接受/回滚全流程；主链及proactive分支宽度均为1。实际服务端验证γ100共168次、γ20共258次。两端FP16、相同输入模板、相同输出截断；Target batch=8，但SpecEdge没有与Native token_budget=512对应的参数。详见[SpecEdge适配说明](specedge/README.md)。

## 图与数据

- [TTFT/TPOT分布图](latency_comparison.png) / [PDF](latency_comparison.pdf)
- [各节点完整负载轨迹](host_load.png)
- [完整统计JSON](results.json) / [CSV](results.csv)
- [冻结配置](manifest.json)、[tokenizer核验](tokenizer_check.json)、[进程退出核验](postflight.json)

原始记录位于`standard/`、`fastsd/`、`specedge/`的各node目录，保留逐请求、逐RPC、CPU核映射、连续负载、完整输出、服务端日志和启动命令。
