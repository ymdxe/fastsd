# node3八个0.6B Draft与GPU一个8B Target：FastSD / SpecEdge

用户确认：仅cpunode3，8个独立Qwen3-0.6B FP32 Draft，每个8线程，固定gamma=8；Qwen3-8B BF16 Target位于gpunode2的一张空闲RTX A6000。每Draft独立泊松率0.5请求/秒，10个正式请求，总计80个MT-Bench第一轮输入，每条最多32 tokens，greedy、关闭thinking、EOS计入输出。每worker一条预热不计入结果。两方法重放相同输入和到达计划。

FastSD保留pipeline、proactive、prefill overlap和latency priority，关闭动态gamma。批容量8，token budget512。SpecEdge沿用已固定官方源码及现有CPU适配器，8个客户端、最大批8、tree budget32、最大beam length8、proactive included；其gamma表示树深度上限，与FastSD线性draft长度不完全等价。

网络为桌面中转SSH，CPU HTTP29635/gRPC29636到GPU19635/19636，本轮不测IB直连。GPU包在/home/hdd3/zhangh/workspace/fastsd/node3-pair-20261027；CPU包在/home/zhangh/workspace/fastsd/node3-pair-20261027。结果为各自仓库exp/node3_pair_20261027/<run_id>，CPU再分node3/worker0..7。

手动执行：转发就绪且GPU空闲后，GPU运行 bash <GPU包路径>/launch.sh server fastsd load05 <新run_id> <GPU编号>；model_ready后在CPU运行 bash <CPU包路径>/launch.sh clients <同一run_id>。SpecEdge将method换为specedge_cpu_adapted并使用另一个run_id。本轮仅期待node3的8个worker。

完成条件：80/80请求、measurement_valid=true、8个worker退出0、server正常退出、cache-final落盘。修复说明见CLOCK_RECOVERY.zh.md。单次短输出结果不等于重复性能结论；输出一致性单独核对。
