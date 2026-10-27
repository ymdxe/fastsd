
### 2026-10-23
- 固定gamma20双方法复测：node3 32个独立Qwen3-1.7B FP16 CPU draft，node1 8线程同模型CPU Target；GSM8K前32题，统一64-token上限，greedy/EOS。FastSD保留流水线、提前草拟和延迟优先级，关闭gamma自适应，顺序校准后统一开始请求。两方法各32/32成功、各2048输出tokens，完整输出逐题一致；两轮所有128次verify的gamma都为20，无HTTP错误。
  Standard SD/FastSD请求E2E中位数324.44/311.27s；估计输出吞吐4.551/5.135 tokens/s；verify HTTP P95 1.416/12.462s。FastSD调度85个计划，复用命中125轮。主机负载不同，单次短输出观察值不能当作正式加速结论，也不能直接与旧cap16结果比较。
  产物：/home/zhangh/workspace/fastsd/experiments/1023-sd-g20、1023-fastsd-g20；node2本地仓库同名副本。保留逐请求/RPC/worker完成时间、连续负载、源码快照、命令、环境及对比。Git SHA ee0e334e222593c100b2ee6fc7fc9e4db2cf7cd3。未修改原推理源码或安装依赖；临时worker/Target全部正常退出，tmux会话6保留。
