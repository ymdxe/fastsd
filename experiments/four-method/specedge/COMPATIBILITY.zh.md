# SpecEdge CPU 适配说明

- 方法名称：`specedge_cpu_adapted`。官方算法来源为 kaist-ina/specedge，固定提交 `1edcaf02ffc41a7b57726450c5357ed216a3b9bc`。
- `official/` 为字节一致的官方源码副本，未修改模型、Tree、SpecExec、proactive、CUDA Graph 或 Target 验证算法。`official-manifest.json` 记录源文件 SHA256。
- `integration/` 保留项目现有 CPU engine、CPU timing、精确输出长度、tree compaction 顺序和 BF16 wire 适配。上述适配路径不能标为“未修改的官方 GPU 部署”。
- 新增 worker 读取每个流独立分片。全局client编号由协调器 expected_workers 顺序确定；CPU1八个0.6B、CPU3八个1.7B、每个8线程FP32；gamma4/8分别映射官方 max_beam_len4/8。tree budget32、max_n_beams32、branch width16，proactive included、proactive depth3。
- EOS 规则只使用 `tokenizer.eos_token_id`（本轮Qwen3为151645），不把 generation_config 中额外151643作为停止条件；每题最多32 token。
- 单个8B BF16 Target运行官方 InferenceController 和 BatchGraphEngine。官方每client GPU KV存储与工作KV保持常驻，FastSD自身KV卸载/预取策略不同；报告必须公开这种实现差异。
- max_len默认2048；固定相同输入和输出上限，运行前要求prompt tokens+32+2*tree_budget严格小于max_len。不足时须同时更新服务端与客户端，禁止静默截断。
- Target真实完成权重加载及CUDA Graph capture后写 `specedge-ready.json`；仅gRPC端口打开不能证明模型就绪。失败写 `specedge-failed.json`。新增servicer将官方硬编码5秒RPC等待延长至1800秒，并清除完成future，保持同一验证队列和InferenceController。
- 先加载全部16 CPU模型、官方Sync、每client一条预热，再进入共同READY/T0。计时以各client单调时钟为准，逐token可见时刻与到达排队均记录。原版client_host的固定15秒启动延迟不用于本套件。

## 解释器与依赖

|节点|解释器|已观测依赖|
|---|---|---|
|gpunode2|`/home/hdd/zhangh/envs/specedge/bin/python`|Python3.14.7、torch2.9.0+cu128、transformers4.57.1、grpc1.75.1|
|cpunode1|`/home/zhangh/miniconda3/envs/fastsd/bin/python`|Python3.11.17、torch2.6.0+cpu、transformers4.57.6、protobuf5.29.6；grpcio1.75.1待安装|
|cpunode3|`/home/zhangh/envs/fastsd_cpu_py314/bin/python314_glibc`|Python3.14.7、torch2.9.0+cpu、transformers4.57.1、grpc1.76.0|

CPU3不能直接调用py314/bin/python，它需要已有的glibc启动器；旧torch2.2.1环境没有torch.uint16，无法满足现有BF16 wire路径。CPU1仅需grpcio1.75.1，protobuf5.29.6已满足生成代码>=5.29.0。`bootstrap.py` 对Python3.11只在内存中将官方util.py的一个含反斜杠日志f-string改成等价chr(10)表达式；官方磁盘源码保持原样，算法不变。

依赖预检（不加载模型、不初始化CUDA）：`<python> specedge/validate.py --imports`。Target入口：`<GPU python> specedge_server.py --config <run-config.json> --host 127.0.0.1 --port 19636`；worker由suite统一调度。CPU端gRPC使用独立SSH转发127.0.0.1:29636到GPU127.0.0.1:19636，协调HTTP继续29635。

本次仅完成代码准备、语法与模拟契约检查；真实模型正确性和性能以实际运行结果为准。