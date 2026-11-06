# node2 Llama-2-7B 验证记录（2026-09-25）

模型位于 `/home/hdd/zhangh/workspace/llama-varlen-test-20260924/two-gpu-http/Llama-2-7b-hf`，是两片 `safetensors` 权重的 Hugging Face Llama checkpoint（32 层、4096 hidden size、32 个 attention heads、32000 vocab）。本次离线测试没有重新下载模型。使用 `/home/hdd/zhangh/envs/fastsd/bin/python`；`new_fastsd` 环境中的 `huggingface-hub==2.0.0` 与 `transformers==4.57.6` 不兼容，不能用来运行这次测试。

单卡测试在 node2 GPU 2 上成功加载全部权重和 tokenizer，完成 12 token 的真实文本生成。接着运行 `scripts/benchmark_llama_varlen.py`，BF16、1 次 warmup、3 次计时：

| 场景 | padded 中位数 | varlen 中位数 | 此次加速 |
| --- | ---: | ---: | ---: |
| 混合长度 Prefill（16/64/256 token） | 145.26 ms | 78.86 ms | 1.84× |
| 混合长度 Verify（4/6/8 token） | 70.60 ms | 40.43 ms | 1.75× |
| 等长 Verify（6/6/6 token） | 50.54 ms | 39.87 ms | 1.27× |

完整微基准结果在 node2 的 `.../llama-varlen-test-20260924/results/llama7b_node2.json`。这是模型前向微基准，不包括网络、调度与服务启动。BF16 的 padded/varlen 输出并非逐位相同：混合 Prefill 的 argmax 一致率为 98.81%，混合 Verify 为 100%；因此这些数字不能当作严格数值等价证明。

同机两卡 HTTP 冒烟测试使用 GPU 2 运行 Llama-2-7B 云端、GPU 0 运行 TinyLlama 边缘端，`token_budget=64`、逻辑 batch 上限 2。4 个请求全部完成，共生成 128 token、进行 57 轮 Verify、69 次 RPC。云端记录 59 次 varlen 前向（9 次 Prefill、50 次 Verify），最大实际 batch 为 2；请求平均端到端时间 3.62 秒，平均传输往返耗时约 5.40 ms。这是同机 localhost 通信，不能代表 node2 与 node3 的跨机延迟，也没有测与 padding 的端到端对照。运行记录在 node2 的 `.../llama-varlen-test-20260924/two-gpu-http/run-20260925-005915/`；测试结束后四张卡都回到约 15 MiB 占用。

TinyLlama 与该 Llama-2-7B 的完整词表和样例编码一致，BOS/EOS ID 也相同；`pad_token_id` 分别为 2 和 0，所以当前严格 tokenizer 检查脚本返回 1。varlen 路径没有向模型填充 pad token，后续跨数据集实验仍需核查 prompt 模板和停止策略。
