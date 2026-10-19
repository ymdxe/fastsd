# Llama 云边运行示例

从仓库根目录运行。把路径换成机器上完整的 Hugging Face/Transformers checkpoint；Llama-2-7B 不能直接使用 Meta 原始 `consolidated.00.pth`。云端也要能读取 Draft 模型目录，因为云端 tokenizer 从 `--draft_model` 加载。开始前运行：

```bash
python scripts/check_tokenizer_consistency.py \
  --draft_model /path/to/TinyLlama-1.1B-Chat-v1.0 \
  --target_model /path/to/Llama-2-7b-hf
```

检查脚本严格比较特殊 token 配置。即使两个模型的词表与普通文本编码一致，仅 `pad_token_id` 不同也会返回 1；应查看报告，区分 padding 配置差异与 token ID 映射不一致。`varlen` 路径不会向模型补 pad token。

两张 GPU 分别运行时，两个终端各只暴露一张卡，进程内都使用 `cuda:0`。下面是同一台机器上的最小 HumanEval 冒烟测试；跨机器时把边缘端 `--server_url` 换成云端可访问地址，并设置云端 `CLOUD_SERVICE_HOST`。

云端终端：

```bash
CUDA_VISIBLE_DEVICES=0 CLOUD_SERVICE_PORT=8001 python cloud/cloud_service.py \
  --draft_model /path/to/TinyLlama-1.1B-Chat-v1.0 \
  --target_model /path/to/Llama-2-7b-hf \
  --target_device cuda:0 --target_dtype bfloat16 \
  --server_sched_mode fastsd --kv_batch_mode varlen \
  --batch_size 4 --token_budget 512 --exp_name llama_cloud_smoke
```

边缘端终端：

```bash
CUDA_VISIBLE_DEVICES=1 python edge/edge.py \
  --draft_model /path/to/TinyLlama-1.1B-Chat-v1.0 \
  --target_model /path/to/Llama-2-7b-hf \
  --draft_device cuda:0 --server_url http://127.0.0.1:8001 \
  --dataset humaneval --data_path data --num_drafts 1 --edge_gpus 1 \
  --max_tasks_per_draft 2 --max_tokens 32 \
  --server_sched_mode fastsd --kv_batch_mode varlen \
  --batch_size 4 --token_budget 512 \
  --enable_latency_priority --overlap_prefill_first_draft \
  --enable_proactive_draft --no-enable_pipeline \
  --exp_name llama_edge_smoke
```

边缘端指标写入 `exp/llama_edge_smoke/`。`--kv_batch_mode varlen` 在模型结构不受支持时直接报错，不会悄悄回退到 padding。`token_budget` 是一轮计划的容量上限；没有足够就绪任务时，不能保证每轮恰好用满。

想单独比较模型前向的 padding 与 varlen，可运行 `python scripts/benchmark_llama_varlen.py --model /path/to/Llama-2-7b-hf --output /path/to/result.json`。这个微基准不包括网络和完整调度，不能当作端到端吞吐结论。

`scripts/eval_suite.py` 用于完整四方法对比；模板在 `configs/evaluation/llama2_7b_tinyllama_four_method_cpu.json`。该模板的 CPU 绑核、NUMA、模型路径和两端地址都要按机器修改。`scripts/run_tests.py` 是测试入口。其他 `scripts/*.py` 多数是 CPU、GGUF 或容量诊断工具，不是上述 Llama 云边主流程。

脚本定位：

| 用途 | 脚本 |
| --- | --- |
| 模型检查与基准 | `check_tokenizer_consistency.py`、`test_target_ar.py`、`benchmark_llama_varlen.py` |
| 多方法评测 | `eval_suite.py`、`make_evaluation_matrix.py`、`run_tests.py` |
| CPU Draft 专项 | `preflight_cpu.py`、`cpu_draft_worker.py`、`launch_cpu_draft_workers.py`、`prepare_32worker_experiment.py`、`test_cpu_draft_communication.py` |
| 显存与并发诊断 | `test_single_gpu_model_capacity.py`、`benchmark_single_gpu_concurrency.py`、`benchmark_gguf_concurrency.py` |
