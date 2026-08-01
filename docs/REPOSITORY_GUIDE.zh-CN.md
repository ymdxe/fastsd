# FastSD 仓库指南

> 本文是 [`REPOSITORY_GUIDE.md`](REPOSITORY_GUIDE.md) 的中文翻译。命令、参数、代码路径和实现状态描述均保留英文原文含义。

本文档是面向本仓库使用者和未来智能体的主要入门指南。它重点介绍当前代码库的组织方式、主运行路径的工作原理、实验的启动方式，以及修改代码前需要关注的注意事项。

## 1. 这个仓库是什么

FastSD 是一个研究云边场景下推测解码的代码仓库。仓库当前同时存在两代相互重叠的代码：

- 较旧的、面向基准测试的路径，直接从 `benchmark/*.py` 和 `comparison/*.py` 运行解码
- 较新的、面向服务的路径，其中：
  - 云端通过 `cloud/cloud_service.py` 运行 FastAPI 目标模型服务
  - 边缘端通过 `edge/edge.py` 运行草稿生成循环与 HTTP 协调逻辑

分支历史表明，较新的路径是逐步加入的：

- `cloud`
- `communication ok`
- `draft while waiting`
- `pipeline`
- `fastsd`
- `energy meter`
- `fix: align FastSD scheduler with paper logic`

这意味着该仓库并不是一个打磨完善的产品包。它是一棵活跃的研究代码树，其中包含遗留脚本、部分重叠的入口，以及一些依赖特定环境的假设。

## 2. 论文目标与当前代码

这一区别至关重要：

- 论文描述的是预期中的 FastSD 系统设计
- 仓库包含的是当前实现状态

二者相互关联，但并不完全相同。未来的智能体绝不能因为“论文说了 X”，就推断“代码已经实现了 X”。

### 2.1 论文层面的系统目标

论文将 FastSD 描述为一个面向大规模异构部署的云边推测解码系统，其中：

- 边缘设备运行草稿 LLM
- 云端运行目标 LLM
- 一个云端目标模型会随时间服务多个边缘草稿客户端
- 系统针对多请求、生产服务式场景进行优化，而不是只考虑一个边缘模型与一个云端模型配对的情况

在论文层面，系统动机来自以下四个事实：

- 仅解码器 LLM 的推理受到顺序自回归解码的限制
- 推测解码可以让目标模型在一次前向计算中验证多个草稿 token，从而减少目标端执行步数
- 云边部署会引入通信延迟、设备异构和多客户端竞争
- 大量并发请求下，KV 缓存在 CPU 或磁盘与 GPU 之间的移动会成为实际瓶颈

### 2.2 论文层面的创新主张

你提供的论文文本将 FastSD 对应到三个主要设计贡献。

#### A. 基于优先级的多队列任务调度

目标与动机：

- 云端会同时收到预填充任务和验证任务
- 不同任务在序列长度、紧迫程度、设备速度、网络延迟和草稿接受质量方面存在差异
- 单一 FIFO 队列会让较长或代价较高的任务阻塞短任务，同时降低批处理效率

论文设计：

- 按任务类型拆分：
  - prefill
  - verify
- 将每种类型再划分为三个长度队列：
  - short
  - medium
  - long
- 使用滑动窗口上的分位数在线更新长度边界
- 在每个队列内部，根据以下因素分配优先级：
  - 草稿端计算速度
  - 通信状况
  - 接受概率
  - 草稿长度
  - 等待时间

#### B. 动态加权轮询批处理

目标与动机：

- 批处理能够提升 GPU 利用率
- 混合长度差异很大的序列会造成 padding 浪费
- 始终优先处理 verify 可能使 prefill 饥饿
- 始终处理 prefill 则可能使 verify 延迟激增

论文设计：

- 按固定的 6:3:1 比例服务 short、medium 和 long 队列
- 默认优先处理 verify 任务
- 只有在 verify 需求不足时才切换到 prefill
- 根据 verify 处理机会利用不足的情况，采用保守的切换规则

#### C. 基于预测的 KV 缓存预加载

目标与动机：

- 在大规模服务中，所有活跃请求的 KV 缓存通常无法一直驻留在 GPU 上
- verify 轮次会反复使用此前构建的 KV 缓存
- 从 CPU 内存或磁盘将 KV 缓存加载到 GPU 会引入不可忽略的延迟

论文设计：

- 预测最有可能接下来执行的 verify 任务
- 在当前 batch 仍在运行时，主动将这些任务的 KV 缓存预加载到 GPU 内存
- 减少下一个 verify batch 开始前的 I/O 停顿

### 2.3 阅读代码时需要理解的论文假设

论文叙述使用了以下术语：

- prefill 任务：
  第一个推测轮次，用来初始化目标端 KV 缓存
- verify 任务：
  后续每个推测轮次，用来依据已缓存的前缀验证新生成的草稿后缀
- 推测轮次：
  边缘端的一次草稿阶段，加上云端的一次验证阶段

这些术语会出现在代码中，但代码仍应被视为一种实现尝试，而不是完整论文系统已经完成的证明。

## 3. 当前工作状态

在当前本地检出版本中：

- 分支：`master`
- 顶层受版本控制的代码目录：`benchmark/`、`cloud/`、`comparison/`、`data/`、`docs/`、`edge/`、`scripts/`、`src/`、`tests/`
- 顶层已经存在输出目录：`exp/`
- 工作树并不干净

当前本地未提交状态与远程开发服务器一致，其中包括：

- 已修改：
  - `cloud/cloud_service.py`
  - `install.sh`
  - 一些 `__pycache__` 文件
- 未跟踪：
  - `requirements.clean.txt`
  - `tests/__pycache__/test_fastsd_scheduler.cpython-310.pyc`

这意味着：

- 不要认为当前检出版本是原始、干净的基线
- 编辑前检查 `git status --short`
- 如果运行新实验，请使用新的 `--exp_name`，以免与已有输出混在一起

## 4. 代码库的思维模型

理解这个仓库最快的方法，是将它看成四个层次。

### 4.1 核心解码层

位置：`src/`

这是项目真正的中心。最重要的文件包括：

- `src/engine.py`
- `src/util.py`
- `src/kvcache.py`
- `src/kvcache_batching.py`
- `src/kvcache4RC.py`
- `src/energy_meter.py`

它们的作用：

- `src/engine.py`
  - 定义抽象基类 `Decoding`
  - 加载模型和 tokenizer
  - 实现多种解码变体
  - 包含后来加入的 FastSD 调度辅助逻辑
  - 集成可选的能耗测量
- `src/util.py`
  - 集中的参数解析
  - 通过 `model_zoo` 将模型别名映射为路径
  - 提供 `norm_logits`、`sample`、`top_k_top_p_filter` 等采样辅助函数
- `src/kvcache_batching.py`
  - 以进程 ID 为键管理批量 KV 缓存
  - 由云端 worker 路径使用
- `src/energy_meter.py`
  - 提供功率采样和面向云端能耗测量的 FastAPI 控制服务

重要设计事实：

- 大多数脚本都只是较薄的封装
- 几乎所有实质性行为最终都会经过 `src/engine.py` 中的 `Decoding`

### 4.2 基准测试层

位置：`benchmark/`

这些文件是与具体数据集对应的评估入口：

- `benchmark/eval_humaneval.py`
- `benchmark/eval_gsm8k.py`
- `benchmark/eval_mgsm.py`
- `benchmark/eval_mt_bench.py`
- `benchmark/eval_SingleDraft.py`
- `benchmark/eval_MultiDraft.py`

共同模式：

- 每个文件都继承 `Decoding`
- 每个文件都定义：
  - `load_data`
  - `preprocess`
  - `postprocess`
  - `eval`
- `eval()` 根据 `args.eval_mode` 选择解码实现

常见模式包括：

- `small`：只在草稿模型上执行自回归解码
- `large`：只在目标模型上执行自回归解码
- `sd`：单进程推测解码
- `para_sd`：双进程或并行推测解码
- `para_sd_wo_1` / `para_sd_wo_2`：分别移除一种优化的消融版本
- `rc_para_sd`：较新的云边路径所使用的一种专用变体

### 4.3 云边服务层

位置：

- `cloud/cloud_service.py`
- `edge/edge.py`

这是当前仓库发展方向中最重要的路径。

云端：

- `cloud/cloud_service.py` 启动 FastAPI 服务
- 它暴露以下接口：
  - `GET /health`
  - `POST /session/init`
  - `POST /prefill`
  - `POST /verify`
  - `POST /exit`
- API 后面会启动一个复用 `Decoding.run_target_process_batching(...)` 的 worker 进程
- 请求队列和响应队列都是多进程队列
- API 线程将请求交给 worker，并通过 future 等待结果

边缘端：

- `edge/edge.py` 包含：
  - `EdgeClient`：访问云端目标服务的 HTTP 客户端
  - `EdgeRunner`：复用 `Decoding` 并运行本地草稿循环
- 该文件负责：
  - 创建会话
  - 生成草稿 token
  - 向云端发送异步 verify 请求
  - 重叠草稿生成与等待过程
  - 流水线特有的 gamma 自适应
  - 将单任务指标记录到 `exp/<exp_name>/edge_metrics_proc*.jsonl`

从概念上看：

1. 边缘端生成草稿 token
2. 边缘端向云端发送 verify 请求
3. 云端使用目标模型进行验证
4. 边缘端合并已接受 token 与最终 token
5. 边缘端重复以上过程，直到触发停止条件

### 4.4 实验 Shell 层

位置：`scripts/`

这些文件是大多数用户实际会接触的入口。

当前相关脚本包括：

- `scripts/run_fastsd_profile.sh`
- `scripts/run_vanilla_profile.sh`
- `scripts/case_study.sh`
- `scripts/ablation_study.sh`
- `scripts/run_sd.sh`
- `scripts/run_para_sd.sh`
- `scripts/run_assist.sh`
- `scripts/run_comp.sh`

重要区别：

- `run_fastsd_profile.sh` 和 `run_vanilla_profile.sh` 面向较新的 `edge/edge.py` 路径
- `run_sd.sh` 和 `run_para_sd.sh` 是较旧的基准测试启动命令集合
- `run_assist.sh` 和 `run_comp.sh` 是外部基线的对比脚本

## 5. 当前实现相对于论文的状态

本节特意与前面的论文目标部分分开。

### 5.1 当前代码中明确体现的内容

以下思想目前可以直接在仓库中看到。

- 云边拆分已经存在。
  - 云端服务位于 `cloud/cloud_service.py`
  - 边缘端运行程序位于 `edge/edge.py`
- 在较新的云端路径中，Prefill 和 Verify 是明确的请求类型。
- `src/engine.py` 中存在多队列调度。
- Short / mid / long 分类已经存在，并且会动态更新。
- 固定的加权轮询顺序已经存在。
- 以 Verify 为先、按条件切换到 Prefill 的调度已经存在。
- 目标端 KV 缓存会跨请求显式管理。
- GPU 与 CPU 之间的 KV 缓存卸载已经明确包含在请求处理逻辑中。
- 云端运行中存在能耗测量钩子。

### 5.2 部分实现或仍处于研究代码水平的内容

以下内容以某种形式存在，但不应视为已经完成的生产级实现。

- 优先级评分已经存在，但应将其视为当前的启发式实现，而不是最终确定的系统抽象。
- 流水线自适应已经存在，但仍与实验代码和调试开关交织在一起。
- `edge/edge.py` 中已经存在主动草拟以及草拟与验证重叠，但控制流仍是研究代码风格。
- 代码结构中已经存在 KV 缓存预加载，但周围系统仍在演进，并非经过加固的服务子系统。
- 仓库仍然混合了旧基准测试路径与较新的云边运行路径。

### 5.3 不能仅根据代码安全推断的内容

未来的阅读者应避免只依据当前仓库就过度声称以下结论。

- 每项论文贡献都已经端到端完整实现
- 所有基线都在相同运行假设下完成了归一化
- 所有 Shell 脚本都对应最新的论文设置
- 当前仓库已经是生产级服务系统
- 论文所述的所有异构设备因素都在当前代码中得到完整建模

### 5.4 目标与实现差异的具体示例

KV 缓存是一个很好的例子：

- 论文目标：
  CPU 或磁盘与 GPU 之间的 KV 缓存移动是生产服务中的瓶颈，因此 FastSD 应通过更智能的预加载和调度来降低延迟
- 当前代码：
  仓库现在会在目标端请求路径中明确地在 GPU 与 CPU 之间卸载 KV 缓存，并且包含与预加载或将选定缓存保留在 GPU 上相关的逻辑

这说明代码体现了论文动机，但并不自动意味着完整的预期服务策略已经完成或得到充分评估。

## 6. 详细目录地图

### `src/`

用途：

- 可复用的解码逻辑、缓存逻辑、模型加载、调度逻辑和工具函数

值得优先阅读的文件：

- `src/engine.py`
- `src/util.py`
- `src/kvcache_batching.py`
- `src/energy_meter.py`

推荐阅读顺序：

1. `src/util.py`
2. `src/engine.py`
3. `src/kvcache_batching.py`
4. `edge/edge.py`
5. `cloud/cloud_service.py`

### `cloud/`

用途：

- 边缘端运行程序所使用的 HTTP 目标模型服务

主要文件：

- `cloud/cloud_service.py`

运行注意事项：

- 该文件使用 `spawn` 多进程启动方式
- 如果你修改 worker 初始化，或者移动必须可被 pickle 序列化的代码，这一点非常重要

### `edge/`

用途：

- 边缘端推测解码循环与云端通信

主要文件：

- `edge/edge.py`

它的重要性：

- 当前 FastSD 的运行行为在这里体现得最清楚
- 流水线和主动草拟行为在这里进行协调

### `benchmark/`

用途：

- 与具体数据集对应的评估工具

覆盖的数据集：

- HumanEval
- GSM8K
- MGSM
- MT-Bench

输出模式：

- 将 `.jsonl` 结果写入 `exp/<exp_name>/`

### `comparison/`

用途：

- Assist、Ouroboros 等非 FastSD 基线的评估脚本

注意事项：

- 一些 Shell 脚本仍然引用 `comparation/...`，这与实际目录名 `comparison/` 不一致
- 应将这些脚本视为历史脚本，并在重新使用前进行验证

### `data/`

用途：

- 以 `.jsonl` 形式打包的评估数据集

当前文件：

- `humaneval.jsonl`
- `gsm8k.jsonl`
- `mgsm.jsonl`
- `mt_bench.jsonl`

### `exp/`

用途：

- 实验输出和边缘端指标汇总

当前目录包括：

- `exp/fastsd`
- `exp/vanilla_run`
- `exp/proactive_run`
- `exp/pipeline_run`
- `exp/both_run`
- `exp/test`

汇总文件字段示例：

- `profile`
- `enable_pipeline`
- `enable_proactive_draft`
- `num_drafts`
- `num_tasks`
- `wallclock_s`
- `total_generated_tokens`
- `system_tok_per_s`
- 延迟分位数

### `tests/`

用途：

- 面向较新逻辑的轻量单元测试

当前测试覆盖：

- `tests/test_energy_meter.py`
  - 验证 `EnergyAccumulator` 的积分逻辑
- `tests/test_fastsd_scheduler.py`
  - 验证：
    - 固定加权轮询顺序
    - Prefill 切换阈值
    - 运行时分位数阈值
    - 优先级分数计算

重要限制：

- 没有针对云边交互的端到端集成测试
- 大多数运行行为仍然通过手工实验验证

## 7. 核心运行架构

当前实际架构如下：

```text
edge/edge.py
  -> EdgeClient 发送 HTTP 请求
  -> cloud/cloud_service.py 接收请求
  -> worker 进程调用 Decoding 的目标端批处理路径
  -> 边缘端合并验证响应并继续草拟
  -> 指标写入 exp/<exp_name>/
```

主要抽象包括：

- `Decoding`
  - 负责模型加载和解码算法
- `KVCacheModel_batching`
  - 在云端按请求保存用于批量验证的缓存状态
- `EdgeClient`
  - 云端 API 封装
- `CloudTargetWorker`
  - 基于 `Decoding` 构建的目标端 worker

### `src/engine.py` 中的 FastSD 调度器组件

较新的调度逻辑包含以下辅助函数：

- `_build_fixed_wrr_order`
- `_update_length_thresholds`
- `_compute_priority_score`
- `_should_switch_to_prefill`

这些函数表明：

- 请求按照 prompt 长度分类
- 调度器在预热后使用动态分位数阈值
- Verify 优先级根据 lag、传输 RTT、等待时间和接受率统计进行调整
- 当 Verify 利用率足够低时，云端可以择机切换到 Prefill

这是论文设计在实现中体现得最明显的位置之一。

## 8. 配置与参数

大多数运行参数由 `src/util.py` 中的 `parse_arguments()` 定义。

重要参数：

- 数据集：
  - `humaneval`
  - `gsm8k`
  - `mt_bench`
- 模型选择：
  - `--draft_model`
  - `--target_model`
- 解码：
  - `--eval_mode`
  - `--gamma`
  - `--num_drafts`
  - `--batch_size`
- 输出：
  - `--exp_name`
- FastSD 行为：
  - `--server_sched_mode`
  - `--profile`
  - `--enable_proactive_draft`
  - `--enable_pipeline`
  - `--pipeline_gamma_adapt`
- 诊断：
  - `--debug_pipeline`
  - `--debug_verify_tokens`
- 能耗：
  - `--measure_energy`
  - `--energy_api_host`
  - `--energy_api_port`

重要注意事项：

- 脚本中的模型名称是别名，而不是直接的文件系统路径
- `src/util.py:model_zoo()` 会将别名映射为本地路径
- 该映射仍包含依赖具体环境的占位符和硬编码本地路径

实际影响：

- 如果脚本在加载模型时失败，首先检查 `model_zoo()`

## 9. 实验通常如何运行

### 7.1 安装

```bash
bash install.sh
```

该命令会安装：

- PyTorch 2.1.2 CUDA 12.1 wheels
- Transformers
- Accelerate
- AutoGPTQ
- FastAPI / Uvicorn
- 用于功率测量的 `nvidia-ml-py3`

### 7.2 较新的云边运行路径

云端：

```bash
python cloud/cloud_service.py --exp_name fastsd
```

边缘端，FastSD 配置：

```bash
bash scripts/run_fastsd_profile.sh fastsd --num_drafts 4 --dataset humaneval --max_tokens 256
```

边缘端，基线配置：

```bash
bash scripts/run_vanilla_profile.sh vanilla vanilla_run --num_drafts 4 --dataset humaneval --max_tokens 256
```

如果云端不在本机，请设置：

```bash
export SERVER_URL=http://<cloud-host>:8001
```

重要说明：

- `scripts/run_fastsd_profile.sh` 默认将边缘端指向 `http://127.0.0.1:8001`
- `cloud/cloud_service.py` 当前在端口 `8001` 上启动 Uvicorn

这种不一致意味着，在使用前必须修改其中一项。这是当前仓库的注意事项，而不是用户操作错误。

### 7.3 较旧的基准测试路径

单进程推测解码示例位于：

- `scripts/run_sd.sh`

并行推测解码示例位于：

- `scripts/run_para_sd.sh`

这些脚本主要调用：

```bash
accelerate launch benchmark/<dataset-script>.py ...
```

### 7.4 对比基线

示例：

- `scripts/run_assist.sh`
- `scripts/run_comp.sh`

应将这些脚本视为参考，而不是保证可以直接运行的生产入口。

## 10. 输出与指标

大多数实验输出位于 `exp/<exp_name>/`。

典型文件：

- 单任务指标：
  - `edge_metrics_proc0.jsonl`
  - `edge_metrics_proc1.jsonl`
  - ...
- 聚合指标：
  - `edge_metrics_summary.json`
- 基准测试结果：
  - `<eval_mode>_<dataset>.jsonl`

汇总文件中已经包含的指标有：

- 墙钟时间
- 生成 token 总数
- 系统吞吐量，单位为 tokens/s
- 平均任务延迟
- p50/p90/p95 延迟

`edge/edge.py` 中的单任务指标还会记录：

- 每轮草稿时间
- 验证等待时间
- 云端服务时间
- 传输 RTT 估计值
- 接受率
- 主动复用统计

## 11. 测试与验证命令

当前单元测试：

```bash
python -m unittest tests.test_energy_meter tests.test_fastsd_scheduler -v
```

它们覆盖得较好的内容：

- 确定性的辅助逻辑
- 能耗积分计算

它们未覆盖的内容：

- 端到端云边交互
- 模型加载
- 针对完整数据集的基准测试正确性
- 脚本兼容性

如果你修改了云边行为，那么在信任手工运行结果之前，至少应再添加一个围绕调度器或指标逻辑的单元测试。

## 12. 已知注意事项与脆弱环节

对于未来的智能体来说，本节比大多数章节都更重要。

### 依赖具体环境的模型路径

`src/util.py` 的 `model_zoo()` 中仍包含本地路径假设和占位符。

### 脚本与服务之间的端口不一致

- `scripts/run_fastsd_profile.sh` 和 `scripts/run_vanilla_profile.sh` 默认使用 `http://127.0.0.1:8001`
- `cloud/cloud_service.py` 运行在端口 `8001`

在认定这些 Shell 脚本可以直接使用之前，应先解决这一问题。

### 历史脚本可能已经过时

一些脚本仍然引用当前不存在的名称，例如 `comparation/...`。

### 多代工作流混合

仓库中同时包含：

- 直接基准测试运行
- 对比基线
- 较新的云边服务编排

它们相互关联，但尚未在统一 CLI 下完全归一化。

### 论文叙述与代码叙述并不相同

阅读这个仓库时，必须分别回答两个问题：

1. 根据论文，FastSD 想要成为怎样的系统？
2. 这份代码目前实际实现了什么？

如果不将二者分开，就很容易编写出错误的文档和实验说明，或者给未来智能体留下错误假设。

### 不干净的工作树

本地仓库有意保持与远程开发工作树中未清理的状态一致。

### `README.md` 此前为空

目前，这份指南是更可靠的入门文档。

## 13. 新智能体的推荐阅读顺序

如果你是未来的智能体，需要快速开始有效工作，请按以下顺序阅读：

1. `README.md`
2. `docs/REPOSITORY_GUIDE.md`
3. `src/util.py`
4. `src/engine.py`
5. `edge/edge.py`
6. `cloud/cloud_service.py`
7. `tests/test_fastsd_scheduler.py`
8. 一个启动脚本：
   - `scripts/run_fastsd_profile.sh`
   - 或 `scripts/run_vanilla_profile.sh`

然后检查：

- `git status --short`
- 目标 `exp/<exp_name>/` 目录
- 当前用户实际准备运行的任何命令

## 14. 推荐的后续清理任务

使用该仓库并不要求先完成以下任务，但这些改进会显著提升可维护性：

1. 统一云端服务端口与 Shell 脚本默认值。
2. 将模型路径映射从源代码移到配置文件。
3. 在文档中将遗留基准测试入口与当前云边运行路径分开。
4. 为 `cloud/cloud_service.py` 与 `edge/edge.py` 增加一个端到端冒烟测试。
5. 在版本控制中清理或忽略临时 `__pycache__` 文件。
6. 替换仍引用 `comparation/` 的过时 Shell 脚本。
7. 增加一份明确文档，将每项论文主张映射到具体代码位置和实验脚本。

## 15. 最小运行检查清单

运行任何内容前：

1. 检查 `git status --short`。
2. 确认 `src/util.py:model_zoo()` 指向真实的本地模型。
3. 确认云端 URL 和端口一致。
4. 使用新的 `--exp_name`。
5. 确定你准备运行的是：
   - 较旧的基准测试脚本
   - 较新的云边配置脚本
6. 如果测量能耗，请确保已经安装 `nvidia-ml-py3`，并且所选 GPU 索引正确。

## 16. 总结

理解这个仓库的最佳方式，是将其视为一棵以 `src/engine.py` 中的 `Decoding` 为中心的研究工作树，其上又通过 `cloud/cloud_service.py` 和 `edge/edge.py` 叠加了较新的云边运行路径。Shell 脚本很有用，但并非所有脚本都同样新。与此同样重要的是，论文目标与当前实现并不相同。对未来工作而言，最安全的默认做法是：

- 将 `src/`、`edge/` 和 `cloud/` 视为规范运行路径
- 将 `benchmark/` 视为数据集评估工具
- 将 `comparison/` 和部分 Shell 脚本视为历史或半手工内容
- 运行实验前核对端口、模型路径和工作树状态
