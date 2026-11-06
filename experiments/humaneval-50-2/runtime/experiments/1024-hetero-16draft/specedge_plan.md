# SpecEdge 单链后续实验计划

本文件是源码审查和待执行计划；尚未启动 SpecEdge、安装依赖或验证模型加载。先完成本轮 Standard SD / FastSD，再释放它们的 Target，按本轮冻结的输入、CPU 核映射、dtype、输出上限和计时口径执行 SpecEdge。

## 固定范围与必要参数

结果名称建议为 `specedge_cpu_adapted_single_chain`，说明保留官方 SpecExec、主动草拟、gRPC 与服务端验证流程，CPU、异构长度和模型 offload 使用实验适配器。

| 参数 | cpunode1 | cpunode3 | gpunode2 |
| --- | --- | --- | --- |
| worker/client 数 | 8 | 8 | `num_clients=16` |
| 模型 | Qwen3-1.7B | Qwen3-1.7B | Qwen3-32B |
| 主 `max_n_beams` / `max_branch_width` | 1 / 1 | 1 / 1 | 服务端容量覆盖最长输入 |
| 主 `max_beam_len` / `max_budget` | 100 / 100 | 20 / 20 | 最大候选100，包含root的输入101 |
| `proactive.type` | included | included | — |
| 主动草拟 `max_n_beams` / `max_branch_width` | 1 / 1 | 1 / 1 | — |
| 主动草拟 `max_beam_len` / `max_budget` | 100 / 101 | 20 / 21 | — |
| 输出上限 | 256 / EOS | 256 / EOS | — |
| dtype | 与本轮相同FP16 | 与本轮相同FP16 | FP16 |
| Target sampling | — | — | `temperature=0`，greedy |
| Target缓存预填充 | — | — | `cache_prefill=false` |
| 执行设备 | CPU，复用本轮绑核/NUMA | CPU，复用本轮绑核/NUMA | 仅物理GPU3，GPU34GiB/CPU96GiB权重预算 |

`max_batch_size` 沿用本轮云端已冻结的值。使用 `batch_type=dynamic`，避免静态批次在一部分客户端提前完成后等待永远凑不齐的请求。`max_len` 要覆盖最大实际prompt长度 + 256 + 主动草拟/bridge临时节点；先根据本轮输入token数计算，再固定为足够的容量（通常1024或2048）。它是静态KV及树mask的容量，不是输出token上限。

## 单链语义已按实现核对

以 `/home/hdd/zhangh/workspace/fastsd/baselines/specedge/official/src/` 为审查基准：

- `specedge/client/specexec.py:173–221`：`max_beam_len` 控制逐层扩展；`max_n_beams=1` 只限制每轮处理的候选数，必须同时将 `max_branch_width=1` 才限制每个父节点只有一个孩子。主深度和budget同为γ时，生成γ个草稿节点。
- 同文件 `378–420`：发送的是已确认末token/root加γ个草稿，`input_ids` 长度γ+1、`parent_indices` 长度γ。root不得计为草稿。
- `specedge/client/proactive.py:31–96`：主动草拟先插入一个猜测bonus/bridge，再扩展指定深度；因此主动深度γ需要budget至少γ+1。`included` 模式在复用命中后从主深度扣除已提前生成的深度，避免重复生成一整块。
- 同文件的主动草拟仍运行官方流程：等待验证时调用完整主动draft；它不像FastSD在每个token后检查验证是否已完成。这个实现差异要保留并披露，不能为匹配结果而悄悄改算法。

验收时记录每轮候选token数、root数、每个父节点孩子数、链深度和主动复用情况。要求节点内主verify候选数固定100/20，所有非root节点形成单条父链。先做包含拒绝、全接受和主动复用的短冒烟，再运行正式请求。

## 当前入口不能直接运行此配置

1. **当前checkout的official目录为空。** 本地 `/home/hdd/zhangh/workspace/fastsd/baselines/specedge/official` 已核验HEAD为 `1edcaf02ffc41a7b57726450c5357ed216a3b9bc` 且工作区干净，可复制为本实验源码快照并记录hash。`new_fastsd`副本也有同HEAD；`new_new_fastsd`副本无可用.git且包含CPU KV等额外改动，不能将它直接标为未修改官方源码。
2. **现成client launcher只有全局树参数。** `integration/client_host.py:38–88` 对所有client读取同一个client配置。实验launcher需分别在tmux6的两个节点设置对应γ参数，分配不重复的全局client_idx 0–15，并保持与本轮相同的题目/worker映射。不能让两个节点各自都使用client_idx 0–7。
3. **官方server不支持混合请求宽度。** `strategy/server_verify/specexec/grpc.py:183–184,440–475` 按单一max_budget分配张量，并将收到的input/parent整行`copy_`进去。101与21的输入无法直接混批。必须实现实验用长度适配：记录每请求实际长度，安全padding和valid-length处理，返回时裁剪selection；或者同长度分批并披露该批处理限制。KV回滚/接受路径必须忽略padding节点，不能只补零后沿用所有固定宽度索引。默认客户端会按原始input长度reshape响应。
4. **server等待超时仅5秒。** `grpc.py:98` 的`asyncio.wait_for(..., timeout=5.0)`需在实验适配器中改为与本轮请求超时一致的1200秒，尤其是32B权重CPU offload和排队场景。

## 模型、GPU和运行环境适配

- 不能直接复用FastSD的Target类：SpecEdge使用 `model.qwen3.Qwen3ForCausalLM` 和自定义 `model.cache.KVCache`，模型forward还要求cache索引、tree attention mask等参数。`util.py:45–59`的`load_graph_model`目前只接受单设备。实验loader应加载同一个32B checkpoint的官方自定义Qwen3类，使用与本轮一致的FP16与GPU/CPU预算，保存实际`hf_device_map`。
- `BatchGraphEngine`支持`use_cuda_graph=False`（`specedge/engine/graph.py:205–239`），服务端创建时需要显式关闭。权重CPU offload不应放入CUDA Graph捕获；禁止为了避免显存问题改为量化或使用占用中的GPU。
- 干净官方版本将每个client完整KV和活动batch KV都放在GPU；16份长上下文KV叠加34GiB权重可能OOM。实验适配器应把非活动client KV留在CPU，仅活动batch放GPU，类似本轮FastSD的驻留方式，记录迁移边界与成本。`new_new_fastsd`副本有此类改动可作参考，但不能无记录地替换官方版本。
- 当前CPU适配器 `integration/cpu_adapter.py`复用官方模型和KV，仅替换GraphEngine和CUDA计时，并在每次forward设置真实source seq_indices；继续使用。
- 云端已存在 `/home/hdd/zhangh/envs/new_fastsd`：Python3.10、torch2.2.1+cu121、Transformers4.57.6、Accelerate1.12、grpcio1.84、protobuf7.36.2、rich15。原官方src用Python3.10进行**仅语法检查**时，只发现 `util.py:137` 的f-string含反斜杠不兼容，可在实验源码副本作等价语法改写并归档补丁。该发现不代表所有依赖和模型已成功运行。
- 另有 `/home/hdd/zhangh/envs/specedge` 的Python3.14/torch2.9/Transformers4.57.1环境，但尚未在本审查中执行torch导入；两CPU节点依赖可用性亦待验证。先检查现有环境，不安装或升级依赖。

## 输入、网络和指标对齐

- 复用本轮已冻结的原题、sample ID、node/worker分配和non-thinking chat模板。现成 `integration/client.py:123–141`只给mt_bench加chat模板，GSM8K直接使用prompt；本实验必须预先格式化或显式适配，并逐题核对prompt token IDs与本轮一致。服务端runtime prefill收到的prefix必须同样一致。
- `integration/client.py:45–49`将wire attention mask硬编码bf16，而服务端按Target dtype解码。本轮FP16需要实验transport明确使用`wire_mask_dtype=torch.float16`，不能发送bf16字节让FP16解读。
- 关闭 `cache_prefill`，将Target实际prefill计入TTFT。模型加载和约定warmup排除，待16个模型及服务端就绪后统一放行；复用本轮协调器的单时钟完成窗口。SpecEdge的Sync是16客户端barrier，但不等同于Target模型加载就绪。
- `IntegratedSpecExecClient.generate_exact`已提供EOS和256-token精确截断。首token时间应只在第一批非空已验证输出返回时记录；本轮正式输出需保留逐请求token IDs以进行方法间一致性检查。
- 继续使用 `TTFT=request_start→first_verified_output`，`TPOT=(E2E−TTFT)/(output_tokens−1)`；总体和两节点分别报告count、mean、p50、p95、输出tokens、接受率、吞吐和失败数。单token请求TPOT不纳入有效TPOT分位数。
- 官方cycle日志的`num_accepted_tokens`**包含bonus token**（`specexec.py:471–523`），不能直接作draft接受数。链式接受率应使用每轮`num_accepted_tokens−1`之和 / 实际提交的draft候选之和，并将最后一块候选接受数与最终保留输出数分开。
- gRPC耗时与本轮HTTP耗时只作各自RPC延迟，协议不同；主比较采用相同请求边界的TTFT/TPOT/E2E与统一公共窗口吞吐。

## 执行顺序

1. 完成本轮两方法，保存配置、原始结果和Target清理记录。
2. 冻结干净官方源码快照及实验适配补丁；检查三个节点现有依赖，准备相同模型视图与输入token校验。
3. 先对长度21/101、拒绝、全接受、主动复用、EOS、尾部截断做真实gRPC冒烟，验证无padding污染和KV回滚错误。
4. tmux8运行SpecEdge GPU3+CPU Target；tmux6在两CPU节点各启动8client，统一屏障释放。只清理本实验产生的进程。
5. 输出本轮同口径JSON/CSV/Markdown结果，并单列SpecEdge适配边界、实际verify宽度和调度/迁移信息。
