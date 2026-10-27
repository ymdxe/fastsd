# 当前服务器运行 Qwen3-32B

入口：`scripts/infer_qwen32b.py`。这里的 Qwen32B 按项目已有配置解释为 **Qwen/Qwen3-32B**。

使用 Transformers + Accelerate，将完整 BF16 权重按层分配到两张 GPU，单请求流式生成。默认关闭思考模式，使用 Qwen3 的 chat template 和非思考模式采样参数。该入口适用于验证模型和单请求推理；按层分配不等于 tensor parallel，也不提供连续批处理 HTTP 服务。

## 服务器检查结果

2026-10-25 检查时：

- 4 张 NVIDIA RTX A6000，每张约 48 GiB 显存。
- 最新复查时 GPU 0 / 1 / 2 / 3 已用显存约 3.5 / 13.8 / 0.02 / 20.1 GiB；GPU3 的占用已变化，使用前重新运行 `nvidia-smi`。
- 现有环境 `/home/hdd/zhangh/envs/new_fastsd`：Python 3.10.20、PyTorch 2.2.1+cu121、Transformers 4.57.6、Accelerate 1.12.0，能够导入 Qwen3，无需为此升级整个项目环境。
- 用户提供的 `/mnt/nfs0/zhangh/models/Qwen3-32B` 已有 32B 权重：17 个 safetensors 分片，707 个张量，全部为 BF16，张量数据总计 65,524,246,528 字节，约 61.02 GiB。
- 已核对全部分片的可读文件头、张量名、偏移和文件大小，与索引一致；没有对 61 GiB 数据执行全量校验和或完整加载。

单张 48 GiB 卡装不下完整 BF16 权重。下面使用物理卡 **0、2**，每卡最多规划 40 GiB 权重，且始终从实际空闲显存中至少预留 4 GiB。预算不足时脚本在加载权重前报错；运行时显存仍取决于输入长度、输出长度和其他进程占用。

## 1. 使用已有 NFS 权重

模型路径为 `/mnt/nfs0/zhangh/models/Qwen3-32B`，无需重复下载。加载脚本仅从指定路径读取。不要用当前默认的 `python`：检查时它属于另一个未安装 PyTorch 的环境。

**当前有目录访问权限阻塞，下面的完整推理命令需要先解决此问题。** 检查时目录权限为 `700`，数值属主 UID 为 `1039`，当前服务器的 `zhangh` UID 为 `1074`。实际检查可以按已知路径读取配置、索引和分片头，但 `ls` / `os.listdir()` 报 `Permission denied`；Transformers 4.57.6 的 `AutoTokenizer.from_pretrained()` 需要枚举目录，因此 tokenizer 加载失败。

需要由 NFS 目录所有者或管理员为当前 UID 配置适当的目录读取、遍历权限，并确认需要读取的模型文件可访问。本次只检查了权限，没有修改 ACL、权限位或属主，也没有复制权重。预检不加载 tokenizer，因此预检成功不代表该权限问题已解决。

## 2. 检查显存分配

```bash
cd /home/hdd3/zhangh/workspace/fastsd
nvidia-smi
/home/hdd/zhangh/envs/new_fastsd/bin/python scripts/infer_qwen32b.py \
  --model /mnt/nfs0/zhangh/models/Qwen3-32B \
  --gpus 0,2 \
  --check-only
```

`--gpus` 指物理卡编号。脚本在导入 PyTorch 前设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 和 `CUDA_VISIBLE_DEVICES`；因此日志中的 `cuda:0` / `cuda:1` 对应上述物理卡 0 / 2。选择自己可使用的卡，不要终止其他任务来腾显存。

检查使用 meta 模型计算参数量和层的分配，不加载实际权重。出现 `[CHECK_OK]` 只证明当时的权重显存规划通过，不能替代真实推理测试，也不验证权重分片完整性。

## 3. 运行推理

```bash
cd /home/hdd3/zhangh/workspace/fastsd
/home/hdd/zhangh/envs/new_fastsd/bin/python -u scripts/infer_qwen32b.py \
  --model /mnt/nfs0/zhangh/models/Qwen3-32B \
  --gpus 0,2 \
  --prompt '请用中文解释什么是投机解码，以及为什么它可以加速大模型推理。' \
  --max-new-tokens 512
```

首次加载可能需要较长时间。看到 `[READY]` 表示权重已加载并完成 CUDA 同步，之后逐步打印回答。正文写入 stdout，配置、设备分配、耗时和 token 数写入 stderr。生成速度包含 prefill 和流式打印时间，不是单独的 decode 吞吐或 TTFT 指标。

可选参数：

| 参数 | 用途 |
| --- | --- |
| `--thinking --max-new-tokens 2048` | 开启思考并提高输出上限；会输出思考过程，达到上限时仍可能尚未给出最终答案 |
| `--dtype float16` | 改用 FP16；默认 BF16 |
| `--gpu-memory-gib 38` | 降低每卡权重上限，给运行时留下更多空间 |
| `--reserve-gib 6` | 从每卡实际空闲显存中至少预留 6 GiB |
| `--max-input-tokens 2048` | 限制输入长度；过长输入会报错，不会静默截断 |
| `--system '你是一名中文编程助手。'` | 自定义系统提示 |

这是单次请求入口，每次启动都会重新加载模型。默认输入上限 4096、输出上限 512；提高长度前需要重新评估 KV cache 和运行时显存。不要把 `--gpu-memory-gib` 理解为进程总显存的硬限制。

## 验证范围

本次完成以下验证：

- Python 语法和 `--help` 正常；重复 GPU 编号、零输出长度、非有限显存预算和空 prompt 均被拒绝。
- 使用现有 `new_fastsd` 环境，在 GPU3 上真实加载本地 Qwen3-0.6B BF16，经过 chat template、SDPA 和 `generate()` 流式生成 23 个 token，正常退出。输出为“大语言模型是一种能够理解和生成高质量文本的模型，能够进行多语言交流，适用于各种应用场景。”
- 使用官方 Qwen3-32B 配置创建 meta 模型，计算出 32,762,123,264 个参数、约 61.02 GiB 的 16 位权重；在物理 GPU0/3 上规划成功，64 个 decoder 层均完整分配至 GPU。把两卡预算降至各 20 GiB 后，脚本按预期在加载前拒绝运行。

- 后续在用户指定的 NFS 目录找到已有 32B 权重，核对了全部 17 个分片的文件头、张量名、偏移、文件大小和索引总大小；tokenizer 加载因目录枚举权限失败。GPU3 新增约 20 GiB 占用后，原 `--gpus 0,3` 组合未通过当前显存规划。
- 改用 `--gpus 0,2` 并直接读取该 NFS 模型的配置后，`--check-only` 输出 `[CHECK_OK]`；权重预算约 39.65 / 40.00 GiB，64 个 decoder 层均分配至 GPU。

**尚未完成 Qwen3-32B 实际加载或生成验证，当前阻塞是 NFS 目录枚举权限。** 小模型推理和 32B meta 显存规划分别只覆盖对应的加载路径和静态容量估计；本次也未实测多卡间的生成传输。

模型架构、chat template 和采样参数核对自 Qwen 官方模型说明：`https://huggingface.co/Qwen/Qwen3-32B`。显存规划使用当前环境的 Accelerate `get_balanced_memory` / `infer_auto_device_map`。
