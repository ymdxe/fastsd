# 部署与验证记录

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: prepare / validate software protocol
- Verification Status: PREPARED；四端无模型预检通过；模型正确性与性能未执行。
- Date: 2026-10-26（Asia/Shanghai）

## 已执行

- 登录 gpunode2、cpunode1、cpunode2、cpunode3，读取适用操作约定，核查代码位置、环境和本轮脚本状态。
- 从 GPU 当前工作副本复制必要 Python 源码和既有对照输入到 E 盘；保留原项目未提交修改。把新实验副本部署到四端 `manual-compare-20261026`，不覆盖原推理源码。
- 四端均通过完整包文件校验，bundle ID：`429dc9449e59a3e18ac1311660e0e2ca5f0b268a43da8d18a916efbd11519ded`。
- 四端 Bash 语法、实际解释器依赖、分词器、模型分片存在性预检通过。GPU 导入 cloud/AR 并注册路由；三CPU实际 Edge --help 均退出0；未触发服务startup或模型加载。
- 词表hash `c96fa22dee2dd9c98c4f0c73ea423e21907bc96abda81fa5c6b2e24a670c66bf`，聊天模板hash `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` 四端一致；EOS151645、PAD151643。模型分片数：8B为5、CPU1 0.6B为1、CPU2 0.6B为1、CPU3 1.7B为2。
- 15种到达计划入口（12份正式trace+single/parity/pilot）、ID/提示映射、逐流顺序、80题parity覆盖及方法开关检查通过。正式原trace文件哈希保留；修正Windows路径分隔符在Linux读取的问题。
- 纯CPU假状态测试通过：READY/T0和时钟偏差换算、run/bundle/trace校验、空流预热、报告归属、排队指标及窗口汇总、异步最终缓存采样。
- 正确性放行器正例和拒绝情况通过：状态泄漏、token不一致、缺worker、旧bundle、oracle差异、完整结果后的中断均拒绝。测试使用独立fixture，交付包没有gate.json。
- 生命周期测试通过：释放延期、WorkItem退休、状态清理、旧预取副本失效、vanilla控制分流、实际模型就绪标记与空闲释放。
- AR模拟工作线程测试通过：连续批预算、分块prefill忽略中间候选、续写、取消、缓存清零及仅关闭自身线程。未运行CUDA模型。

## 实际环境

| 节点 | 解释器 | PyTorch | Transformers | 逻辑CPU亲和性 |
|---|---|---|---|---:|
| gpunode2 | /home/hdd/zhangh/envs/new_fastsd/bin/python | 2.2.1+cu121 | 4.57.6 | 80 |
| cpunode1 | /home/zhangh/miniconda3/envs/fastsd/bin/python | 2.6.0+cpu | 4.57.6 | 80 |
| cpunode2 | /home/zhangh/venvs/fastsd_cpu/bin/python | 2.6.0+cpu | 4.57.6 | 94 |
| cpunode3 | /home/zhangh/envs/fastsd_cpu/bin/python | 2.2.1+cpu | 4.57.6 | 192 |

cpunode2模型使用 `/vol1/zhangh/models`，不套用其他CPU的NFS挂载路径。CPU1/2不是Git工作副本；GPU为kvpre，CPU3为main，均观察到原HEAD `ee0e334e222593c100b2ee6fc7fc9e4db2cf7cd3`（有Git的节点）。当前工作树保持，统一测量以独立包manifest为准。

## 空间与运行状态

写入前/后目标分区可用空间快照：GPU hdd3约880/880GiB，CPU1 HOME约600/600GiB，CPU2 HOME约1.9/1.9TiB，CPU3 HOME约403/401GiB。共享机器空间有同期变化，本包约2.4MB，不能把全部空间差值归因于本次写入。GPU hdd约69GiB，现有权重直接读取；不新增模型或环境。

检查时四张GPU均有其他计算任务；本轮19635无监听，CPU1/2/3的29635隧道未启动。未停止原有Llama服务和其他隧道；没有修改系统配置、权限、环境、防火墙或端口策略。未复制模型权重，未启动新模型、实验服务或隧道，未删除任何文件。

## 尚待用户手动执行

真实8B oracle、三方法单客户端与16客户端parity、60秒试测、完整36轮矩阵。原19/80输出差异根因未确定；软件预检不能替代真实模型验收。正式入口需真实结果生成gate.json后才能运行。

准备期间的少量失败已修正：SSH在受限执行环境中不可用时改走批准的联网执行；一次多源SCP的路径兼容问题改用明确目录复制；一次只读版本查询的括号错误已由实际解释器预检替代。模型实验没有因此启动。

## 本地证据

`E:\Codex\temp\fastsd-prepare-20261026\cpu-review-test.py`、`gate-review-test.py`、`test_lifecycle.py` 及各自结果目录。原始小型源码和既有对照材料保存在同任务temp子目录，不自动清理。