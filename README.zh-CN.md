# FastSD

> 本文是 [`README.md`](README.md) 的中文翻译。命令、参数、代码路径和专有名词保持与英文原文一致。

FastSD 是一个面向云边协同场景下推测解码的研究代码库。当前仓库将三个层次的内容放在了一起：

- `src/` 中的核心解码与 KV 缓存逻辑
- `benchmark/` 和 `comparison/` 中的基准测试与对比实验入口
- `cloud/` 和 `edge/` 中较新的云端/边缘端服务路径

阅读本仓库时，需要始终牢记一个重要区别：

- 论文目标描述了预期中的 FastSD 系统及其研究贡献
- 当前代码库只是该目标的一份仍在演进中的部分实现

不要认为论文中的每项功能都已得到完整实现，也不要认为仓库中的每个文件都能与最终论文叙述一一对应。

如果你刚接触这个仓库，请先阅读详细指南：

- [docs/REPOSITORY_GUIDE.zh-CN.md](docs/REPOSITORY_GUIDE.zh-CN.md)
- [docs/CAMPUS_SERVER_EXTERNAL_ACCESS.md](docs/CAMPUS_SERVER_EXTERNAL_ACCESS.md)

快速入口：

- 安装依赖：`bash install.sh`
- 启动云端目标模型服务：`python cloud/cloud_service.py --exp_name <name>`
- 使用预设配置启动边缘端运行程序：`bash scripts/run_fastsd_profile.sh <exp_name>`
- 运行测试：`python -m unittest tests.test_energy_meter tests.test_fastsd_scheduler -v`

SpecEdge 论文基线：

- 固定版本的官方实现和与论文设置对齐的配置：[baselines/specedge/README.md](baselines/specedge/README.md)
- 基线控制层测试：`python -m unittest tests.test_specedge_repro -v`

当前仓库包含从远程开发服务器复制而来的本地未提交更改。在认定当前基线是干净状态之前，请先检查 `git status --short`。
