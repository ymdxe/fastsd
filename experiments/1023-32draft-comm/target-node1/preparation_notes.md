# 准备过程记录

- 两台 CPU 服务器无法连接 node2 临时文件服务端口；最终通过现有 SSH/tmux 终端同步代码并核对 hash，临时文件服务已关闭。
- node1 原 Conda 环境 huggingface-hub 2.2.0 与 Transformers 4.57.6 冲突。Target 使用可用 venv 的 torch/transformers/hub，并在其后附加 Conda site-packages 以复用缺少的 FastAPI 等包；未安装/升级或改动原环境。确切 PYTHONPATH 见 node1 target_launch.json。
- 初次启动发现 profile=vanilla 需要同时设置 server_sched_mode=vanilla，修正后启动成功。
- 单 worker 冒烟成功后调整实验适配器，使 dataset 明确标为 gsm8k，同时保留相同 non-thinking 模板；单 worker 用过的适配器另存 edge_comm.smoke1.py。
- node3 的另一个 Python3.14 环境因 GLIBC 版本无法导入 torch，未用于本实验。
- 正式 32-worker 使用 source/ 中项目原始协议，未改原仓库源码。实验适配器只控制 CPU/NUMA、启动屏障、统一模板、日志及未同步时钟的一向耗时占位。没有开启共享权重。
- 主机输出中的 CPU kernel 版本警告和缺少 CUDA 扩展提示未导致本次 CPU 测试失败。
- 测试全部完成后退出所有 draft，node1 Target 以 SIGINT 正常结束，保留 tmux 会话和用户正在运行的 Pi 窗格。
