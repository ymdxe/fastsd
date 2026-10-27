# 2026-10-24 泊松全量实验

记录时间：2026-10-24T13:00:30.297290+08:00

- 用户指定 exp tmux；运行窗口 exp:campaign，两台CPU SSH窗口链接在 exp:1；未使用第三个无关CPU pane。
- Fetch origin main，最新仍为 ee0e334e222593c100b2ee6fc7fc9e4db2cf7cd3；保留 kvpre 未提交修改，复制为不可变推理快照。未 commit/push。
- node1/node3各16个独立 Qwen3-0.6B FP16 draft，每个1物理核/1线程；实际代码哈希已与GPU端核对一致。
- Target Qwen3-1.7B FP16在物理GPU1，完整驻GPU。GPU0/2/3上的其他任务未触碰。
- 全量GSM8K test1319题，EOS/1024-token上限，三种方法×三gamma×三全局lambda，共27组35613请求。
- 关键无模型检查已通过；没有追加模型smoke。首轮采集器上下文故障保留于 fastsd-g20-l1 并从正式比较排除；修复后正式首组为 fastsd-g20-l1-a2。
- 当前状态 running。首批两节点结果均已回传，至少一个draft已完成两道题，预取确实搬运且无失败。
- 独立按GPU1 UUID采样功率，修正内置服务固定NVML0及旧版NVML保留显存口径的影响。
- 本地结果路径 /home/hdd3/zhangh/workspace/fastsd/experiments/1024-poisson32-qwen06-17；两CPU端原始日志位于 /home/zhangh/workspace/fastsd/experiments/1024-poisson32-qwen06-17。
- 未安装环境、未新增模型/数据下载、未更改权限或系统服务；只开启本任务loopback控制/推理监听及SSH loopback reverse forwards。
- GPU主盘写入前约934GiB可用；CPU node1约609GiB、node3约514GiB可用。模型均复用既有检查点。
- 27组全部结束后由finish_monitor自动检查全覆盖、输出一致性并生成统计和PNG/PDF图；当前并未完成全部实验。

## 到达率修正：全局1/10/50请求每秒

按用户要求将原0.1/0.5/1改为1/10/50，保留所有非到达率参数及完整1319题。当前 fastsd-g20-l1-a2 与32个draft/GPU Target继续运行；lambda1文件字节哈希不变。两个CPU节点已同步校验新时间表；控制器会在当前组完成后自动切换并跳过已完成组。当前与后续预定组共27组，首个旧队列后续任务前设置了切换保护。原配置和同步包保存在rate-update-1-10-50/backups。纯配置和切换保护检查通过，未增加模型smoke。仅发流时长下界更新为约3.63小时，完整计算仍由CPU处理能力决定。

## 用户缩小范围：γ200/500，λ50，仅六组

已按用户新指令停止γ20/λ1首组，保留356题已回传结果（node1:219，node3:137）及全部功率/预取原始记录，标记interrupted_by_user，不计入六组正式比较。旧GPU服务exit0；旧组间切换和最终汇总监视器已停止。27组配置保留在subset-g200-g500-l50/backups。当前活动矩阵仅γ200/500×λ50×三种方法，共6组/7914题；模型、32个单核单线程draft、GPU1、全量1319题/组及1024-token上限保持一致。两个CPU节点已同步并核验推理代码和参数。新首组FastSD γ200/λ50已运行，32draft就绪，实际验证成功且KV预取已触发。汇总、输出一致性检查、图表和结束条件均改为从六组矩阵读取；最终汇总监视器已重新启动。未追加模型smoke。
