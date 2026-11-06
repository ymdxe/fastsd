# 实验目录索引

四台服务器使用相同短名称；本机原来没有的实验包不重复复制。
当前 14B 全量实验：`14b/`。结果继续保存在原 `exp/` 路径。
目录仅移动和改名，冻结包及原始证据内容保留。完整旧名对照见 `NAMES.json`；本机迁移记录见 `organize-20261103.json`。
CPU3 的 NVMe 实验包通过链接统一从本目录访问。运行中进程所需的旧入口暂时保留兼容链接。

| 新名称 | 原名称 | 存储位置 | 旧入口兼容链接 |
|---|---|---|---|
| `bf16-32` | `bf16-32-suite-20261029` | `/home/hdd3/zhangh/workspace/fastsd/experiments/bf16-32` | 无 |
| `14b-old1` | `config1-14b-full4-p60-1103T104120Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/14b-old1` | 无 |
| `14b-old2` | `config1-14b-full4-p60-1103T113438Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/14b-old2` | 保留 |
| `14b` | `config1-14b-full4-p60-1103T115837Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/14b` | 保留 |
| `ablation-1` | `config1-ablation-p25-n30-1103T040303Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/ablation-1` | 无 |
| `ablation-2` | `config1-ablation-p25-n30-1103T041405Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/ablation-2` | 无 |
| `ablation-3` | `config1-ablation-p25-n30-1103T042006Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/ablation-3` | 无 |
| `ablation-4` | `config1-ablation-p25-n30-1103T043258Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/ablation-4` | 无 |
| `chain-check` | `config1-chain-contract-20261031` | `/home/hdd3/zhangh/workspace/fastsd/experiments/chain-check` | 无 |
| `diagnostic-check` | `config1-diagnostic-validation-20261101` | `/home/hdd3/zhangh/workspace/fastsd/experiments/diagnostic-check` | 无 |
| `diagnostic` | `config1-diagnostics-p30-n30-1101T084737Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/diagnostic` | 无 |
| `fa2-check` | `config1-fa2-validation-20261031` | `/home/hdd3/zhangh/workspace/fastsd/experiments/fa2-check` | 无 |
| `humaneval-full` | `config1-humaneval-full-p20-1031T144115Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/humaneval-full` | 无 |
| `humaneval-100` | `config1-humaneval-resident-sdpa-p30-n100-1101T010346Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/humaneval-100` | 无 |
| `humaneval-50-1` | `config1-p20-n50-1031T092354Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/humaneval-50-1` | 无 |
| `humaneval-50-2` | `config1-p20-n50-1031T093637Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/humaneval-50-2` | 无 |
| `round3` | `config1-round3-p25-n30-total-1102T142614Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/round3` | 无 |
| `compare-8b` | `formal-compare-8b-20261025` | `/home/hdd3/zhangh/workspace/fastsd/experiments/compare-8b` | 无 |
| `four-method` | `four-method-suite` | `/home/hdd3/zhangh/workspace/fastsd/experiments/four-method` | 无 |
| `four16` | `four16-p30-n30-1101T165951Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/four16` | 无 |
| `four16-check` | `four16-validation-20261101` | `/home/hdd3/zhangh/workspace/fastsd/experiments/four16-check` | 无 |
| `compare` | `manual-compare-20261026` | `/home/hdd3/zhangh/workspace/fastsd/experiments/compare` | 无 |
| `hetero16` | `manual-hetero16-20261025` | `/home/hdd3/zhangh/workspace/fastsd/experiments/hetero16` | 无 |
| `cpu-pair` | `node3-pair-20261027` | `/home/hdd3/zhangh/workspace/fastsd/experiments/cpu-pair` | 无 |
| `specedge` | `specedge-cpu-gpu-20261026` | `/home/hdd3/zhangh/workspace/fastsd/experiments/specedge` | 无 |
| `draft32-1` | `tri32-8b-p15-28417f158abc` | `/home/hdd3/zhangh/workspace/fastsd/experiments/draft32-1` | 无 |
| `draft32-2` | `tri32-8b-p15-4be26e618a1e` | `/home/hdd3/zhangh/workspace/fastsd/experiments/draft32-2` | 无 |
| `draft32-3` | `tri32-8b-p15-97c6ab3760e7` | `/home/hdd3/zhangh/workspace/fastsd/experiments/draft32-3` | 无 |
| `draft32-4` | `tri32-8b-p15-9a624c5ff5ea` | `/home/hdd3/zhangh/workspace/fastsd/experiments/draft32-4` | 无 |
| `draft32-5` | `tri32-8b-p15-9a7144a0e788` | `/home/hdd3/zhangh/workspace/fastsd/experiments/draft32-5` | 无 |
| `prefetch-1` | `wrr-prefetch-p30-n30-1101T125456Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/prefetch-1` | 无 |
| `prefetch-2` | `wrr-prefetch-p30-n30-1101T134617Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/prefetch-2` | 无 |
| `prefetch-3` | `wrr-prefetch-p30-n30-1101T140124Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/prefetch-3` | 无 |
| `prefetch-4` | `wrr-prefetch-p30-n30-1101T152459Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/prefetch-4` | 无 |
| `prefetch-5` | `wrr-prefetch-p30-n30-1101T161044Z` | `/home/hdd3/zhangh/workspace/fastsd/experiments/prefetch-5` | 无 |
| `prefetch-check` | `wrr-prefetch-validation-20261101` | `/home/hdd3/zhangh/workspace/fastsd/experiments/prefetch-check` | 无 |
