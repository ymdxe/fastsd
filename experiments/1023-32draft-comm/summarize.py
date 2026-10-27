"""Summarize the completed 32-worker cross-host communication smoke test."""
from pathlib import Path
import collections
import hashlib
import json
import math
import os
import statistics
import time

ROOT=Path('/home/zhangh/workspace/fastsd/experiments/1023-32draft-comm')
RUN=ROOT/'run32'

def stats(values):
    values=sorted(float(v) for v in values if v is not None)
    if not values:return None
    def quantile(q):
        position=(len(values)-1)*q
        left=int(position);right=min(left+1,len(values)-1)
        return values[left]+(values[right]-values[left])*(position-left)
    return {'count':len(values),'min':values[0],'mean':statistics.mean(values),'p50':quantile(.5),'p95':quantile(.95),'max':values[-1]}

status=json.loads((ROOT/'run32_status.json').read_text())
records=[json.loads(line) for p in sorted(RUN.glob('edge_metrics_proc*.jsonl')) for line in p.read_text().splitlines() if line]
workers=[json.loads(p.read_text()) for p in sorted(RUN.glob('worker-*.json'))]
rpcs=[json.loads(line) for p in sorted(RUN.glob('rpc-*.jsonl')) for line in p.read_text().splitlines() if line]
assert status['exit_status']==0,status
assert len(records)==32 and len(workers)==32,(len(records),len(workers))
assert {r['global_sample_index'] for r in records}==set(range(32))
assert {r['proc_id'] for r in records}==set(range(32))
assert all(r['ok'] for r in rpcs)
assert len({w['cpu_affinity'][0] for w in workers})==32
assert all(w['threads']==1 and w['dtype']=='float16' and w['device']=='cpu' for w in workers)
mapping=json.loads((ROOT/'worker_map.json').read_text())
for worker in workers:
    placement=mapping[worker['worker_id']]
    assert worker['cpu_affinity']==[placement['cpu']]
    assert 'bind:'+str(placement['numa']) in worker['numa_policies'],worker
assert all(0<r['generated_tokens']<=16 and len(r['output_token_ids'])==r['generated_tokens'] for r in records)
remaining=[w['pid'] for w in workers if Path('/proc/'+str(w['pid'])).exists()]
assert not remaining,remaining
verify=[r for r in rpcs if r['endpoint']=='/verify']
prefill=[r for r in rpcs if r['endpoint']=='/prefill']
init=[r for r in rpcs if r['endpoint']=='/session/init']
assert len(init)==32 and len(prefill)==32
assert len(verify)==sum(r['rounds'] for r in records)
health=json.loads((ROOT/'health_before.json').read_text())
starts={r['pid']:r['started_monotonic'] for r in init}
by_worker={w['worker_id']:w for w in workers}
# Edge request_start immediately precedes /session/init. Derived endpoint differs only by local bookkeeping.
end_estimate=max(starts[by_worker[r['proc_id']]['pid']]+r['request_e2e_ms']/1000 for r in records)
barrier=min(w['barrier_release_monotonic'] for w in workers)
window=end_estimate-barrier
summary={
 'status':'completed','dataset':'gsm8k','source_total_rows':1319,'selected_rows':32,'selection':'first 32 source rows, one per draft worker',
 'target_host':'node1','draft_host':'node3','target_url':'http://10.66.0.3:18601','target_model':'Qwen3-1.7B','draft_model':'Qwen3-1.7B','dtype':'float16',
 'mode':'vanilla/Standard SD greedy protocol, no proactive drafting, no pipeline','gamma':4,'max_new_tokens':16,'warmup_requests':0,
 'completed_requests':len(records),'failed_requests':0,'draft_workers':len(workers),'target_threads':8,'draft_threads_per_worker':1,
 'generated_tokens':sum(r['generated_tokens'] for r in records),'drafted_tokens':sum(r['drafted_total'] for r in records),'accepted_draft_tokens':sum(r['accepted_total'] for r in records),
 'acceptance_rate':sum(r['accepted_total'] for r in records)/sum(r['drafted_total'] for r in records),
 'rpc_count':len(rpcs),'rpc_counts':dict(collections.Counter(r['endpoint'] for r in rpcs)),
 'health_http_ms_first':health[0]['http_ms'],'health_http_ms_reused':stats(r['http_ms'] for r in health[1:]),
 'verify_http_ms':stats(r['http_ms'] for r in verify),'verify_cloud_total_ms':stats(r['cloud_total_ms'] for r in verify),'verify_compute_ms':stats(r['verify_ms'] for r in verify),
 'verify_http_minus_cloud_ms':stats(r['http_ms']-r['cloud_total_ms'] for r in verify),
 'verify_cloud_minus_compute_ms':stats(r['cloud_total_ms']-r['verify_ms'] for r in verify),
 'prefill_http_ms':stats(r['http_ms'] for r in prefill),'session_init_http_ms':stats(r['http_ms'] for r in init),
 'request_e2e_seconds':stats(r['request_e2e_ms']/1000 for r in records),'ttft_seconds':stats(r['ttft_ms']/1000 for r in records),
 'first_draft_seconds':stats(r['first_draft_ms']/1000 for r in records),
 'verify_rounds_per_request':stats(r['rounds'] for r in records),
 'application_request_bytes':sum(r['request_bytes'] for r in records),'application_response_bytes':sum(r['response_bytes'] for r in records),
 'estimated_barrier_to_completion_seconds':window,'estimated_aggregate_tokens_per_second':sum(r['generated_tokens'] for r in records)/window,
 'common_window_note':'All times measured on node3. End derived from /session/init start + Edge request E2E; includes prefill and local draft, excludes model loading. Not a formal performance measurement.',
 'barrier_release_spread_ms':(max(w['barrier_release_monotonic'] for w in workers)-barrier)*1000,
 'remaining_draft_pids':remaining,
 'limits':['32/1319 samples; truncated 16-token communication test, not accuracy evaluation','Same checkpoint used on both ends; acceptance not representative of 32B target','node1 torch2.6 vs node3 torch2.2; do not attribute compute gap only to CPU hardware','HTTP residual includes serialization, transport and runtime overhead; not pure network RTT','One-way push/pull fields are unavailable for vanilla unsynchronized clocks; zero placeholders in native metrics must be read as NA','Shared server; CPU affinity is not exclusive reservation','No warmup; loading synchronized but model compute cold']
}
(ROOT/'communication_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
lines=['# 32 draft 跨机通信测试','',
'日期：2026-10-23。32/32 请求成功，32 个 draft 进程均已退出。',
'', '## 实验内容','',
'node3 启动 32 个独立 Qwen3-1.7B draft；node1 启动一个 Qwen3-1.7B 临时 Target。复用项目 vanilla/Standard SD 的真实 session/init → prefill → verify → 接受/回滚流程，测试通信与并发。',
'', '## 配置与步骤','',
'- 两侧模型均为 `/mnt/nfs0/zhangh/models/Qwen3-1.7B`，FP16，greedy，γ=4。',
'- node3 每 worker 一个独立物理核心、一个计算线程，4 个 NUMA 节点各 8 个，内存绑定同节点。选择和核验记录见 worker_map.json、cpu_selection.json、run32/worker-*.json。',
'- node1 使用 NUMA 0 的 CPU 0,2,4,6,8,10,12,14，8 个计算线程；HTTP `10.66.0.3:18601`，内网直连，没有 SSH 隧道。',
'- 数据：用户指定 `/home/hdd3/zhangh/workspace/fastsd/data/gsm8k.jsonl`，SHA256 见 manifest；共 1319 条。本次取前 32 条，每 worker 一条，最多输出 16 tokens。保留原题，使用 Target non-thinking chat 模板，不添加 8-shot 示例。',
'- 先进行 1 worker、4-token 验收，然后加载 32 份独立 draft 权重，通过 barrier 统一释放。没有跨进程共享权重，也没有正式 warmup。',
'- 每次 RPC 记录本机单调时钟和 Target 返回的服务耗时；使用同一节点的时钟计算时长，不相减不同主机的 monotonic 时间。',
'- 单 worker 阶段使用 mt_bench 编码标志调用项目 non-thinking 模板，数据仍为 GSM8K；32-worker 阶段实验适配器显式实现同一模板，dataset 标签为 gsm8k。',
'', '## 实验结果','',
'| 指标 | 结果 |','| --- | --- |',
f'| 完成请求 | {len(records)}/32 |',
f'| 有效输出 tokens | {summary["generated_tokens"]} |',
f'| 验证轮次 | {len(verify)} |',
f'| draft 接受率 | {summary["acceptance_rate"]*100:.2f}%（同模型临时 Target） |',
f'| 请求 E2E p50 / p95 | {summary["request_e2e_seconds"]["p50"]:.2f} / {summary["request_e2e_seconds"]["p95"]:.2f} 秒 |',
f'| 首次 draft 耗时 p50 / p95 | {summary["first_draft_seconds"]["p50"]:.2f} / {summary["first_draft_seconds"]["p95"]:.2f} 秒 |',
f'| TTFT p50 / p95 | {summary["ttft_seconds"]["p50"]:.2f} / {summary["ttft_seconds"]["p95"]:.2f} 秒 |',
f'| verify HTTP 总耗时 p50 / p95 | {summary["verify_http_ms"]["p50"]:.2f} / {summary["verify_http_ms"]["p95"]:.2f} ms |',
f'| Target verify 计算 p50 / p95 | {summary["verify_compute_ms"]["p50"]:.2f} / {summary["verify_compute_ms"]["p95"]:.2f} ms |',
f'| HTTP 减 Target 服务耗时 p50 / p95 | {summary["verify_http_minus_cloud_ms"]["p50"]:.2f} / {summary["verify_http_minus_cloud_ms"]["p95"]:.2f} ms |',
f'| 空载复用连接 HTTP p50 / p95 | {summary["health_http_ms_reused"]["p50"]:.2f} / {summary["health_http_ms_reused"]["p95"]:.2f} ms |',
f'| 应用层请求 / 响应 JSON 字节 | {summary["application_request_bytes"]} / {summary["application_response_bytes"]} |',
f'| 估计公共窗口 / 总吞吐 | {window:.2f} 秒 / {summary["estimated_aggregate_tokens_per_second"]:.3f} tokens/s |',
'', '## 解释与限制','']
lines += ['- '+limit for limit in summary['limits']]
lines += ['', '本轮可确认 32 个独立 draft 与跨机 CPU Target 的真实请求和验证流程成功，网络、服务计算和本地草拟耗时可追溯。不能据此评价 Qwen3-32B、FastSD 调度收益或 GSM8K 答题准确率。',
'', '## 文件','',
'- `manifest.json`、`commands.txt`、`*_launch.json`：配置、原始数据 hash、启动命令。',
'- `source/`：同一版本代码快照；`edge_comm.py`：实验专用绑核、NUMA、启动屏障、模板和 RPC 记录适配器。',
'- `workload32.jsonl`：本次 32 条原题和参考答案。',
'- `run32/edge_metrics_proc*.jsonl`：逐请求输出 token、文本、接受率和耗时。',
'- `run32/rpc-*.jsonl`：逐次 HTTP 请求日志。',
'- `communication_summary.json`：从原始记录生成的完整汇总。',
'- `smoke1/`：单 worker 冒烟记录；`run32.log`：32 worker 完整输出。',
'- node1 同名目录保存 Target 配置与 `target.log`。',
'', '运行界面均在 tmux 会话 `6`：`6:0.0` 为 node1 Target，`6:0.1` 为 node3 draft。用户的 `6:0.2` 保持原状。','']
(ROOT/'README.md').write_text('\n'.join(lines))
print('=== COMMUNICATION SUMMARY ===',flush=True)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
print('=== ALL 32 WORKERS VERIFIED ===',flush=True)
