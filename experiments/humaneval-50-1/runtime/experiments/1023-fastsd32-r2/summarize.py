from pathlib import Path
import collections,json,statistics
ROOT=Path('/home/zhangh/workspace/fastsd/experiments/1023-fastsd32-r2')
BASE=ROOT.parent/'1023-32draft-comm'
def records(folder,pattern):
 return [json.loads(line) for p in sorted(folder.glob(pattern)) for line in p.read_text().splitlines() if line]
def stats(values):
 v=sorted(float(x) for x in values if x is not None)
 if not v:return None
 def q(a):
  p=(len(v)-1)*a;l=int(p);return v[l]+(v[min(l+1,len(v)-1)]-v[l])*(p-l)
 return dict(count=len(v),min=v[0],mean=statistics.mean(v),p50=q(.5),p95=q(.95),max=v[-1])
run=ROOT/'run32'
r=records(run,'edge_metrics_proc*.jsonl');b=records(BASE/'run32','edge_metrics_proc*.jsonl');rpc=records(run,'rpc-*.jsonl')
w=[json.loads(p.read_text()) for p in sorted(run.glob('worker-*.json'))]
clocks=[json.loads(p.read_text()) for p in sorted(run.glob('clock-*.json'))]
status=json.loads((ROOT/'run32_status.json').read_text())
assert status['exit_status']==0,status
assert len(r)==32 and len(w)==32 and len(clocks)==32
assert {m['global_sample_index'] for m in r}==set(range(32))
assert all(m['profile']=='custom' and m['server_sched_mode']=='fastsd' and m['enable_pipeline'] and m['enable_proactive_draft'] and m['enable_latency_priority'] for m in r)
assert all(m['ok'] for m in rpc)
assert all(0<m['generated_tokens']<=16 and len(m['output_token_ids'])==m['generated_tokens'] for m in r)
assert not any(Path('/proc/'+str(x['pid'])).exists() for x in w)
assert len({x['cpu_affinity'][0] for x in w})==32
mapping=json.loads((ROOT/'worker_map.json').read_text())
assert mapping==json.loads((BASE/'worker_map.json').read_text())
assert (ROOT/'workload32.jsonl').read_bytes()==(BASE/'workload32.jsonl').read_bytes()
for worker in w:
 placement=mapping[worker['worker_id']]
 assert worker['cpu_affinity']==[placement['cpu']] and worker['threads']==1
 assert worker['dtype']=='float16' and 'bind:'+str(placement['numa']) in worker['numa_policies']
base_by_id={x['global_sample_index']:x for x in b}
differences=[dict(sample_index=m['global_sample_index'],baseline=base_by_id[m['global_sample_index']]['output_token_ids'],fastsd=m['output_token_ids']) for m in r if m['output_token_ids']!=base_by_id[m['global_sample_index']]['output_token_ids']]
verify=[x for x in rpc if x['endpoint']=='/verify'];prefill=[x for x in rpc if x['endpoint']=='/prefill'];init=[x for x in rpc if x['endpoint']=='/session/init']
assert len(init)==32 and len(prefill)==32
assert len(verify)==sum(x['rounds'] for x in r)
by_id={x['worker_id']:x for x in w};starts={x['pid']:x['started_monotonic'] for x in init}
start=min(x['barrier_release_monotonic'] for x in w)
end=max(starts[by_id[x['proc_id']]['pid']]+x['request_e2e_ms']/1000 for x in r)
baseline=json.loads((BASE/'communication_summary.json').read_text())
summary={
 'status':'completed','method':'FastSD','clock_setup':'Serialized native five-probe clock calibration, followed by common barrier','first_attempt_completed_requests':18,'first_attempt_failed_requests':14,'profile':'custom','server_sched_mode':'fastsd','enable_pipeline':True,'enable_proactive_draft':True,'enable_latency_priority':True,'pipeline_gamma_adapt':True,'initial_gamma':4,
 'selected_samples':32,'dataset_total_samples':1319,'max_new_tokens':16,'dtype':'float16','target':'node1 Qwen3-1.7B CPU 8 threads','draft':'node3 32 independent Qwen3-1.7B CPU workers, 1 thread each',
 'completed_requests':len(r),'failed_requests':0,'generated_tokens':sum(x['generated_tokens'] for x in r),
 'rpc_count':len(rpc),'rpc_counts':dict(collections.Counter(x['endpoint'] for x in rpc)),
 'accepted_tokens':sum(x['accepted_total'] for x in r),'drafted_tokens':sum(x['drafted_total'] for x in r),
 'acceptance_rate':sum(x['accepted_total'] for x in r)/sum(x['drafted_total'] for x in r),
 'request_e2e_seconds':stats(x['request_e2e_ms']/1000 for x in r),'ttft_seconds':stats(x['ttft_ms']/1000 for x in r),
 'first_draft_seconds':stats(x['first_draft_ms']/1000 for x in r),'prefill_http_ms':stats(x['http_ms'] for x in prefill),
 'verify_http_ms':stats(x['http_ms'] for x in verify),'verify_cloud_total_ms':stats(x['cloud_total_ms'] for x in verify),'verify_compute_ms':stats(x['verify_ms'] for x in verify),
 'verify_http_minus_cloud_ms':stats(x['http_ms']-x['cloud_total_ms'] for x in verify),
 'verify_rounds_per_request':stats(x['rounds'] for x in r),'clock_uncertainty_ms':stats(x['clock_uncertainty_ns']/1e6 for x in clocks),
 'application_request_bytes':sum(x['request_bytes'] for x in r),'application_response_bytes':sum(x['response_bytes'] for x in r),
 'proactive_reuse_hit_rounds':sum(x['reuse_hit_rounds'] for x in r),'proactive_reuse_miss_rounds':sum(x['reuse_miss_rounds'] for x in r),
 'estimated_barrier_to_completion_seconds':end-start,'estimated_aggregate_tokens_per_second':sum(x['generated_tokens'] for x in r)/(end-start),
 'token_identical_requests':32-len(differences),'token_differences':differences,
 'baseline_path':str(BASE),'baseline_request_e2e_seconds':baseline['request_e2e_seconds'],'baseline_verify_http_ms':baseline['verify_http_ms'],
 'baseline_aggregate_tokens_per_second':baseline['estimated_aggregate_tokens_per_second'],
 'request_e2e_median_ratio_baseline_over_fastsd':baseline['request_e2e_seconds']['p50']/statistics.median(x['request_e2e_ms']/1000 for x in r),
 'limits':['单次前32题短输出通信测试，不是全量GSM8K准确率或正式速度评测','两侧同一1.7B模型，接受率不代表32B Target','CPU Target不衡量GPU KV搬运收益','历史基线与本轮背景负载不同，不能将单次耗时差全部归因于算法','FastSD自适应gamma开启，4是初始值；基线gamma固定4','HTTP减服务耗时包含网络与序列化等开销，并非纯网络RTT','公共窗口由session/init开始时间加请求E2E估算，排除模型加载但包含校准与prefill','node1 PyTorch2.6、node3 PyTorch2.2；无warmup']
}
(ROOT/'communication_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
lines=['# FastSD 32 draft 跨机测试','', '日期：2026-10-23。32/32 请求成功，512 个输出 token。','',
'## 配置与步骤','',
'沿用 1023-32draft-comm 的源码快照、模型、FP16 精度、CPU/NUMA 分配和 GSM8K 前 32 条题目。node3 使用 32 个独立 draft，node1 使用 8 线程 CPU Target；每题最多输出 16 tokens。',
'',
'FastSD 参数：`--profile custom --server_sched_mode fastsd --enable_pipeline --enable_proactive_draft --enable_latency_priority --pipeline_gamma_adapt`。初始 γ=4；其余配置见 launch JSON。两端模型为 `/mnt/nfs0/zhangh/models/Qwen3-1.7B`。',
'',
'加载全部模型后以 barrier 同时释放，然后依次执行各 worker 的原生五探针时钟校准，全部校准后再经第二个 barrier 同时进入 arrival、prefill 和 tail-only verify 流程。实验适配器保留上一轮的相同 non-thinking 模板，新增时钟日志、校准启动顺序控制和完成数检查；未修改原生 FastSD 调度、计算和时间校验。',
'', '首次运行见 `../1023-fastsd32/`：并发校准后，10 个 worker 触发负 pull 时长校验，4 个收到 HTTP 422，只完成 18/32。诊断复现了负 upload 校验；本目录为串行校准后的重跑，初次失败记录完整保留。', '', '## 与上一轮比较','', '| 指标 | Standard SD | FastSD |','| --- | ---: | ---: |']
for label,key,unit in [('请求 E2E 中位数','request_e2e_seconds','秒'),('请求 TTFT 中位数','ttft_seconds','秒'),('首轮 draft 中位数','first_draft_seconds','秒'),('verify HTTP 中位数','verify_http_ms','ms'),('Target verify 计算中位数','verify_compute_ms','ms'),('HTTP 减 Target 服务耗时中位数','verify_http_minus_cloud_ms','ms')]:
 lines.append(f'| {label} | {baseline[key]["p50"]:.2f} {unit} | {summary[key]["p50"]:.2f} {unit} |')
lines += [f'| RPC 总数（不含健康检查与时钟校准） | {baseline["rpc_count"]} | {summary["rpc_count"]} |',f'| 验证轮次 | {baseline["rpc_counts"]["/verify"]} | {len(verify)} |',f'| 估计总吞吐 | {baseline["estimated_aggregate_tokens_per_second"]:.3f} tokens/s | {summary["estimated_aggregate_tokens_per_second"]:.3f} tokens/s |','',
 f'本轮输出 token 与上一轮逐题完全一致的请求：**{32-len(differences)}/32**。FastSD 提前草拟复用命中 {summary["proactive_reuse_hit_rounds"]} 轮，未命中 {summary["proactive_reuse_miss_rounds"]} 轮。原生指标中的接受数/候选数为 {summary["accepted_tokens"]}/{summary["drafted_tokens"]}。',
'',f'时钟校准不确定度最大 {summary["clock_uncertainty_ms"]["max"]:.3f} ms；保持项目默认 10 ms 上限。',
'', '## 限制','']
lines += ['- '+x for x in summary['limits']]
lines += ['', '## 文件','',
'- `source/`：与基线相同的源码快照；未修改原仓库推理源码。',
'- `launch_target.py`、`launch_drafts.py`、`*_launch.json`：启动脚本、命令和环境。',
'- `run32/edge_metrics_proc*.jsonl`：逐请求输出、耗时、接受率与复用计数。',
'- `run32/rpc-*.jsonl`、`run32/clock-*.json`：逐次通信与时钟校准记录。',
'- `communication_summary.json`：汇总与逐题输出一致性结果。',
'- node1 同名目录 `target/scheduler_metrics.json`：正常退出时保存的 FastSD 调度证据；`target.log`：服务日志。',
'', '运行界面：tmux 会话 6，node1 为 6:0.0，node3 为 6:0.1。','']
(ROOT/'README.md').write_text('\n'.join(lines))
print('=== FASTSD SUMMARY ===',json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
