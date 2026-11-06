import os
os.environ["G20_RUN"]="1023-sd-g20"
"""Summarize one gamma=20 run from raw request, RPC, and worker records."""
from pathlib import Path
import collections,json,os,statistics,sys
ROOT=Path('/home/zhangh/workspace/fastsd/experiments')/os.environ.get('G20_RUN','1023-sd-g20')

def rows(path,pattern):
 return [json.loads(line) for p in sorted(path.glob(pattern)) for line in p.read_text().splitlines() if line]
def stats(values):
 v=sorted(float(x) for x in values if x is not None)
 if not v:return None
 def q(a):
  p=(len(v)-1)*a;i=int(p);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(p-i)
 return dict(count=len(v),min=v[0],mean=statistics.mean(v),p50=q(.5),p95=q(.95),max=v[-1])
run=ROOT/'run32';manifest=json.loads((ROOT/'manifest.json').read_text())
r=rows(run,'edge_metrics_proc*.jsonl');rpc=rows(run,'rpc-*.jsonl')
w=[json.loads(p.read_text()) for p in sorted(run.glob('worker-*.json'))]
done=[json.loads(p.read_text()) for p in sorted(run.glob('completion-*.json'))]
status=json.loads((ROOT/'run32_status.json').read_text())
assert status['exit_status']==0,status
assert len(r)==len(w)==len(done)==32
assert all(x['success'] for x in done)
assert {x['global_sample_index'] for x in r}==set(range(32))
assert all(x['ok'] for x in rpc)
assert all(0<x['generated_tokens']<=64 and len(x['output_token_ids'])==x['generated_tokens'] for x in r)
assert not any(Path('/proc/'+str(x['pid'])).exists() for x in w)
assert len({x['cpu_affinity'][0] for x in w})==32
mapping=json.loads((ROOT/'worker_map.json').read_text())
for worker in w:
 place=mapping[worker['worker_id']]
 assert worker['cpu_affinity']==[place['cpu']] and worker['threads']==1
 assert worker['dtype']=='float16' and 'bind:'+str(place['numa']) in worker['numa_policies']
fast=manifest['method']=='FastSD'
assert all(x['server_sched_mode']==('fastsd' if fast else 'vanilla') and x['enable_pipeline']==fast and x['enable_proactive_draft']==fast and x['enable_latency_priority']==fast for x in r)
verify=[x for x in rpc if x['endpoint']=='/verify'];prefill=[x for x in rpc if x['endpoint']=='/prefill'];init=[x for x in rpc if x['endpoint']=='/session/init']
assert len(init)==len(prefill)==32
assert len(verify)==sum(x['rounds'] for x in r)
assert all(x['gamma']==20 for x in verify)
assert all(x['suggested_gamma'] in (None,20) for x in verify)
cmd=json.loads((ROOT/'run32_launch.json').read_text())['command']
assert cmd[cmd.index('--gamma')+1]=='20' and cmd[cmd.index('--max_tokens')+1]=='64' and '--no-pipeline_gamma_adapt' in cmd
starts={x['pid']:x['started_monotonic'] for x in init};by_id={x['worker_id']:x for x in w}
barrier=min(x['barrier_release_monotonic'] for x in w)
end=max(starts[by_id[x['proc_id']]['pid']]+x['request_e2e_ms']/1000 for x in r)
worker_end=max(x['completed_monotonic'] for x in done)
tokens=sum(x['generated_tokens'] for x in r)
clocks=[json.loads(p.read_text()) for p in run.glob('clock-*.json')]
if fast:assert len(clocks)==32
old=ROOT.parent/'1023-32draft-comm'
old_rows={x['global_sample_index']:x for x in rows(old/'run32','edge_metrics_proc*.jsonl')}
old_prefix_match=sum(x['output_token_ids'][:16]==old_rows[x['global_sample_index']]['output_token_ids'] for x in r)
s={
 'status':'completed','method':manifest['method'],'gamma':20,'max_new_tokens':64,'adaptive_gamma':False,'requests':32,'failed_requests':0,'generated_tokens':tokens,'output_tokens_per_request':stats(x['generated_tokens'] for x in r),
 'rpc_count':len(rpc),'rpc_counts':dict(collections.Counter(x['endpoint'] for x in rpc)),'observed_verify_gamma_counts':dict(collections.Counter(x['gamma'] for x in verify)),
 'request_e2e_seconds':stats(x['request_e2e_ms']/1000 for x in r),'ttft_seconds':stats(x['ttft_ms']/1000 for x in r),'first_draft_seconds':stats(x['first_draft_ms']/1000 for x in r),
 'prefill_http_ms':stats(x['http_ms'] for x in prefill),'verify_http_ms':stats(x['http_ms'] for x in verify),'verify_cloud_total_ms':stats(x['cloud_total_ms'] for x in verify),'verify_compute_ms':stats(x['verify_ms'] for x in verify),'verify_http_minus_cloud_ms':stats(x['http_ms']-x['cloud_total_ms'] for x in verify),
 'accepted_draft_tokens':sum(x['accepted_total'] for x in r),'drafted_tokens':sum(x['drafted_total'] for x in r),'acceptance_rate':sum(x['accepted_total'] for x in r)/sum(x['drafted_total'] for x in r),
 'verify_rounds_per_request':stats(x['rounds'] for x in r),'proactive_reuse_hit_rounds':sum(x['reuse_hit_rounds'] for x in r),'proactive_reuse_miss_rounds':sum(x['reuse_miss_rounds'] for x in r),
 'application_request_bytes':sum(x['request_bytes'] for x in r),'application_response_bytes':sum(x['response_bytes'] for x in r),
 'clock_uncertainty_ms':stats(x['clock_uncertainty_ns']/1e6 for x in clocks),
 'estimated_output_window_seconds':end-barrier,'estimated_output_tokens_per_second':tokens/(end-barrier),
 'worker_completion_window_seconds':worker_end-barrier,'tokens_per_second_including_worker_completion':tokens/(worker_end-barrier),
 'all_workers_finish_minus_last_output_seconds':worker_end-end,
 'matching_first_16_tokens_vs_old_standard':old_prefix_match,
 'timing_definition':'Both windows start at model-ready barrier. Output window ends at max(session/init start + native request E2E), an estimate. Worker window ends at return from native per-worker implementation, including outstanding executor work. Model loading excluded; clock startup and prefill included.',
 'limits':['32 questions, one repeat, same 1.7B checkpoint on both hosts; not GSM8K accuracy evaluation','gamma20 with cap64 cannot be compared directly against prior gamma4/cap16 tests','FastSD gamma adaptation disabled for this fixed-gamma comparison; other FastSD features remain enabled','Shared CPU servers; affinity is not exclusive and background load can vary','CPU Target does not measure GPU KV transfer benefit','Full gamma drafts can extend beyond the remaining output budget; accepted candidates can exceed retained output tokens','HTTP minus cloud service includes transport and serialization overhead, not pure network RTT']
}
(ROOT/'communication_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
manifest.update(status='completed',completed_requests=32,failed_requests=0,generated_tokens=tokens,observed_verify_gammas=[20]);(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('=== GAMMA20 SUMMARY ===',json.dumps(s,ensure_ascii=False,indent=2),flush=True)
