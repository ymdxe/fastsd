"""Rebuild complete or partial summaries from immutable per-request records."""
import csv
import json
from pathlib import Path
import statistics
from common import ROOT,write_json


def stats(values):
    a=sorted(float(v) for v in values if v is not None)
    if not a:return {'n':0,'mean':None,'p50':None,'p95':None,'p99':None}
    def q(p):
        x=(len(a)-1)*p;i=int(x)
        return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
    return {'n':len(a),'mean':statistics.mean(a),'p50':q(.5),'p95':q(.95),'p99':q(.99)}


def load_rows(directory):
    return [json.loads(s) for p in sorted(directory.glob('node*/requests.jsonl')) for s in p.read_text().splitlines() if s.strip()]


def summarize(cell):
    directory=ROOT/cell['id'];rows=load_rows(directory)
    result=dict(cell,completed_requests=len(rows),unique_samples=len({r['global_sample_index'] for r in rows}),output_tokens=sum(r['generated_tokens'] for r in rows))
    status=directory/'status.json'
    result['status']=json.loads(status.read_text())['status'] if status.exists() else 'pending'
    for k in ('arrival_ttft_ms','service_ttft_ms','tpot_ms','arrival_request_e2e_ms','service_request_e2e_ms','arrival_lag_ms'):
        result[k]=stats(r.get(k) for r in rows)
    result['by_node']={node:{k:stats(r.get(k) for r in rows if r['node']==node) for k in ('arrival_ttft_ms','service_ttft_ms','tpot_ms','arrival_lag_ms')} for node in ('node1','node3')}
    for name in ('scheduler_metrics.json','runtime_metrics.json'):
        p=directory/'target'/name
        if p.exists():
            metrics=json.loads(p.read_text());result['scheduler']=metrics.get('fastsd_scheduler',metrics);break
    p=directory/'target/kv_moves.jsonl'
    if p.exists():
        moves=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        async_moves=[r for r in moves if r.get('asynchronous_prefetch')]
        demands=[r for r in moves if r.get('event')=='verify_demand']
        denominator=sum(r['requests'] for r in demands)
        numerator=sum(r['prefetch_demand_hits'] for r in demands)
        result['kv']={'async_move_calls':len(async_moves),'async_move_bytes':sum(r.get('bytes',0) for r in async_moves),
                      'async_move_host_wall_ms':stats(r.get('host_wall_ms') for r in async_moves),
                      'verify_demand_requests':denominator,'verify_prefetch_hits':numerator,'verify_prefetch_hit_rate':numerator/denominator if denominator else None}
    p=directory/'energy.json'
    if p.exists():result['energy']=json.loads(p.read_text())
    write_json(directory/'summary.json',result)
    return result


def main():
    results=[summarize(c) for c in json.loads((ROOT/'matrix.json').read_text())]
    rate_label='/'.join(f'{r:g}' for r in sorted({r['rate'] for r in results}))
    gamma_label='/'.join(str(g) for g in sorted({r['gamma'] for r in results}))
    write_json(ROOT/'results.json',results)
    records=[]
    for r in results:
        energy=r.get('energy',{})
        records.append({'run':r['id'],'status':r['status'],'completed':r['completed_requests'],'tokens':r['output_tokens'],
                        'arrival_ttft_mean_s':(r['arrival_ttft_ms']['mean']/1000 if r['arrival_ttft_ms']['mean'] is not None else None),
                        'service_ttft_mean_s':(r['service_ttft_ms']['mean']/1000 if r['service_ttft_ms']['mean'] is not None else None),
                        'tpot_mean_ms':r['tpot_ms']['mean'],'energy_J':energy.get('energy_j'),'J_per_token':energy.get('joules_per_token'),
                        'kv_prefetch_moved':r.get('scheduler',{}).get('prefetch_async_moved'),'kv_prefetch_failed':r.get('scheduler',{}).get('prefetch_async_failed'),
                        'kv_demand_hit_rate':r.get('kv',{}).get('verify_prefetch_hit_rate')})
    with (ROOT/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# 实验进度与结果','',
           'Qwen3-0.6B CPU FP32 draft：node1/node3 各16个；Qwen3-1.7B FP32 Target：gpunode2 GPU1；完整 GSM8K test 1319题，EOS/1024-token 上限。',
           f'全局泊松 λ={rate_label} 请求/秒，γ={gamma_label}，三种方法共{len(results)}组。尚未完成的组不作为最终结果。','',
           '| Run | 状态 | 完成 | 含排队 TTFT(s) | 服务 TTFT(s) | TPOT(ms) | GPU能耗(J) | 预取搬运 |','|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return '—' if v is None else f'{v:.3f}'
    for r in records:lines.append(f'| {r["run"]} | {r["status"]} | {r["completed"]}/1319 | {fmt(r["arrival_ttft_mean_s"])} | {fmt(r["service_ttft_mean_s"])} | {fmt(r["tpot_mean_ms"])} | {fmt(r["energy_J"])} | {r["kv_prefetch_moved"] if r["kv_prefetch_moved"] is not None else "—"} |')
    lines+=['','能耗只积分首个计划到达至最后一个请求完成的区间，排除模型加载。KV 搬运时间为原有阻塞式 `.to()` 的主机墙钟耗时，不能当作纯 PCIe 传输或 CUDA 重叠证据。',
            'SpecEdge 无原生异步 KV 预取机制，相关指标标为不适用；同步缓存搬运与整卡能耗仍计入其流程。标准SD的预取计数也不代表另加了预取实现。',
            'TTFT 是首批验证输出可见时间；TPOT 是首批可见输出之后的摊销 token 间隔。到达后的 draft 排队时间单独保存，并加入 arrival_ttft_ms。',
            '每个参数组合一次完整运行；共享 CPU 的负载记录保留在节点 status/preflight 中。SSH隧道结果不代表IB网络性能。']
    (ROOT/'RESULTS.zh.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
