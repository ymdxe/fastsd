"""Build the fixed-gamma comparison after both remote runs complete."""
from pathlib import Path
import json
BASE=Path('/home/zhangh/workspace/fastsd/experiments')
SD=BASE/'1023-sd-g20';FAST=BASE/'1023-fastsd-g20'
def load_rows(root):
 return {x['global_sample_index']:x for p in (root/'run32').glob('edge_metrics_proc*.jsonl') for x in [json.loads(line) for line in p.read_text().splitlines()]}
a=json.loads((SD/'communication_summary.json').read_text());b=json.loads((FAST/'communication_summary.json').read_text())
a_rows=load_rows(SD);b_rows=load_rows(FAST)
assert a['requests']==b['requests']==32 and a['gamma']==b['gamma']==20
assert (SD/'workload32.jsonl').read_bytes()==(FAST/'workload32.jsonl').read_bytes()
assert (SD/'worker_map.json').read_bytes()==(FAST/'worker_map.json').read_bytes()
diffs=[{'sample_index':i,'standard':a_rows[i]['output_token_ids'],'fastsd':b_rows[i]['output_token_ids']} for i in range(32) if a_rows[i]['output_token_ids']!=b_rows[i]['output_token_ids']]
comparison={'gamma':20,'max_new_tokens':64,'adaptive_gamma':False,'matched_output_requests':32-len(diffs),'mismatched_outputs':diffs,'standard_sd':a,'fastsd':b,'request_e2e_median_reduction_percent':100*(1-b['request_e2e_seconds']['p50']/a['request_e2e_seconds']['p50']),'output_throughput_change_percent':100*(b['estimated_output_tokens_per_second']/a['estimated_output_tokens_per_second']-1),'worker_completion_throughput_change_percent':100*(b['tokens_per_second_including_worker_completion']/a['tokens_per_second_including_worker_completion']-1)}
text=['# 固定 γ=20：Standard SD 与 FastSD','',
'日期：2026-10-23。两种方法各 32/32 请求完成，使用同一批 GSM8K 前 32 条题目。','',
'## 配置','',
'- node3：32 个独立 Qwen3-1.7B FP16 draft，每 worker 一个物理核心、一个计算线程；4 个 NUMA 节点各 8 个 worker，绑定节点内存。',
'- node1：Qwen3-1.7B FP16 CPU Target，8 个计算线程；batch_size=8，token_budget=256。',
'- 两种方法固定 gamma=20，greedy，正常 EOS，max_new_tokens=64，无 warmup；依次执行，源代码、CPU/NUMA 映射、问题和 non-thinking 模板相同。',
'- FastSD 使用 custom/fastsd、pipeline、proactive drafting、latency priority；关闭 gamma adaptation，min/max 同设为20。逐次 RPC 记录确认实际 verify gamma 均为20。',
'- FastSD 沿用上轮验证通过的顺序时钟校准和校准后统一放行；保留所有原生时间校验。',
'- 上一轮上限16小于本次gamma20，因此本轮将两侧上限统一提高为64。不得将本轮与旧的16-token测试直接比较。',
'', '## 结果','', '| 指标 | Standard SD | FastSD |','| --- | ---: | ---: |']
for label,key,unit in [('请求 E2E 中位数','request_e2e_seconds','秒'),('请求 TTFT 中位数','ttft_seconds','秒'),('首轮 draft 中位数','first_draft_seconds','秒'),('prefill HTTP 中位数','prefill_http_ms','ms'),('verify HTTP 中位数','verify_http_ms','ms'),('Target verify 计算中位数','verify_compute_ms','ms'),('HTTP 减 Target 服务时间中位数','verify_http_minus_cloud_ms','ms')]:
 text.append(f'| {label} | {a[key]["p50"]:.2f} {unit} | {b[key]["p50"]:.2f} {unit} |')
for label,key in [('请求 E2E P95（秒）','request_e2e_seconds'),('verify HTTP P95（ms）','verify_http_ms')]:
 text.append(f'| {label} | {a[key]["p95"]:.2f} | {b[key]["p95"]:.2f} |')
for label,key in [('输出 tokens','generated_tokens'),('协议 RPC 数','rpc_count'),('提前草拟复用命中','proactive_reuse_hit_rounds')]:
 text.append(f'| {label} | {a[key]} | {b[key]} |')
text.extend([f'| verify 次数 | {a["rpc_counts"]["/verify"]} | {b["rpc_counts"]["/verify"]} |',f'| 草稿接受率 | {a["acceptance_rate"]:.2%} | {b["acceptance_rate"]:.2%} |'])
for label,key in [('估计最后输出完成窗口（秒）','estimated_output_window_seconds'),('估计输出吞吐（tokens/s）','estimated_output_tokens_per_second'),('全部 worker 完成窗口（秒）','worker_completion_window_seconds'),('包含 worker 收尾的吞吐（tokens/s）','tokens_per_second_including_worker_completion'),('最后输出到全部 worker 完成间隔（秒）','all_workers_finish_minus_last_output_seconds')]:
 text.append(f'| {label} | {a[key]:.3f} | {b[key]:.3f} |')
text += ['',f'逐题完整输出 token 一致：**{32-len(diffs)}/32**。差异明细（若有）保存在 comparison.json。',
'', '## 统计口径与限制','',
'- 输出窗口从全部模型就绪时的 barrier 起算，结束时间由每个 session/init 开始时刻加原生请求 E2E 估算；包括启动校准、prefill 和 decode，排除模型加载。',
'- worker 窗口使用进程内完成时间戳，包含原生执行器等待尚未完成工作的收尾时间。两种口径均报告，避免隐藏提前草拟的额外工作。',
'- 候选块保持固定gamma20，最后一块可能超过剩余输出额度；输出按64-token/EOS截断，因此接受的候选数可能超过最终保留的输出数。',
'- HTTP总耗时包含Target排队和计算；HTTP减服务耗时也包含序列化等开销，不是纯网络RTT。',
'- 两侧为同一1.7B模型，接受率不代表未来32B Target；CPU预取计数不能代表GPU传输收益。',
'- 共享服务器，单次32题短输出测试，后台负载随时间变化；不能作为正式加速结论或GSM8K准确率。',
'', '## 复现与文件','',
'两个目录均包含 launch_target.py、launch_drafts.py、*_launch.json、commands.txt、manifest.json、source/、workload32.jsonl、worker_map.json、preflight.txt、progress.jsonl、run32.log 和逐 worker/RPC/输出记录。',
'',
'对照目录：`../1023-sd-g20/` 与 `../1023-fastsd-g20/`。node1 对应目录保存 target.log、Target 启动环境及正常关闭时的调度统计。运行界面均使用 tmux 会话6；实验结束后关闭临时进程，保留窗格。','']
for root in [SD,FAST]:
 (root/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
 (root/'README.md').write_text('\n'.join(text))
print('=== GAMMA20 COMPARISON ===',json.dumps({k:v for k,v in comparison.items() if k not in ['standard_sd','fastsd']},ensure_ascii=False),flush=True)
