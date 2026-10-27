"""Reuse the validated FP32 harness; change only gamma and the rate matrix."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'1024-fp32-g100-l10-cpu32'


def replace(path,before,after):
    text=path.read_text()
    assert before in text,(path,before)
    path.write_text(text.replace(before,after))


def main():
    assert not (ROOT/'source').exists(),'Do not overwrite a snapshot'
    for name in ('common.py','coordinator.py','campaign.py','report.py','target.py','telemetry.py','node_agent.py','native_worker.py','finalize.py','start_control.py','bundle.py','allocation.py','run_gpu.py','validate_config.py','check_control.py'):
        shutil.copy2(OLD/name,ROOT/name)
    for name in ('source','specedge'):
        shutil.copytree(OLD/name,ROOT/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('node1-worker-map.json','node3-worker-map.json'):
        shutil.copy2(OLD/name,ROOT/name)
    (ROOT/'workloads').mkdir()
    for rate in (1,10):
        shutil.copy2(ROOT.parent/'1024-poisson32-qwen06-17'/f'workloads/lambda-{rate}.jsonl',ROOT/'workloads'/f'lambda-{rate}.jsonl')
    matrix=[{'id':f'{method}-g4-l{rate}','method':method,'gamma':4,'rate':float(rate),'cap':1024,'expected_requests':1319}
            for rate in (1,10) for method in ('fastsd','standard','specedge')]
    (ROOT/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    manifest=json.loads((OLD/'manifest.json').read_text())
    manifest.pop('hashes',None)
    manifest.update(status='prepared',gamma=[4],rates_global_rps=[1.,10.],planned_runs=6,planned_requests=7914,
                    source_snapshot_parent=str(OLD),specedge_generation='single chain: beams1, branch_width1, draft_budget4, proactive_chain1')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    text=(OLD/'bootstrap_node.py.txt').read_text()
    text=text.replace('1024-fp32-g100-l10-cpu32','1024-fp32-g4-l1-10-cpu32').replace('1024-poisson32-qwen06-17','1024-fp32-g100-l10-cpu32')
    (ROOT/'bootstrap_node.py.txt').write_text(text)
    replace(ROOT/'check_control.py',"'fastsd-g100-l10'","'fastsd-g4-l1'")
    validate=ROOT/'validate_config.py'
    replace(validate,"assert len(matrix)==3 and {(r['method'],r['gamma'],r['rate']) for r in matrix}=={(m,100,10.) for m in ('standard','fastsd','specedge')}",
            "assert len(matrix)==6 and {(r['method'],r['gamma'],r['rate']) for r in matrix}=={(m,4,rate) for m in ('standard','fastsd','specedge') for rate in (1.,10.)}")
    before="""    samples=[json.loads(s) for s in (ROOT/'workloads/lambda-10.jsonl').read_text().splitlines() if s.strip()]
    assert len(samples)==1319 and {r['source_index'] for r in samples}==set(range(1319))
    assert all(a['scheduled_arrival_s']<b['scheduled_arrival_s'] for a,b in zip(samples,samples[1:]))
    old=ROOT.parent/'1024-poisson32-qwen06-17'
    assert hashlib.sha256((ROOT/'workloads/lambda-10.jsonl').read_bytes()).digest()==hashlib.sha256((old/'workloads/lambda-10.jsonl').read_bytes()).digest()
"""
    after="""    old=ROOT.parent/'1024-poisson32-qwen06-17'
    schedules={}
    for rate in (1,10):
        path=ROOT/'workloads'/f'lambda-{rate}.jsonl'
        samples=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        assert len(samples)==1319 and {r['source_index'] for r in samples}==set(range(1319))
        assert all(a['scheduled_arrival_s']<b['scheduled_arrival_s'] for a,b in zip(samples,samples[1:]))
        assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256((old/'workloads'/path.name).read_bytes()).digest()
        schedules[str(rate)]=samples[-1]['scheduled_arrival_s']
    agent=(ROOT/'node_agent.py').read_text()
    for setting in ("'SPECEDGE_MAX_N_BEAMS':1", "'SPECEDGE_MAX_BRANCH_WIDTH':1", "'SPECEDGE_PROACTIVE_MAX_N_BEAMS':1", "'SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH':1"):
        assert setting in agent
"""
    replace(validate,before,after)
    replace(validate,'width=101;max_len=3072','width=5;max_len=3072')
    replace(validate,'parent_indices=bytes(100*8)','parent_indices=bytes(4*8)')
    replace(validate,"'matrix_cells':3,'dataset_requests':1319,'total_requests':3957","'matrix_cells':6,'dataset_requests':1319,'total_requests':7914")
    replace(validate,"'last_scheduled_arrival_s':samples[-1]['scheduled_arrival_s']","'last_scheduled_arrival_s_by_rate':schedules,'specedge_single_chain':True")
    (ROOT/'preflight').mkdir()
    probe=(ROOT.parents[1]/'tmp/fp32-resource.py').read_text()
    prefix='''"""Inspect resources after the preceding FP32 task has exited."""
import json,os
from pathlib import Path
import socket,subprocess,time
OLD=Path('/home/zhangh/workspace/fastsd/experiments/1024-fp32-g100-l10-cpu32')
node='node3' if 'node3' in socket.gethostname() else 'node1'
agent=json.loads((OLD/(node+'-agent.json')).read_text())['pid']
for item in [{'pid':agent},*json.loads((OLD/'fastsd-g100-l10'/node/'launch.json').read_text())['workers']]:
    path=Path(f'/proc/{item["pid"]}/cmdline')
    assert not path.exists() or str(OLD).encode() not in path.read_bytes(),'Old task still running'
'''
    (ROOT/'preflight_cpu.py.txt').write_text(prefix+probe[probe.index('def read_cpu():'):])
    print('PREPARED',len(matrix),'cells',sum(c['expected_requests'] for c in matrix),'requests')


if __name__=='__main__':main()
