"""Change only Poisson rates, preserving live lambda1 and all inference settings."""
import ast
import hashlib
import json
from pathlib import Path
import time
from common import ROOT,write_json

UPDATE=ROOT/'rate-update-1-10-50'
BACKUP=UPDATE/'backups'


def main():
    old=json.loads((BACKUP/'matrix.json').read_text())
    status=json.loads((ROOT/'status.json').read_text())
    assert status['status']=='running' and status['cell']['rate']==1
    assert not (ROOT/'standard-g20-l1/status.json').exists()
    mapping={1.:1.,.5:10.,.1:50.}
    matrix=[]
    for cell in old:
        new=dict(cell);new['rate']=mapping[cell['rate']]
        if new['rate']!=cell['rate']:new['id']=f'{new["method"]}-g{new["gamma"]}-l{new["rate"]:g}'
        assert {k:v for k,v in cell.items() if k not in ('rate','id')}=={k:v for k,v in new.items() if k not in ('rate','id')}
        matrix.append(new)
    assert len(matrix)==27 and len({c['id'] for c in matrix})==27
    assert {(c['method'],c['gamma'],c['rate']) for c in matrix}=={(m,g,r) for m in ('standard','fastsd','specedge') for g in (20,200,500) for r in (1.,10.,50.)}
    assert next(c for c in matrix if c['id']==status['cell']['id'])==status['cell']
    canonical=ROOT/'workloads/lambda-1.jsonl';canonical_hash=hashlib.sha256(canonical.read_bytes()).hexdigest()
    samples=[json.loads(s) for s in canonical.read_text().splitlines() if s.strip()]
    assert len(samples)==1319
    for rate in (10.,50.):
        path=ROOT/'workloads'/f'lambda-{rate:g}.jsonl'
        assert not path.exists(),path
        rows=[dict(r,scheduled_arrival_s=r['scheduled_arrival_s']/rate) for r in samples]
        assert all(a['scheduled_arrival_s']<b['scheduled_arrival_s'] for a,b in zip(rows,rows[1:]))
        assert all({k:v for k,v in a.items() if k!='scheduled_arrival_s'}=={k:v for k,v in b.items() if k!='scheduled_arrival_s'} for a,b in zip(samples,rows))
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    assert hashlib.sha256(canonical.read_bytes()).hexdigest()==canonical_hash
    for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    write_json(ROOT/'matrix.json',matrix)
    validation=json.loads((ROOT/'validation.json').read_text())
    times={str(r):samples[-1]['scheduled_arrival_s']/r for r in (1.,10.,50.)}
    lower_hours=sum(times.values())*9/3600
    validation.update(scheduled_last_arrival_s=times,minimum_serial_schedule_hours=lower_hours,rate_revision=3)
    write_json(ROOT/'validation.json',validation)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    manifest.update(rates_global_rps=[1.,10.,50.],rate_revision=3,rates_updated_epoch=time.time(),
                    previous_rates_global_rps=[.1,.5,1.],rate_update_policy='Finish current lambda1 unchanged; restart controller at the next run boundary and resume the updated27-cell matrix.')
    for name in ('workloads/lambda-10.jsonl','workloads/lambda-50.jsonl'):
        manifest['hashes'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    write_json(ROOT/'manifest.json',manifest)
    readme=(ROOT/'README.zh.md').read_text().replace('λ={0.1,0.5,1}','λ={1,10,50}').replace('约42.18小时',f'约{lower_hours:.2f}小时')
    readme+='\n到达率已按用户修正为全局1/10/50请求每秒。正在运行的lambda1组完整保留；旧0.1/0.5时间表仅作为历史文件保留，不进入后续队列。控制器在当前组结束后切换，CPU模型与运行中的GPU服务不因参数修正中断。\n'
    (ROOT/'README.zh.md').write_text(readme)
    # The currently loaded legacy controller holds its original matrix in
    # memory. This prelaunch sentinel stops it before it starts another run.
    # A boundary watcher archives the sentinel, then resumes completed work.
    write_json(ROOT/'standard-g20-l1/status.json',{'status':'reconfiguration_boundary','reason':'User changed Poisson rates; switch controllers after current lambda1 completes.',
        'rate_update':'rate-update-1-10-50','no_inference_started':True,'epoch':time.time()})
    write_json(UPDATE/'status.json',{'status':'staging','active_run_preserved':status['cell']['id'],'rates':[1,10,50],
        'boundary_cell':'standard-g20-l1','epoch':time.time(),'last_arrival_s':times,'minimum_serial_schedule_hours':lower_hours})
    write_json(UPDATE/'checks.json',{'ok':True,'matrix_cells':27,'requests_per_cell':1319,'lambda1_byte_hash_unchanged':canonical_hash,
        'all_non_rate_settings_unchanged':True,'same_samples_and_scaled_poisson_intervals':True,'no_model_smokes':True})


if __name__=='__main__':main()
