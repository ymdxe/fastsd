"""Select gamma200/500 at lambda50; retain the user-interrupted partial run."""
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import ROOT,write_json
from report import load_rows

DIRECTORY=ROOT/'subset-g200-g500-l50'
BACKUP=DIRECTORY/'backups'


def main():
    transition=json.loads((DIRECTORY/'transition.json').read_text());previous=transition['previous_run']
    control=json.loads((ROOT/previous/'control.json').read_text())
    assert len(control['done'])==2 and all(d['finished'] for d in control['done'].values())
    assert json.loads((ROOT/previous/'target/exit.json').read_text())['exit_status']==0
    legacy=json.loads((ROOT/'rate-update-1-10-50/watcher.json').read_text())['legacy_campaign_pid']
    proc=Path(f'/proc/{legacy}/cmdline')
    assert not proc.exists() or str(ROOT/'campaign.py').encode() not in proc.read_bytes()
    old=json.loads((BACKUP/'matrix.json').read_text())
    chosen={(r['gamma'],r['method']):r for r in old if r['gamma'] in (200,500) and r['rate']==50}
    matrix=[chosen[(g,m)] for g in (200,500) for m in ('fastsd','standard','specedge')]
    assert len(matrix)==6 and all(r['cap']==1024 and r['expected_requests']==1319 for r in matrix)
    assert {(r['gamma'],r['rate'],r['method']) for r in matrix}=={(g,50.,m) for g in (200,500) for m in ('standard','fastsd','specedge')}
    for r in matrix:assert not (ROOT/r['id']/'target/process.json').exists(),r['id']
    # Preserve the controller's raw shutdown classification before marking this
    # as an intentional user interruption instead of an inference failure.
    previous_status=json.loads((ROOT/previous/'status.json').read_text())
    write_json(BACKUP/'previous-run-status.json',previous_status)
    write_json(BACKUP/'campaign-status.json',json.loads((ROOT/'status.json').read_text()))
    rows=load_rows(ROOT/previous)
    write_json(ROOT/previous/'status.json',dict(previous_status,status='interrupted_by_user',
        reason='Only gamma200/500 and lambda50 requested; partial results retained outside current matrix.',completed_requests=len(rows)))
    write_json(DIRECTORY/'previous-run-summary.json',{'run':previous,'status':'interrupted_by_user','completed_requests':len(rows),
        'by_node':{n:sum(r['node']==n for r in rows) for n in ('node1','node3')},'target_exit_status':0,'included_in_subset':False})
    rate_status=json.loads((ROOT/'rate-update-1-10-50/status.json').read_text())
    write_json(ROOT/'rate-update-1-10-50/status.json',dict(rate_status,status='cancelled_by_user_subset',epoch=time.time()))
    guard=ROOT/'standard-g20-l1/status.json'
    if guard.exists():
        write_json(BACKUP/'old-boundary-guard.json',json.loads(guard.read_text()))
        write_json(guard,{'status':'cancelled_by_user_subset','no_inference_started':True})
    write_json(ROOT/'matrix.json',matrix)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    protected=[name for name in manifest['hashes'] if name.startswith(('source/','specedge/')) or name in ('common.py','node_agent.py','native_worker.py','target.py','telemetry.py','workloads/lambda-50.jsonl')]
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==manifest['hashes'][name] for name in protected)
    manifest.update(status='prepared',gamma=[200,500],rates_global_rps=[50.],planned_runs=6,planned_requests=7914,
        active_run=None,scope_revision=4,scope_updated_epoch=time.time(),excluded_partial_run=previous)
    write_json(ROOT/'manifest.json',manifest)
    validation=json.loads((ROOT/'validation.json').read_text())
    samples=[json.loads(s) for s in (ROOT/'workloads/lambda-50.jsonl').read_text().splitlines() if s.strip()]
    assert len(samples)==1319 and {s['source_index'] for s in samples}==set(range(1319))
    validation.update(matrix_cells=6,scheduled_last_arrival_s={'50.0':samples[-1]['scheduled_arrival_s']},
        minimum_serial_schedule_hours=samples[-1]['scheduled_arrival_s']*6/3600,scope_revision=4)
    write_json(ROOT/'validation.json',validation)
    readme=(ROOT/'README.zh.md').read_text()
    readme=readme.replace('γ={20,200,500} × 全局 λ={1,10,50}','γ={200,500} × 全局 λ={50}')
    readme=readme.replace('共27组、35613个正式请求','共6组、7914个正式请求')
    readme=readme.replace('27组全部结束后','6组全部结束后')
    readme=readme.replace('仅计划到达时间的串行下界约3.63小时，实际还需要草拟和排空队列。','本轮每组的最后一个计划到达约为26秒；完整运行仍需要CPU草拟和队列排空。')
    marker='到达率已按用户修正为全局1/10/50请求每秒。'
    if marker in readme:readme=readme[:readme.index(marker)]
    readme+='当前用户仅要求 γ=200/500、λ=50 的六组实验。旧 γ20/λ1 组已按指令停止，部分结果完整保留，不计入当前六组统计。旧27组配置保存在 subset-g200-g500-l50/backups，原组间切换监视器已取消。\n'
    (ROOT/'README.zh.md').write_text(readme)
    for path in ROOT.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    write_json(DIRECTORY/'checks.json',{'ok':True,'matrix_cells':6,'expected_requests':7914,'gamma':[200,500],'lambda':[50],
        'inference_files_unchanged':len(protected),'old_target_exit_status':0,'no_model_smokes':True})
    # All draft workers have exited; the two idle agents tolerate a short
    # controller reconnect and will receive only the selected jobs afterwards.
    controller=json.loads((ROOT/'control-launch.json').read_text());pid=controller['pid']
    assert str(ROOT/'coordinator.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    write_json(BACKUP/'control-launch.json',controller)
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        p=Path(f'/proc/{pid}/stat')
        if not p.exists() or p.read_text().split()[2]=='Z':break
        time.sleep(.1)
    subprocess.run([sys.executable,str(ROOT/'start_control.py')],cwd=ROOT,check=True)
    launch=ROOT/'finish-monitor-launch.json'
    if launch.exists():launch.rename(BACKUP/'retired-finish-monitor-launch.json')
    write_json(ROOT/'status.json',{'status':'preparing_subset','gamma':[200,500],'rates_global_rps':[50],
        'planned_runs':6,'planned_requests':7914,'epoch':time.time()})
    write_json(DIRECTORY/'transition.json',dict(transition,status='ready_to_sync',epoch=time.time()))


if __name__=='__main__':main()
