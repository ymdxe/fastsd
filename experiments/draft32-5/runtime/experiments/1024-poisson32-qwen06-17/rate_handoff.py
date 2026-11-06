"""Switch the legacy controller at a run boundary without interrupting inference."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback
from common import ROOT,control,write_json

UPDATE=ROOT/'rate-update-1-10-50'
os.environ['CAMPAIGN_CONTROL']='http://127.0.0.1:18634'


def save(phase,**extra):
    previous=json.loads((UPDATE/'status.json').read_text())
    write_json(UPDATE/'status.json',dict(previous,status=phase,epoch=time.time(),**extra))


def keep_report_label_current():
    # The already-loaded report function has its original heading in memory.
    p=ROOT/'RESULTS.zh.md'
    if p.exists():
        text=p.read_text()
        updated=text.replace('全局泊松 λ=0.1/0.5/1 请求/秒','全局泊松 λ=1/10/50 请求/秒')
        if text!=updated:
            temporary=UPDATE/'RESULTS.updated.md';temporary.write_text(updated);temporary.replace(p)


def main():
    config=json.loads((UPDATE/'status.json').read_text())
    active=config['active_run_preserved'];boundary=config['boundary_cell']
    for node in ('node1','node3'):
        ack=json.loads((UPDATE/f'{node}-staged.json').read_text())
        assert ack['rates']==[1,10,50] and ack['inference_unchanged']
    target_pid=json.loads((ROOT/active/'target/process.json').read_text())['pid']
    parent=int(next(s for s in Path(f'/proc/{target_pid}/status').read_text().splitlines() if s.startswith('PPid:')).split()[1])
    assert str(ROOT/'campaign.py').encode() in Path(f'/proc/{parent}/cmdline').read_bytes()
    write_json(UPDATE/'watcher.json',{'pid':os.getpid(),'legacy_campaign_pid':parent,'active_run':active,'boundary_cell':boundary})
    save('waiting_for_current_run',cpu_nodes_staged=True)
    while True:
        keep_report_label_current()
        current=json.loads((ROOT/'status.json').read_text())
        active_status=json.loads((ROOT/active/'status.json').read_text())
        if active_status['status']=='failed':
            raise RuntimeError('Active inference failed independently of the rate update; inspect its preserved logs.')
        if current.get('status')=='failed' and current.get('cell',{}).get('id')==boundary:
            if 'FileExistsError' not in current.get('error',''):raise RuntimeError(current)
            break
        time.sleep(2)
    assert active_status['status']=='completed'
    assert not (ROOT/boundary/'target/process.json').exists()
    active_control=json.loads((ROOT/active/'control.json').read_text())
    assert len(active_control['results'])==1319 and len(active_control['done'])==2
    assert json.loads((ROOT/active/'target/exit.json').read_text())['exit_status']==0
    for _ in range(100):
        p=Path(f'/proc/{parent}/stat')
        if not p.exists() or p.read_text().split()[2]=='Z':break
        time.sleep(.1)
    else:raise RuntimeError('Legacy controller has not exited')
    save('switching_controllers')
    archive=UPDATE/'handoff';archive.mkdir(exist_ok=False)
    (ROOT/'status.json').rename(archive/'legacy-controller-exit.json')
    (ROOT/boundary/'status.json').rename(archive/'prelaunch-guard-status.json')
    state_path=ROOT/boundary/'control.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        assert not state['ready'] and not state['results'] and not state['done']
        state_path.rename(archive/'prelaunch-guard-control.json')
        write_json(state_path,dict(state,abort=False))
    write_json(ROOT/'status.json',{'status':'reconfiguring','rates_global_rps':[1,10,50],'completed_preserved':active,'epoch':time.time()})
    old=json.loads((ROOT/'control-launch.json').read_text());pid=old['pid']
    assert str(ROOT/'coordinator.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    write_json(archive/'previous-controller.json',old)
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        p=Path(f'/proc/{pid}/stat')
        if not p.exists() or p.read_text().split()[2]=='Z':break
        time.sleep(.1)
    subprocess.run([sys.executable,str(ROOT/'start_control.py')],cwd=ROOT,check=True)
    for _ in range(100):
        try:
            if control('health').get('ok'):break
        except Exception:pass
        time.sleep(.1)
    else:raise RuntimeError('Updated controller did not become healthy')
    cmd=shlex.quote(sys.executable)+' -u '+shlex.quote(str(ROOT/'campaign.py'))+' --resume'
    cmd+=' > >(tee -a '+shlex.quote(str(ROOT/'campaign.log'))+') 2>&1'
    subprocess.run(['tmux','new-window','-d','-t','exp','-n','campaign','-c',str(ROOT),cmd],check=True)
    save('applied',completed_run_preserved=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save('handoff_failed',error=repr(exc),traceback=traceback.format_exc())
        raise
