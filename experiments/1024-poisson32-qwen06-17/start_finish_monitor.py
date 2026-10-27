import subprocess
import sys
from common import ROOT,write_json

assert not (ROOT/'finish-monitor-launch.json').exists(),'Monitor already started'
with (ROOT/'finish-monitor.log').open('a') as log:
    p=subprocess.Popen([sys.executable,'-u',str(ROOT/'finish_monitor.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write_json(ROOT/'finish-monitor-launch.json',{'pid':p.pid,'python':sys.executable})
