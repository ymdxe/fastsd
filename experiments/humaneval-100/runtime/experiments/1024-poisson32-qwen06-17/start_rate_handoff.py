import subprocess
import sys
from common import ROOT,write_json

directory=ROOT/'rate-update-1-10-50'
assert not (directory/'launch.json').exists(),'Handoff watcher already started'
with (directory/'watcher.log').open('a') as log:
    p=subprocess.Popen([sys.executable,'-u',str(ROOT/'rate_handoff.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write_json(directory/'launch.json',{'pid':p.pid,'python':sys.executable})
