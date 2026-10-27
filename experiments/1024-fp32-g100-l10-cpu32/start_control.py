"""Start only this campaign's private loopback controller, with an audit PID."""
import os
from pathlib import Path
import socket
import subprocess
import sys
from common import ROOT,write_json

with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock.bind(('127.0.0.1',18634))
with (ROOT/'coordinator.log').open('a') as log:
    process=subprocess.Popen([sys.executable,'-u',str(ROOT/'coordinator.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write_json(ROOT/'control-launch.json',{'pid':process.pid,'python':sys.executable})
