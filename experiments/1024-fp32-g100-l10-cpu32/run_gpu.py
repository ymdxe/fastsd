"""GPU-window supervisor: run all three methods, then finalize and release control."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from common import ROOT,write_json


class Tee:
    def __init__(self,terminal,log):self.terminal=terminal;self.log=log
    def write(self,value):self.terminal.write(value);self.log.write(value);return len(value)
    def flush(self):self.terminal.flush();self.log.flush()
    def isatty(self):return self.terminal.isatty()


def main():
    write_json(ROOT/'gpu-supervisor.json',{'pid':os.getpid(),'epoch':time.time(),'automatic_finalization':True})
    with (ROOT/'campaign.log').open('a',buffering=1) as log:
        sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
        try:
            import campaign
            campaign.main()
            import finalize
            finalize.main()
            pid=json.loads((ROOT/'control-launch.json').read_text())['pid'];cmd=Path(f'/proc/{pid}/cmdline')
            if cmd.exists() and str(ROOT/'coordinator.py').encode() in cmd.read_bytes():os.kill(pid,signal.SIGTERM)
        except BaseException as exc:
            state=ROOT/'status.json'
            if state.exists() and json.loads(state.read_text()).get('status')=='completed':
                write_json(ROOT/'finalization.json',{'status':'failed','error':repr(exc),'epoch':time.time()})
            traceback.print_exc();raise
        finally:
            sys.stdout=sys.__stdout__;sys.stderr=sys.__stderr__


if __name__=='__main__':main()
