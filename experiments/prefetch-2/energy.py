"""Read-only 1 Hz device/package energy sampling over the common measurement window."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from common import write_json

def integrate(points, start, end, counter_range=None):
    total = 0.0
    covered = 0.0
    for (a, x), (b, y) in zip(points, points[1:]):
        lo, hi = max(a, start), min(b, end)
        if hi <= lo or b <= a or b-a > 5:
            continue
        fraction = (hi-lo)/(b-a)
        if counter_range is not None:
            delta = y-x
            if delta < 0:
                delta += counter_range
            total += delta*fraction/1e6
        else:
            left = x+(y-x)*(lo-a)/(b-a)
            right = x+(y-x)*(hi-a)/(b-a)
            total += (left+right)*.5*(hi-lo)
        covered += hi-lo
    return {'joules':total if covered else None, 'covered_s':covered,
            'window_s':end-start, 'complete_coverage':covered >= end-start-2}

class EnergySampler:
    def __init__(self, output, config, role, server_offset_s=0.0):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.config, self.role, self.offset = config, role, server_offset_s
        self.stop_event = threading.Event()
        self.packages = {}
        self.unavailable = {}
        if role == 'cpu':
            for path in Path('/sys/class/powercap').glob('intel-rapl:*'):
                if path.name.count(':') != 1:
                    continue
                try:
                    int((path/'energy_uj').read_text())
                    self.packages[path.name] = {'path':str(path),
                        'name':(path/'name').read_text().strip(),
                        'max_energy_range_uj':int((path/'max_energy_range_uj').read_text())}
                except (OSError, ValueError) as exc:
                    self.unavailable[path.name] = str(exc)
        write_json(self.output/'energy-capabilities.json', {'role':role, 'packages':self.packages,
            'unavailable':self.unavailable, 'gpu_uuid':config['gpu_uuid'] if role=='gpu' else None,
            'system_energy_available':False, 'server_clock_offset_s':self.offset,
            'scope':'GPU device or top-level CPU packages; no system total; no idle subtraction'})
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def run(self):
        last_resource = 0.0
        with (self.output/'power.jsonl').open('x',encoding='utf-8') as handle:
            while not self.stop_event.is_set():
                started = time.monotonic()
                row = {'server_epoch_s':time.time()+self.offset, 'local_epoch_s':time.time()}
                try:
                    if self.role == 'gpu':
                        result = subprocess.run(['nvidia-smi','--id='+self.config['gpu_uuid'],
                            '--query-gpu=uuid,power.draw,memory.used,utilization.gpu',
                            '--format=csv,noheader,nounits'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            universal_newlines=True,timeout=3,check=True)
                        values = [v.strip() for v in result.stdout.strip().split(',')]
                        if values[0] != self.config['gpu_uuid']:
                            raise RuntimeError('GPU identity changed')
                        row.update(gpu_power_w=float(values[1]),gpu_memory_mib=float(values[2]),gpu_utilization=float(values[3]))
                    else:
                        row['package_energy_uj'] = {}
                        for key, package in self.packages.items():
                            row['package_energy_uj'][key] = int((Path(package['path'])/'energy_uj').read_text())
                except Exception as exc:
                    row['sampling_error'] = repr(exc)
                if started-last_resource >= 5:
                    row['loadavg'] = list(os.getloadavg())
                    row['memory'] = {line.split(':',1)[0]:line.split(':',1)[1].strip()
                        for line in Path('/proc/meminfo').read_text().splitlines()
                        if line.startswith(('MemAvailable:','MemFree:','MemTotal:'))}
                    last_resource = started
                handle.write(json.dumps(row)+'\n')
                handle.flush()
                self.stop_event.wait(max(.01,1-(time.monotonic()-started)))

    def finish(self, start=None, end=None):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if start is None or end is None:
            write_json(self.output/'energy-summary.json',{'valid':False,'reason':'common window unavailable'})
            return
        rows = [json.loads(line) for line in (self.output/'power.jsonl').read_text(encoding='utf-8').splitlines()]
        streams = {}
        if self.role == 'gpu':
            streams['gpu_device'] = ([(r['server_epoch_s'],r['gpu_power_w']) for r in rows if 'gpu_power_w' in r],None)
        for key, package in self.packages.items():
            streams[key] = ([(r['server_epoch_s'],r['package_energy_uj'][key]) for r in rows if key in r.get('package_energy_uj',{})],package['max_energy_range_uj'])
        write_json(self.output/'energy-summary.json', {'t0_server_epoch_s':start,'end_server_epoch_s':end,
            'measurement':{key:integrate(points,start,end,limit) for key,(points,limit) in streams.items()},
            'idle_baseline':{key:integrate(points,start-60,start,limit) for key,(points,limit) in streams.items()},
            'system_energy_available':False,'missing_packages':self.unavailable,
            'sampling_errors':sum('sampling_error' in r for r in rows)})
