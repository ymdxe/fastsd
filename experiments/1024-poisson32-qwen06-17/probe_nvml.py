import json
from pathlib import Path
import pynvml as n
from common import ROOT,write_json
n.nvmlInit();h=n.nvmlDeviceGetHandleByUUID(b'GPU-422501a2-5189-7f5b-5ce6-95464276b27f')
m=n.nvmlDeviceGetMemoryInfo(h)
write_json(ROOT/'nvml-probe.json',{'uuid':n.nvmlDeviceGetUUID(h).decode(),'memory':{'total':m.total,'used':m.used,'free':m.free},
    'compute_processes':[{'pid':p.pid,'used':p.usedGpuMemory} for p in n.nvmlDeviceGetComputeRunningProcesses(h)],
    'graphics_processes':[{'pid':p.pid,'used':p.usedGpuMemory} for p in n.nvmlDeviceGetGraphicsRunningProcesses(h)],'power_w':n.nvmlDeviceGetPowerUsage(h)/1000})
n.nvmlShutdown()
