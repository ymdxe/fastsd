"""Validate observation clocks and real copy contracts, not formal output equivalence."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import unittest

root=Path(sys.argv[1]).resolve()
output=root.parent
sys.path[:0]=[str(root),str(root/'runtime'),str(root/'runtime/tests')]
status={'state':'running','started_epoch_s':time.time(),'scope':'diagnostic_observation_only','formal_output_equivalence':False}
def save():
 (output/'gpu-status.json').write_text(json.dumps(status,indent=2))
save()
try:
 from promotion import content_digest
 from suite import verify_bundle
 config=json.loads((root/'config.json').read_text())
 digest=content_digest(root)
 cpu=json.loads((output/'cpu-status.json').read_text())
 if cpu.get('state')!='passed' or cpu.get('validated_content_sha256')!=digest:
  raise RuntimeError('CPU evidence absent, failed, or bound to other content')
 bundle=verify_bundle()
 from campaign import wait_gpu_memory
 wait_gpu_memory(config,output,'fastsd')
 os.environ['FASTSD_DIAGNOSTIC_DIR']=str(output/'gpu-observer-probe')
 import torch
 torch.set_num_threads(1)
 torch.cuda.set_device(0)
 from diagnostic_trace import trace
 trace.gpu_init(torch,'cuda:0')
 # Existing production transfer tests cover ownership, partial chunks, draining,
 # admission protection, and the memory/CPU preparation fallback.
 tests=unittest.defaultTestLoader.loadTestsFromName('test_round2_cache.TransferIsolationTests')
 result=unittest.TextTestRunner(verbosity=2).run(tests)
 if not result.wasSuccessful() or result.skipped:
  raise RuntimeError('Actual CUDA transfer checks did not all run and pass')
 # Timestamp calibration uses a synthetic busy kernel, not model throughput.
 # It cannot establish whether production prefetch overlaps production forward.
 pinned=torch.arange(4*1024*1024,dtype=torch.float32).pin_memory()
 destination=torch.empty_like(pinned,device='cuda')
 compute=torch.cuda.Stream();copy=torch.cuda.Stream()
 def event_pair(): return torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 def calibrate(name,concurrent):
  a,b=event_pair();c,d=event_pair()
  with torch.cuda.stream(compute):
   a.record();torch.cuda._sleep(50_000_000);b.record()
  with torch.cuda.stream(copy):
   copy.wait_event(a if concurrent else b)
   c.record();destination.copy_(pinned,non_blocking=True);d.record()
  b.synchronize();d.synchronize()
  torch.testing.assert_close(destination.cpu(),pinned,rtol=0,atol=0)
  trace.gpu_span('probe_compute',a,b,calibration=name)
  trace.gpu_span('probe_copy',c,d,calibration=name)
  aa,bb,cc,dd=[trace.origin.elapsed_time(x) for x in (a,b,c,d)]
  return {'compute_ms':[aa,bb],'copy_ms':[cc,dd],'intersection_ms':max(0,min(bb,dd)-max(aa,cc))}
 negative=calibrate('sequential_negative',False)
 positive=calibrate('concurrent_positive',True)
 if negative['intersection_ms']>0.05 or positive['intersection_ms']<=0:
  raise RuntimeError('CUDA interval overlap calibration failed')
 torch.cuda.synchronize()
 trace.gpu_collect();trace.close()
 observer=json.loads(trace.path.with_name('trace-summary-'+str(os.getpid())+'.json').read_text())
 if any(observer[k] for k in ('dropped','writer_errors','unresolved_cuda_spans','cuda_limit_drops','writer_alive_at_close')):
  raise RuntimeError('Observer incomplete: '+repr(observer))
 rows=[json.loads(x) for x in trace.path.read_text().splitlines()]
 spans=[r for r in rows if r['kind']=='cuda_span']
 if not any(r['operation']=='kv_copy' and r['bytes']>0 for r in spans):
  raise RuntimeError('No real production-transfer CUDA span recorded')
 if any(r['end_ms']<r['start_ms'] for r in spans):raise RuntimeError('Invalid CUDA intervals')
 gates={'cpu_contracts':True,'trace_contracts':True,'observer_contracts':True,'gpu_event_transfer_probe':True}
 summary={'bundle_id':bundle,'validated_content_sha256':digest,'tests_run':result.testsRun,
  'cuda_spans':len(spans),'negative_control':negative,'positive_control':positive,'observer':observer,
  'probe_scope':'Synthetic clock and actual transfer validation; does not assert production overlap or output equality'}
 (output/'gpu-summary.json').write_text(json.dumps(summary,indent=2))
 certificate={'scope':'diagnostic_observation_only','formal_output_equivalence':False,'passed':True,
  'gates':gates,'validated_content_sha256':digest,'parent_bundle_id':config['parent_bundle_id'],
  'cpu_evidence_sha256':hashlib.sha256((output/'cpu-status.json').read_bytes()).hexdigest(),
  'gpu_evidence_sha256':hashlib.sha256((output/'gpu-summary.json').read_bytes()).hexdigest(),
  'created_epoch_s':time.time()}
 (output/'diagnostic-certificate.json').write_text(json.dumps(certificate,indent=2))
 status.update(state='passed',gates=gates,diagnostic_launch_allowed=True,formal_launch_allowed=False,
  tests_run=result.testsRun,validated_content_sha256=digest,finished_epoch_s=time.time())
 save()
except BaseException as exc:
 status.update(state='failed',error=repr(exc),traceback=traceback.format_exc(),finished_epoch_s=time.time())
 save();raise
