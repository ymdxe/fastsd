"""Checks observations do not change planner decisions, and diagnostic workload is fixed."""
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import unittest

ROOT=Path(__file__).resolve().parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 return module

class DiagnosticsContract(unittest.TestCase):
 def test_planner_decisions_unchanged_with_observation(self):
  # The original, content-hashed planner is an explicit control for this overlay.
  old=load("baseline_diagnostic_scheduler",ROOT/"runtime/src/fastsd_scheduler.py")
  new=load("observed_diagnostic_scheduler",ROOT/"runtime/src/fastsd_scheduler.py")
  events=[]
  class Observer:
   enabled=True
   context=threading.local()
   def emit(self,kind,**fields):events.append((kind,fields))
  observer=Observer();observer.context.plan_kind="admission";observer.context.plan_id=1
  previous=new.DIAG;new.DIAG=observer
  try:
   rng=random.Random(20261101)
   for case in range(80):
    definitions=[]
    for j in range(rng.randint(1,32)):
     kind=rng.choice(["verify","prefill"]);tokens=rng.choice([4,8,16,31,128,250])
     req={"proc_id":str(j),"task_id":str(j),"task_type":kind,"gamma":tokens if kind=="verify" else 4,
          "prompt_len":tokens,"prefix_len":rng.choice([64,256,640]),"server_enqueue_monotonic":90+rng.random()*9,
          "current_time":0,"lag":0.2,"timing_ready":True,"timing_required":False}
     definitions.append((req,rng.choice(["short","mid","long"])))
    def state(module):
     q={t:{c:[] for c in ("short","mid","long")} for t in ("verify","prefill")}
     for j,(r,c) in enumerate(definitions):
      item=module.WorkItem.from_request(dict(r),category=c,cycle=0,work_id=str(j))
      q[r["task_type"]][c].append(item)
     return q
    for policy in ("legacy",):
     kwargs=dict(token_budget=rng.choice([32,64,128,512]),max_num_seqs=16,
                 now=100,selection_policy=policy,candidate_value=lambda item,sl,selected:1.0/(1+int(item.proc_id)))
     a=old.plan_iteration(state(old),**kwargs);b=new.plan_iteration(state(new),**kwargs)
     self.assertEqual(a.selected_work_ids,b.selected_work_ids)
     self.assertEqual([dataclasses.asdict(x) for x in a.verify_slices+a.prefill_slices],
                      [dataclasses.asdict(x) for x in b.verify_slices+b.prefill_slices])
     self.assertEqual((a.used_tokens,a.next_wrr_cursor,a.next_cycle),(b.used_tokens,b.next_wrr_cursor,b.next_cycle))
   self.assertTrue(events)
  finally:new.DIAG=previous

 def test_buffered_trace_close_and_no_drops(self):
  from diagnostic_trace import DiagnosticTrace
  import unittest.mock
  with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
   with unittest.mock.patch.dict(os.environ,{"FASTSD_DIAGNOSTIC_DIR":tmp}):
    trace=DiagnosticTrace()
    for i in range(100):trace.emit("test",index=i)
    trace.close()
    rows=[json.loads(x) for x in trace.path.read_text().splitlines()]
    self.assertEqual(list(range(100)),[r["index"] for r in rows if r["kind"]=="test"])
    summary=json.loads(trace.path.with_name("trace-summary-"+str(os.getpid())+".json").read_text())
    self.assertEqual(summary["dropped"],0);self.assertEqual(summary["writer_errors"],0)
    self.assertFalse(summary["writer_alive_at_close"])

 def test_backlog_counts_only_arrived_waiting_tasks(self):
  from diagnostic_trace import backlog
  self.assertEqual(0,backlog([1,2,10],1.5,0))
  self.assertEqual(1,backlog([1,2,10],3,0))
  self.assertEqual(2,backlog([1,2,10],11,0))

 def test_only_paper_priority_wrr_condition(self):
  c=json.loads((ROOT/"config.json").read_text())
  self.assertTrue(c["diagnostic_only"])
  self.assertEqual([("fastsd","legacy"),("specedge","single_chain"),("sd","fcfs"),("target_only","ar")],
                   [(s["method"],s["scheduler_policy"]) for s in c["stages"]])
  self.assertEqual(480,c["tasks_per_worker"]*sum(n["workers"] for n in c["nodes"].values()))
  self.assertEqual("sdpa_per_sequence",c["target_attention_backend"])

if __name__=="__main__":unittest.main()

