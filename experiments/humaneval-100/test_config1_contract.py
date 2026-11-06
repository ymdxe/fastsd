"""No-model checks for workload identity, coordinator completion, and energy windows."""
import json
import random
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from common import Coordinator, summarize_dir, trace_hash, write_json, validate_trace
from clients import build_worker_command
from energy import integrate

ROOT=Path(__file__).resolve().parent

class ConfigContract(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'config.json').read_text())

    def test_all_conditions_and_traces(self):
        c=self.config
        self.assertEqual(len(c['datasets'])*len(c['methods']),len(c['stages']))
        self.assertEqual({(d,m) for d in c['datasets'] for m in c['methods']},
                         {(s['dataset'],s['method']) for s in c['stages']})
        self.assertEqual('sdpa_per_sequence',c['target_attention_backend'])
        self.assertEqual('gpu_resident',c['target_only_kv_policy'])
        self.assertEqual(100,c['tasks_per_worker'])
        self.assertEqual(1/30,c['rate_per_worker'])
        self.assertTrue(all(s['requests']==3200 and '-p30-n100-' in s['condition_id'] for s in c['stages']))
        schedules=[]
        for dataset in c['datasets']:
            rows=[json.loads(line) for line in (ROOT/'traces'/(dataset+'.jsonl')).read_text(encoding='utf-8').splitlines()]
            self.assertEqual(3200,validate_trace(rows,c))
            self.assertEqual(3200,len(rows))
            self.assertEqual(3200,len({r['sample_id'] for r in rows}))
            self.assertTrue(all(len(r['input_token_ids'])+400<2048 for r in rows))
            for node,ncfg in c['nodes'].items():
                keys=None
                for worker in range(ncfg['workers']):
                    selected=[r for r in rows if r['node']==node and r['worker']==worker]
                    self.assertEqual(100,len(selected))
                    rng=random.Random(selected[0]['arrival_seed'])
                    elapsed=0.0
                    for row in selected:
                        elapsed+=rng.expovariate(1/30)
                        self.assertEqual(elapsed,row['scheduled_arrival_s'])
                    inputs={r['input_key'] for r in selected}
                    self.assertEqual(100,len(inputs))
                    self.assertEqual(set(json.loads((ROOT/'trace-provenance.json').read_text())['source_ids']),{r['source_id'] for r in selected})
                    if keys is not None:self.assertEqual(keys,inputs)
                    keys=inputs
                    self.assertTrue(all(a['scheduled_arrival_s']<b['scheduled_arrival_s'] for a,b in zip(selected,selected[1:])))
            schedules.append([(r['node'],r['worker'],r['request_index'],r['scheduled_arrival_s']) for r in rows])
        self.assertTrue(all(schedule==schedules[0] for schedule in schedules))

    def test_coordinator_all32_and_global_window(self):
        rows=[json.loads(line) for line in (ROOT/'traces/humaneval.jsonl').read_text(encoding='utf-8').splitlines()]
        cfg={**self.config,'trace_rows':rows,'run_id':'test','method':'target_only','trace':'humaneval',
            'bundle_id':'test','trace_sha256':trace_hash(rows),'arrival_window_s':max(r['scheduled_arrival_s'] for r in rows),
            'expected_workers':[f'{n}-w{w}' for n,c in self.config['nodes'].items() for w in range(c['workers'])]}
        test_root = ROOT.parent/'evidence'
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as tmp:
            output=Path(tmp)
            assert output.resolve().parent == test_root.resolve()
            (output/'reports').mkdir()
            write_json(output/'config.json',cfg)
            coordinator=Coordinator(cfg,output)
            for worker in cfg['expected_workers']:
                coordinator.register({'run_id':'test','worker':worker,'bundle_id':'test','trace_sha256':cfg['trace_sha256'],'clock':{}})
            self.assertGreaterEqual(coordinator.t0-time.time(),64)
            for worker in cfg['expected_workers']:
                assigned=[r for r in rows if f"{r['node']}-w{r['worker']}"==worker]
                records=[{'task_id':r['sample_id'],'generated_tokens':1,'completion_s':r['scheduled_arrival_s']+1,
                    'arrival_lag_ms':0,'request_e2e_ms':1,'ttft_ms':1,'tpot_ms':None} for r in assigned]
                coordinator.report({'run_id':'test','worker':worker,'records':records})
            summary=summarize_dir(output)
            self.assertTrue(summary['measurement_valid'])
            self.assertEqual([800,800,1600],[summary['groups'][n]['requests'] for n in ('node1','node2','node3')])
            self.assertAlmostEqual(coordinator.measurement_end-coordinator.t0,summary['drain_time_s'],places=4)

    def test_ar_entrypoint_and_threads(self):
        cfg={**self.config,'method':'target_only','run_id':'test','bundle_id':'test','trace_sha256':'test'}
        cmd,env=build_worker_command(cfg,'node2',0,Path('/inputs'),Path('/out'),list(range(8)),'http://localhost:1')
        self.assertTrue(any(str(x).endswith('ar_worker.py') for x in cmd))
        self.assertEqual('1',env['TORCH_NUM_INTEROP_THREADS'])
        self.assertEqual('8',env['OMP_NUM_THREADS'])

    def test_energy_boundaries_and_wrap(self):
        self.assertAlmostEqual(integrate([(0,100),(1,100),(2,100)],.25,1.75)['joules'],150)
        self.assertAlmostEqual(integrate([(0,900000),(1,100000)],0,1,1000000)['joules'],.2)
        self.assertIsNone(integrate([(0,10),(10,10)],0,10)['joules'])

    def test_missing_fixed_gpu_is_wait_not_substitution(self):
        from campaign import query_gpu_memory
        uuid=self.config['gpu_uuid']
        other=SimpleNamespace(returncode=0,stdout='GPU-other, 49140, 48000, 1140\n',stderr='')
        with patch('campaign.subprocess.run',return_value=other):
            state=query_gpu_memory(uuid)
        self.assertFalse(state['device_available'])
        self.assertEqual(uuid,state['gpu_uuid'])
        self.assertEqual(0,state['free_mib'])

    def test_missing_wire_index_rejected_before_launch(self):
        rows=[json.loads(line) for line in (ROOT/'traces/humaneval.jsonl').read_text(encoding='utf-8').splitlines()]
        del rows[0]['global_index']
        with self.assertRaisesRegex(ValueError,'global_index'):
            validate_trace(rows,self.config)

if __name__=='__main__':unittest.main()
