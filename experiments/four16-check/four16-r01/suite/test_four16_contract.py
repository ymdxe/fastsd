"""Integration boundaries that changed when reducing the client population."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


class FourMethodContract(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'config.json').read_text())

    def test_four_actual_methods_keep_their_semantics(self):
        from suite import method_flags
        self.assertEqual(['fastsd','specedge','sd','target_only'], self.config['methods'])
        with patch.dict(os.environ, {'FASTSD_DIAGNOSTIC_POLICY':'legacy'}):
            fast, sd = method_flags('fastsd'), method_flags('sd')
        for flag in ('--enable_pipeline','--enable_proactive_draft','--enable_latency_priority','--kv_prefetch'):
            self.assertIn(flag, fast)
            self.assertIn('--no-'+flag[2:], sd)
        self.assertEqual('legacy', fast[fast.index('--scheduler_policy')+1])
        self.assertEqual('fcfs', sd[sd.index('--scheduler_policy')+1])
        self.assertEqual('gpu_resident',self.config['target_only_kv_policy'])

    def test_repaired_runtime_has_exact_frozen_source(self):
        doc=json.loads((ROOT/'source-exact-manifest.json').read_text())
        self.assertEqual(self.config['source_sha'],doc['source_sha'])
        self.assertGreaterEqual(len(doc['files']),90)
        for name,sha in doc['files'].items():
            self.assertEqual(sha,hashlib.sha256((ROOT/'runtime'/name).read_bytes()).hexdigest(),name)

    def test_specedge_server_and_all_clients_map_16_mixed_gammas(self):
        from specedge_server import configure_environment
        from specedge_worker import configure_environment as client_environment
        cfg=self.config.copy()
        expected=[f'{node}-w{w}' for node,n in cfg['nodes'].items() for w in range(n['workers'])]
        self.assertEqual(16,len(expected))
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            cfg.update(output_dir=directory, expected_workers=expected, model=cfg['target_model'],run_id='contract')
            with patch.dict(os.environ,{},clear=False):
                configure_environment(cfg)
                self.assertEqual('16',os.environ['SPECEDGE_NUM_CLIENTS'])
                self.assertEqual([4]*8+[8]*8,json.loads(os.environ['FASTSD_SPECEDGE_CLIENT_GAMMAS']))
                for identity in expected:
                    node=identity.split('-w')[0]
                    args=SimpleNamespace(gamma=cfg['nodes'][node]['gamma'],exp_name=directory)
                    os.environ.update(FASTSD_SUITE_WORKER=identity,FASTSD_SUITE_RUN_ID='contract')
                    index=client_environment(args,cfg,[{'sample_id':'contract'}])
                    self.assertEqual(expected.index(identity),index)
                    self.assertEqual('1',os.environ['SPECEDGE_MAX_N_BEAMS'])
                    self.assertEqual('1',os.environ['SPECEDGE_PROACTIVE_MAX_N_BEAMS'])
                    self.assertEqual('1',os.environ['SPECEDGE_MAX_BRANCH_WIDTH'])
                    self.assertEqual(str(cfg['nodes'][node]['gamma']),os.environ['SPECEDGE_MAX_BUDGET'])

    def test_16_arrival_streams_are_independently_seeded(self):
        rows=[json.loads(line) for line in (ROOT/'traces/humaneval.jsonl').read_text().splitlines()]
        streams={(r['node'],r['worker']):r['arrival_seed'] for r in rows}
        self.assertEqual(16,len(streams))
        self.assertEqual(16,len(set(streams.values())))
        affinity=json.loads((ROOT/'affinity.json').read_text())
        for node,cfg in self.config['nodes'].items():
            assigned=affinity[node]['workers']
            self.assertEqual(cfg['workers'],len(assigned))
            cpus=[cpu for a in assigned for cpu in a['logical_cpus']]
            self.assertEqual(8*cfg['workers'],len(set(cpus)))
            self.assertTrue(all(a['torch_num_threads']==8 and a['torch_num_interop_threads']==1 for a in assigned))


if __name__=='__main__':
    unittest.main()
