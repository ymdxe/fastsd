"""Critical fault checks for measurement errors after a stateful Verify."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'source')]
from edge.edge import EdgeClient as BaseClient
from clock_runtime import ClockRuntimeMixin


class Client(ClockRuntimeMixin,BaseClient):
    def __init__(self,offsets=()):
        super().__init__('http://unused');self.offsets=list(offsets);self.refreshes=0;self.events=[]
    def _synchronize_runtime_clock(self,samples=5):
        self.refreshes+=1
        if self.offsets:self.clock_offset_ns=self.offsets.pop(0)
        self.clock_uncertainty_ns=1000
        return {'clock_offset_ns':self.clock_offset_ns,'clock_uncertainty_ns':1000}
    def _clock_publish(self,event=None):self.events.append(event)


class Session:
    def __init__(self,statuses):self.statuses=list(statuses);self.payloads=[]
    def post(self,url,**kwargs):
        self.payloads.append(json.loads(kwargs['data']));status=self.statuses.pop(0)
        response=requests.Response();response.status_code=status;response.url=url
        response._content=json.dumps({'detail':'synchronized clocks produced a negative upload duration; retry clock calibration'} if status==422 else {'accepted':4,'final_token':7}).encode()
        return response


class ClockChecks(unittest.TestCase):
    def test_valid_pull_unchanged(self):
        c=Client();r=c._attach_pull({'cloud_send_time_ns':1000,'accepted':4},1500)
        self.assertEqual(r['pull_s'],500/1e9);self.assertTrue(r['pull_timing_valid']);self.assertEqual(c.refreshes,0)
    def test_negative_pull_recalibrates_original_response(self):
        c=Client([500]);c.session=Session([])
        r=c._attach_pull({'cloud_send_time_ns':1200,'accepted':4,'final_token':7},1000)
        self.assertEqual(r['pull_s'],300/1e9);self.assertEqual(r['accepted'],4);self.assertEqual(c.session.payloads,[])
    def test_uncertain_pull_keeps_scheduler_estimate_explicit(self):
        c=Client([0]);c._last_valid_pull_s=.001
        r=c._attach_pull({'cloud_send_time_ns':1200,'accepted':4},1000)
        self.assertEqual(r['pull_s'],.001);self.assertFalse(r['pull_timing_valid']);self.assertIsNone(r['measured_pull_ms']);self.assertEqual(r['pull_raw_ns'],-200)
    def test_no_previous_estimate_stays_unavailable(self):
        c=Client([0]);r=c._attach_pull({'cloud_send_time_ns':1200,'accepted':4},1000)
        self.assertNotIn('pull_s',r);self.assertFalse(r['pull_timing_valid']);self.assertEqual(r['pull_s_source'],'unavailable')
    def test_only_rejected_upload_can_be_retried(self):
        c=Client([100]);c.session=Session([422,200]);payload={'edge_send_time_ns':1,'clock_offset_ns':0,'draft_output':[1,2]}
        result=c._post('/verify',payload)
        self.assertEqual(result['accepted'],4);self.assertEqual(len(c.session.payloads),2)
        self.assertEqual(c.session.payloads[1]['draft_output'],[1,2]);self.assertGreater(c.session.payloads[1]['edge_send_time_ns'],1)
    def test_other_http_errors_are_not_replayed(self):
        c=Client();c.session=Session([500])
        with self.assertRaises(requests.HTTPError):c._post('/verify',{})
        self.assertEqual(len(c.session.payloads),1);self.assertEqual(c.refreshes,0)
    def test_runtime_calibration_bypasses_startup_barrier(self):
        import native_worker
        c=native_worker.Client('http://unused')
        with patch.object(native_worker._BaseEdgeClient,'synchronize_clock',return_value={'ok':True}) as method:
            self.assertEqual(c._synchronize_runtime_clock(5),{'ok':True})
            method.assert_called_once_with(c,5)


suite=unittest.defaultTestLoader.loadTestsFromTestCase(ClockChecks)
result=unittest.TextTestRunner(verbosity=2).run(suite)
(ROOT/'diagnostics/clock-runtime-check.json').write_text(json.dumps({'ok':result.wasSuccessful(),'tests':result.testsRun,'model_smokes':0},indent=2)+'\n')
if not result.wasSuccessful():raise SystemExit(1)
