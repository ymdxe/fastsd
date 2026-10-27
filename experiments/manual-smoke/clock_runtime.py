"""Maintain clock estimates without replaying a completed model verification."""
import json
import os
from pathlib import Path
import time
import requests


class ClockRuntimeMixin:
    clock_refresh_interval_s=60

    def _clock_publish(self,event=None):
        directory=os.environ.get('EXP_OUTPUT')
        if not directory:return
        directory=Path(directory)
        counters=getattr(self,'_clock_counters',{})
        snapshot=dict(counters,clock_offset_ns=self.clock_offset_ns,
                      clock_uncertainty_ns=self.clock_uncertainty_ns,epoch=time.time())
        path=directory/'timing-runtime-status.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(snapshot)+'\n');temporary.replace(path)
        if event:
            with (directory/'timing-runtime.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(event,**snapshot))+'\n')

    def _clock_count(self,key):
        if not hasattr(self,'_clock_counters'):self._clock_counters={}
        self._clock_counters[key]=self._clock_counters.get(key,0)+1

    def _refresh_runtime_clock(self,reason):
        start=time.monotonic()
        result=self._synchronize_runtime_clock(samples=5)
        self._clock_refresh_at=time.monotonic()
        self._clock_count('recalibrations')
        self._clock_publish({'event':'recalibration','reason':reason,'elapsed_ms':(time.monotonic()-start)*1000})
        return result

    def _maybe_refresh_runtime_clock(self):
        if time.monotonic()-getattr(self,'_clock_refresh_at',0)>self.clock_refresh_interval_s:
            self._refresh_runtime_clock('periodic')

    def verify(self,*args,**kwargs):
        self._maybe_refresh_runtime_clock()
        return super().verify(*args,**kwargs)

    def prefill(self,*args,**kwargs):
        self._maybe_refresh_runtime_clock()
        return super().prefill(*args,**kwargs)

    def _post(self,path,payload):
        for attempt in range(3):
            try:return super()._post(path,payload)
            except requests.HTTPError as exc:
                response=exc.response
                try:detail=response.json().get('detail','') if response is not None else ''
                except (ValueError,AttributeError):detail=''
                # The server rejects this timestamp before enqueue/admission.
                # Other errors, including ambiguous disconnects, are not replayed.
                if path!='/verify' or response is None or response.status_code!=422 or not str(detail).startswith('synchronized clocks produced a negative upload duration') or attempt==2:raise
                self._refresh_runtime_clock('upload_rejected_before_admission')
                payload=dict(payload,clock_offset_ns=int(self.clock_offset_ns),edge_send_time_ns=time.time_ns())
                self._clock_count('upload_pre_admission_retries')
                self._clock_publish({'event':'upload_retry','attempt':attempt+1})

    def _attach_pull(self,response,edge_receive_time_ns):
        recalibrated=False
        try:
            result=super()._attach_pull(response,edge_receive_time_ns)
        except RuntimeError as exc:
            if str(exc)!='synchronized clocks produced a negative pull duration; retry clock calibration':raise
            original_raw=int(edge_receive_time_ns)-int(response['cloud_send_time_ns'])+int(self.clock_offset_ns)
            self._refresh_runtime_clock('negative_pull')
            recalibrated=True
            try:
                # Reuse the original receive timestamp. Calibration delay is
                # not included in the one-way duration, and Verify is not resent.
                result=super()._attach_pull(response,edge_receive_time_ns)
            except RuntimeError as second:
                if str(second)!='synchronized clocks produced a negative pull duration; retry clock calibration':raise
                raw=int(edge_receive_time_ns)-int(response['cloud_send_time_ns'])+int(self.clock_offset_ns)
                estimate=getattr(self,'_last_valid_pull_s',None)
                result=response
                result.update(pull_timing_valid=False,measured_pull_ms=None,pull_raw_ns=raw,
                              pull_uncertainty_ns=self.clock_uncertainty_ns,
                              pull_s_source='previous_valid_scheduler_estimate' if estimate is not None else 'unavailable')
                if estimate is None:
                    result.pop('pull_s',None);result.pop('pull_ms',None)
                else:result.update(pull_s=estimate,pull_ms=estimate*1000)
                self._clock_count('unreliable_pull_measurements')
                self._clock_publish({'event':'unreliable_pull','raw_before_ns':original_raw,'raw_after_ns':raw,
                                     'scheduler_estimate_s':estimate,'cloud_send_time_ns':response['cloud_send_time_ns'],
                                     'edge_receive_time_ns':edge_receive_time_ns})
                return result
        if response.get('cloud_send_time_ns') is not None:
            self._last_valid_pull_s=result['pull_s']
            result.update(pull_timing_valid=True,measured_pull_ms=result['pull_ms'],
                          pull_s_source='recalibrated_clock' if recalibrated else 'calibrated_clock',
                          pull_uncertainty_ns=self.clock_uncertainty_ns)
            self._clock_count('valid_pull_measurements')
            if self._clock_counters['valid_pull_measurements']%100==0:self._clock_publish()
        return result
