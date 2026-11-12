"""Process ownership and cleanup integration checks; no SSH/GPU required."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from scripts.zjh_ab import managed_job as jobs

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/zjh_ab/managed_job.py'


class ManagedJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = self.root / 'job.json'

    def tearDown(self):
        if self.state.exists():
            self.call('stop', str(self.state), '--grace', '.2')
        self.tmp.cleanup()

    def call(self, *args):
        return subprocess.run([sys.executable, str(HELPER), *args],
                              capture_output=True, text=True, timeout=12)

    def start(self, program):
        result = self.call('start', '--state', str(self.state), '--log', str(self.root/'log'),
                          '--cwd', str(self.root), '--', sys.executable, '-c', program)
        self.assertEqual(result.returncode, 0, result.stderr)

    def await_value(self, predicate):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(.05)
        self.fail('timeout; log=' + (self.root/'log').read_text())

    def test_nonzero_exit_is_recorded(self):
        self.start('raise SystemExit(17)')
        self.await_value(lambda: jobs.read(self.state).get('phase') == 'finished')
        self.assertEqual(self.call('status', str(self.state)).stdout.strip(), '17')

    def test_stop_kills_descendants_in_new_session_and_preserves_unrelated(self):
        outsider = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            self.start('import subprocess, sys, time\n'
                       'p=subprocess.Popen([sys.executable,"-c",'
                       '"import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"],start_new_session=True)\n'
                       'open("child", "w").write(str(p.pid))\n'
                       'time.sleep(60)')
            self.await_value(lambda: (self.root/'child').exists())
            data = jobs.read(self.state)
            self.assertGreaterEqual(len(jobs.owned_pids(data['token'])), 2)
            self.assertEqual(self.call('stop', str(self.state), '--grace', '.3').returncode, 0)
            self.assertEqual(jobs.owned_pids(data['token']), [])
            self.assertIsNone(outsider.poll())
            self.assertEqual(self.call('stop', str(self.state), '--grace', '.1').returncode, 0)
        finally:
            outsider.terminate()
            outsider.wait(timeout=3)

    def test_recovery_after_supervisor_sigkill(self):
        self.start('import time; open("ready","w").write("yes"); time.sleep(60)')
        self.await_value(lambda: (self.root/'ready').exists())
        data = jobs.read(self.state)
        os.kill(data['supervisor_pid'], signal.SIGKILL)
        self.assertEqual(self.call('stop', str(self.state), '--grace', '.2').returncode, 0)
        self.assertEqual(jobs.owned_pids(data['token']), [])

    def test_stop_before_delayed_start_prevents_launch(self):
        self.assertEqual(self.call('stop', str(self.state)).returncode, 0)
        result = self.call('start', '--state', str(self.state), '--log', str(self.root/'log'),
                          '--cwd', str(self.root), '--', sys.executable, '-c', 'raise SystemExit(0)')
        self.assertNotEqual(result.returncode, 0)

    def test_stale_controller_identity_does_not_signal_pid(self):
        jobs.write(self.state, dict(pid=os.getpid(), start='impossible-start-time'))
        self.assertEqual(self.call('stop', str(self.state), '--grace', '.1').returncode, 0)

    def test_exit_trap_cleans_job_when_runner_fails(self):
        # Source production definitions only, then force failure after launch.
        text = (ROOT/'scripts/zjh_ab/runall.sh').read_text().split('\ncd "$ROOT"\n')[0]
        script = self.root/'failure.sh'
        state_dir = self.root/'registry'
        state_dir.mkdir()
        (state_dir/'config.env').write_text('LOCAL_PY=' + sys.executable + '\n')
        script.write_text(text + '\nROOT="$1"\nSTATE_DIR="$2"\n'
            'trap runner_exit EXIT\n'
            'launch_job local "$3" "$ROOT" "$4" "$5" "$3" -c "import time; time.sleep(60)"\n'
            'false\n')
        result = subprocess.run(['bash', str(script), str(ROOT), str(state_dir), sys.executable,
                                 str(self.state), str(self.root/'log')],
                                text=True, capture_output=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((state_dir/'runner.exit').exists(), result.stderr)
        self.assertEqual(jobs.owned_pids(jobs.read(self.state)['token']), [])

    def test_manual_stop_recovers_manifest_after_controller_exit(self):
        self.start('import time; time.sleep(60)')
        registry = self.root/'registry'/'test'
        registry.mkdir(parents=True)
        (registry/'config.env').write_text('LOCAL_PY=' + sys.executable + '\n')
        jobs.write(registry/'controller.json', dict(pid=os.getpid(), start='stale'))
        (registry/'jobs.tsv').write_text('local\t%s\t%s\t%s\n' % (sys.executable, HELPER, self.state))
        result = subprocess.run(['bash', str(ROOT/'scripts/zjh_ab/stop_runall.sh'), 'test'],
                                env=dict(os.environ, RUNALL_STATE_ROOT=str(registry.parent)),
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(jobs.owned_pids(jobs.read(self.state)['token']), [])


if __name__ == '__main__':
    unittest.main()
