from pathlib import Path
import tempfile
import unittest
from scripts.round2_experiment import prepare, verify_runtime, MATRIX


class RuntimeArtifactTests(unittest.TestCase):
    def test_frozen_runtime_is_verifiable_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'package'
            manifest = prepare(output, ['--temp', '0'])
            verified, runtime = verify_runtime(output)
            self.assertEqual(verified, manifest)
            self.assertIn('src/cache_transfer.py', manifest['files'])
            self.assertEqual(manifest['common_argv'][-2:], ['--temp', '0'])
            with self.assertRaises(FileExistsError):
                prepare(output)
            (runtime / 'src/cache_transfer.py').write_text('# changed')
            with self.assertRaisesRegex(ValueError, 'runtime changed'):
                verify_runtime(output)

    def test_prefetch_comparison_fixes_ahead_gamma_and_scheduler(self):
        for name in ('off', 'overlap', 'cpu_prepare'):
            argv = MATRIX[name]
            self.assertIn('--no-enable_proactive_draft', argv)
            self.assertIn('--no-pipeline_gamma_adapt', argv)
            self.assertEqual(argv[argv.index('--scheduler_policy') + 1], 'fcfs')
        # protect_forward was removed: overlap is the only speculative mode.
        self.assertNotIn('protect_forward', MATRIX)
        for name in ('ahead', 'gamma', 'progress'):
            self.assertIn('--kv_prefetch_mode', MATRIX[name])
            self.assertEqual(
                MATRIX[name][MATRIX[name].index('--kv_prefetch_mode') + 1], 'overlap')


if __name__ == '__main__':
    unittest.main()
