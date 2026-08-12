import math
import random
import tempfile
import unittest
from pathlib import Path

from src.arrival import (
    build_trace_manifest,
    generate_poisson_trace,
    read_trace_jsonl,
    sha256_file,
    validate_trace_rows,
    write_trace_jsonl,
)


class PoissonArrivalTests(unittest.TestCase):
    @staticmethod
    def records(count=12):
        return [{"task_id": f"task-{index}"} for index in range(count)]

    def test_same_seed_produces_identical_trace(self):
        first = generate_poisson_trace(
            self.records(), rate_rps=2.5, seed=17, num_clients=3
        )
        second = generate_poisson_trace(
            self.records(), rate_rps=2.5, seed=17, num_clients=3
        )
        self.assertEqual(first, second)

    def test_different_seed_changes_interarrival_sequence(self):
        first = generate_poisson_trace(self.records(), rate_rps=2.5, seed=17)
        second = generate_poisson_trace(self.records(), rate_rps=2.5, seed=18)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first[0]["interarrival_s"], second[0]["interarrival_s"])

    def test_offsets_are_monotonic_and_preserve_first_sample(self):
        rate = 1.5
        seed = 9
        trace = generate_poisson_trace(self.records(), rate_rps=rate, seed=seed)
        offsets = [row["scheduled_offset_s"] for row in trace]
        expected_first_sample = random.Random(seed).expovariate(rate)
        self.assertEqual(trace[0]["interarrival_s"], expected_first_sample)
        self.assertEqual(trace[0]["scheduled_offset_s"], trace[0]["interarrival_s"])
        self.assertTrue(all(offset > 0.0 for offset in offsets))
        self.assertTrue(all(next_offset > offset for offset, next_offset in zip(offsets, offsets[1:])))

    def test_generation_does_not_change_global_random_state(self):
        random.seed(418)
        expected_next_value = random.random()
        random.seed(418)
        generate_poisson_trace(self.records(), rate_rps=1.5, seed=9)
        self.assertEqual(random.random(), expected_next_value)

    def test_empirical_interarrival_mean_matches_rate(self):
        rate = 2.5
        trace = generate_poisson_trace(
            self.records(20000), rate_rps=rate, seed=20260812
        )
        observed_mean = sum(row["interarrival_s"] for row in trace) / len(trace)
        self.assertAlmostEqual(observed_mean, 1.0 / rate, delta=0.02)

    def test_invalid_inputs_are_rejected(self):
        records = self.records()
        for invalid_rate in (0, -0.1, math.inf, -math.inf, math.nan, "bad"):
            with self.subTest(rate=invalid_rate):
                with self.assertRaises(ValueError):
                    generate_poisson_trace(records, rate_rps=invalid_rate, seed=1)

        with self.assertRaises(ValueError):
            generate_poisson_trace([], rate_rps=1.0, seed=1)
        with self.assertRaises(ValueError):
            generate_poisson_trace(records, rate_rps=1.0, seed=1, num_clients=0)
        with self.assertRaises(ValueError):
            generate_poisson_trace(records, rate_rps=1.0, seed=1, max_requests=13)

    def test_dataset_assignments_are_unique_and_client_assignment_is_stable(self):
        trace = generate_poisson_trace(
            self.records(9), rate_rps=1.0, seed=4, num_clients=4
        )
        self.assertEqual([row["dataset_index"] for row in trace], list(range(9)))
        self.assertEqual(len({row["dataset_index"] for row in trace}), 9)
        self.assertEqual([row["task_id"] for row in trace], [f"task-{i}" for i in range(9)])
        self.assertEqual([row["client_id"] for row in trace], [0, 1, 2, 3, 0, 1, 2, 3, 0])

    def test_jsonl_round_trip_and_manifest_are_stable(self):
        trace = generate_poisson_trace(self.records(4), rate_rps=3.0, seed=13)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "arrival_trace.jsonl"
            write_trace_jsonl(path, trace)
            self.assertEqual(read_trace_jsonl(path), trace)
            manifest = build_trace_manifest(path)
            self.assertEqual(manifest["request_count"], len(trace))
            self.assertEqual(manifest["trace_sha256"], sha256_file(path))
            with self.assertRaises(FileExistsError):
                write_trace_jsonl(path, trace)

    def test_validation_rejects_duplicate_dataset_assignment(self):
        trace = generate_poisson_trace(self.records(2), rate_rps=1.0, seed=6)
        trace[1]["dataset_index"] = trace[0]["dataset_index"]
        with self.assertRaisesRegex(ValueError, "duplicate dataset assignment"):
            validate_trace_rows(trace)


if __name__ == "__main__":
    unittest.main()
