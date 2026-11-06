import random
import unittest

from src.arrival import poisson_arrival_offsets, shard_samples


class PoissonArrivalTests(unittest.TestCase):
    def test_offsets_are_deterministic_cumulative_exponentials(self):
        expected_rng = random.Random(17)
        expected = []
        elapsed = 0.0
        for _ in range(4):
            elapsed += expected_rng.expovariate(2.5)
            expected.append(elapsed)

        actual = poisson_arrival_offsets(4, 2.5, 17)

        self.assertEqual(actual, expected)
        self.assertTrue(all(left < right for left, right in zip(actual, actual[1:])))

    def test_invalid_poisson_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            poisson_arrival_offsets(-1, 1.0, 1)
        with self.assertRaises(ValueError):
            poisson_arrival_offsets(1, 0.0, 1)

    def test_round_robin_shards_cover_dataset_once(self):
        samples = list("abcdefgh")
        shards = [shard_samples(samples, 3, shard_id, 0) for shard_id in range(3)]
        flattened = sorted(item for shard in shards for item in shard)
        self.assertEqual(flattened, list(enumerate(samples)))

    def test_positive_limit_applies_per_shard(self):
        selected = shard_samples(list(range(10)), 2, 1, 2)
        self.assertEqual(selected, [(1, 1), (3, 3)])


if __name__ == "__main__":
    unittest.main()
