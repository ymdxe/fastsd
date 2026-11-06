import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "baselines" / "specedge" / "repro.py"
SPEC = importlib.util.spec_from_file_location("specedge_repro", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
specedge_repro = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(specedge_repro)


class SpecEdgeReproTests(unittest.TestCase):
    def test_official_revision_is_pinned(self):
        self.assertEqual(
            specedge_repro.EXPECTED_OFFICIAL_SHA,
            "1edcaf02ffc41a7b57726450c5357ed216a3b9bc",
        )

    def test_six_paper_configs_validate(self):
        paths = specedge_repro.config_paths()
        self.assertEqual(len(paths), 6)
        for path in paths:
            with self.subTest(path=path.name):
                self.assertEqual(specedge_repro.validate_config(path), [])

    def test_paper_depth_examples(self):
        self.assertEqual(specedge_repro.recommend_depth(94.2, 11.0, 15.0), 7)
        self.assertEqual(specedge_repro.recommend_depth(94.2, 11.0, 40.0), 5)
        self.assertEqual(specedge_repro.recommend_depth(94.2, 11.0, 50.0), 4)

    def test_depth_has_lower_bound(self):
        self.assertEqual(specedge_repro.recommend_depth(20.0, 11.0, 25.0), 1)


if __name__ == "__main__":
    unittest.main()
