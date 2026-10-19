import unittest
from pathlib import Path


class RuntimeDefaultsTests(unittest.TestCase):
    def test_cloud_and_edge_default_to_port_8001(self):
        repo = Path(__file__).resolve().parents[1]
        edge_text = (repo / "edge" / "edge.py").read_text(encoding="utf-8")
        cloud_text = (repo / "cloud" / "cloud_service.py").read_text(encoding="utf-8")

        self.assertIn('default="http://127.0.0.1:8001"', edge_text)
        self.assertIn('CLOUD_SERVICE_PORT', cloud_text)
        self.assertIn('"8001"', cloud_text)


if __name__ == "__main__":
    unittest.main()
