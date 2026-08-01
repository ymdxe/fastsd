import py_compile
from pathlib import Path
import unittest


class EdgeEntrypointTests(unittest.TestCase):
    def test_edge_module_compiles(self):
        repo = Path(__file__).resolve().parents[1]
        py_compile.compile(str(repo / "edge" / "edge.py"), doraise=True)


if __name__ == "__main__":
    unittest.main()
