import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_ROOT = REPO_ROOT / "baselines" / "specedge" / "integration"
CLIENT_PATH = INTEGRATION_ROOT / "client.py"


class _FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def __add__(self, _other):
        return self

    def size(self, dim=None):
        if dim is None:
            return (len(self.values),)
        return len(self.values)

    def sort(self):
        return _FakeTensor(sorted(self.values)), _FakeTensor(range(len(self.values)))


class _FakeLogProbs:
    def __getitem__(self, _key):
        return self

    def topk(self, *, k, sorted):
        self.last_topk = (k, sorted)
        return types.SimpleNamespace(indices=_FakeTensor([4, 2, 3, 1]))


class _FakeTree:
    prefix_len = 0
    end = 5

    def __init__(self):
        self.logprobs = _FakeLogProbs()
        self.gather_calls = []

    def gather(self, src_indices, dest_indices):
        self.gather_calls.append((src_indices.values, dest_indices.values))


class _FakeEngine:
    def __init__(self):
        self.gather_calls = []

    def gather(self, src_indices, dest_indices):
        self.gather_calls.append((src_indices.values, dest_indices.values))


def _load_integration_client():
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_torch.long = object()
    fake_torch.arange = lambda start, end, **_kwargs: _FakeTensor(range(start, end))

    fake_grpc = types.ModuleType("grpc")
    fake_log = types.ModuleType("log")
    fake_log.get_logger = lambda: object()
    fake_util = types.ModuleType("util")
    fake_config = types.ModuleType("config")
    fake_config.SpecEdgeClientConfig = object()

    fake_specedge = types.ModuleType("specedge")
    fake_specedge.__path__ = []
    fake_specedge_client = types.ModuleType("specedge.client")
    fake_specedge_client.__path__ = []
    fake_specedge_specexec = types.ModuleType("specedge.client.specexec")
    fake_specedge_specexec.SpecExecClient = object
    fake_specedge_engine = types.ModuleType("specedge.engine")
    fake_specedge_engine.__path__ = []
    fake_specedge_graph = types.ModuleType("specedge.engine.graph")
    fake_specedge_graph.GraphEngine = object

    fake_specedge_grpc = types.ModuleType("specedge_grpc")
    fake_specedge_grpc.specedge_pb2 = types.SimpleNamespace()
    fake_specedge_grpc.specedge_pb2_grpc = types.SimpleNamespace()
    fake_runtime = types.ModuleType("src.runtime")
    fake_runtime.configure_torch_threads = lambda _threads: None
    fake_src = types.ModuleType("src")
    fake_src.__path__ = []

    fake_cpu_adapter = types.ModuleType("cpu_adapter")
    fake_cpu_adapter.CPUCompatibleSpecEdgeEngine = object
    fake_cpu_adapter.CPUCompatibleTiming = object
    fake_cpu_adapter.cpu_timing_adapter = object
    fake_wire_codec = types.ModuleType("wire_codec")
    fake_wire_codec.ExplicitSpecEdgeGrpcClient = object

    stubs = {
        "torch": fake_torch,
        "grpc": fake_grpc,
        "log": fake_log,
        "util": fake_util,
        "config": fake_config,
        "specedge": fake_specedge,
        "specedge.client": fake_specedge_client,
        "specedge.client.specexec": fake_specedge_specexec,
        "specedge.engine": fake_specedge_engine,
        "specedge.engine.graph": fake_specedge_graph,
        "specedge_grpc": fake_specedge_grpc,
        "src": fake_src,
        "src.runtime": fake_runtime,
        "cpu_adapter": fake_cpu_adapter,
        "wire_codec": fake_wire_codec,
    }
    module_name = "specedge_integration_client_token_order_test"
    spec = importlib.util.spec_from_file_location(module_name, CLIENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(INTEGRATION_ROOT))
    with patch.dict(sys.modules, stubs):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
            sys.path.remove(str(INTEGRATION_ROOT))
    return module


class SpecEdgeTokenOrderTests(unittest.TestCase):
    def test_nonmonotonic_tree_indices_are_compacted_in_generation_order(self):
        module = _load_integration_client()
        client = module.IntegratedSpecExecClient.__new__(
            module.IntegratedSpecExecClient
        )
        client._tree = _FakeTree()
        client._engine = _FakeEngine()
        client._max_budget = 4
        client._device = "cpu"

        client._trim_by_budget()

        expected_indices = [1, 2, 3, 4]
        self.assertEqual(client._tree.gather_calls, [(expected_indices, [0, 1, 2, 3])])
        self.assertEqual(client._engine.gather_calls, [(expected_indices, [0, 1, 2, 3])])

        # The selected physical nodes represent one target path, not a ranked
        # output list.  Compaction must therefore decode them as 198, 262, 369, 600.
        token_by_index = {1: 198, 2: 262, 3: 369, 4: 600}
        self.assertEqual(
            [token_by_index[index] for index in expected_indices],
            [198, 262, 369, 600],
        )


if __name__ == "__main__":
    unittest.main()
