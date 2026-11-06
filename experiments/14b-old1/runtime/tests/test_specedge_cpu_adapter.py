import contextlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = REPO_ROOT / "baselines" / "specedge" / "integration" / "cpu_adapter.py"


class _FakeTensor:
    def __init__(self, shape, values=None):
        self._shape = tuple(shape)
        self.values = list(values) if values is not None else None

    def size(self, dim=None):
        if dim is None:
            return self._shape
        return self._shape[dim]


class _FakeIndices:
    def __init__(self, values):
        self.values = list(values)

    def to(self, *args, **kwargs):
        return self


class _FakeKVCache:
    instance = None

    def __init__(self, config, batch_size, max_n_beams, max_len, device, dtype):
        self.seq_indices = _FakeIndices(range(max_n_beams))
        self.storage = {}
        self.update_calls = []
        self.gather_calls = []
        _FakeKVCache.instance = self

    @contextlib.contextmanager
    def prefill_context(self, n_beams, batch_idx):
        previous = self.seq_indices
        self.seq_indices = _FakeIndices(range(n_beams))
        try:
            yield
        finally:
            self.seq_indices = previous

    def update(
        self,
        k_cache,
        v_cache,
        layer_idx,
        cache_batch_indices,
        cache_seq_indices,
    ):
        source_indices = self.seq_indices.values
        source_width = k_cache.size(2)
        if any(index >= source_width for index in source_indices):
            raise IndexError(
                f"index {max(source_indices)} is out of bounds for dimension 1 "
                f"with size {source_width}"
            )

        destinations = cache_seq_indices.values
        self.update_calls.append((source_indices[:], destinations[:]))
        for source_index, destination in zip(source_indices, destinations):
            self.storage[destination] = k_cache.values[source_index]
        return None, None

    def gather(self, batch_idx, src_indices, dest_indices):
        sources = src_indices.values
        destinations = dest_indices.values
        self.gather_calls.append((batch_idx, sources[:], destinations[:]))
        copied = [self.storage.get(source) for source in sources]
        for destination, value in zip(destinations, copied):
            self.storage[destination] = value

    def clear(self):
        self.storage.clear()


class _FakeModel:
    device = "cpu"
    dtype = "float32"
    config = object()

    def __init__(self):
        self.forward_count = 0

    def forward(self, *, input_ids, cache_seq_indices, past_key_values, **kwargs):
        width = input_ids.size(1)
        base = 100 * (self.forward_count + 1)
        self.forward_count += 1
        key_values = _FakeTensor(
            (1, 1, width, 1), values=[base + index for index in range(width)]
        )
        past_key_values.update(
            key_values,
            key_values,
            0,
            kwargs["cache_batch_indices"],
            cache_seq_indices,
        )
        return _FakeTensor((1, width, 4)), None


def _load_adapter_with_fake_official_dependencies():
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = _FakeTensor
    fake_torch.long = object()
    fake_torch.inference_mode = lambda: (lambda function: function)
    fake_torch.arange = lambda stop, **kwargs: _FakeIndices(range(int(stop)))
    fake_torch.zeros = lambda shape, **kwargs: _FakeIndices([0] * int(shape[0]))

    fake_log = types.ModuleType("log")
    fake_log.get_logger = lambda: object()

    fake_util = types.ModuleType("util")
    fake_util.invert_mask = lambda value: value
    fake_util.Timing = object

    fake_model_package = types.ModuleType("model")
    fake_model_package.__path__ = []
    fake_model_cache = types.ModuleType("model.cache")
    fake_model_cache.KVCache = _FakeKVCache

    fake_specedge = types.ModuleType("specedge")
    fake_specedge.__path__ = []
    fake_specedge_client = types.ModuleType("specedge.client")
    fake_specedge_client.__path__ = []
    fake_specedge_specexec = types.ModuleType("specedge.client.specexec")
    fake_specedge_specexec.SpecExecClient = object

    stubs = {
        "torch": fake_torch,
        "log": fake_log,
        "util": fake_util,
        "model": fake_model_package,
        "model.cache": fake_model_cache,
        "specedge": fake_specedge,
        "specedge.client": fake_specedge_client,
        "specedge.client.specexec": fake_specedge_specexec,
    }
    module_name = "specedge_cpu_adapter_contract_test"
    spec = importlib.util.spec_from_file_location(module_name, ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


class SpecEdgeCpuAdapterContractTests(unittest.TestCase):
    def test_one_worker_prefill_forward_gather_round_trip_uses_source_width(self):
        adapter = _load_adapter_with_fake_official_dependencies()
        engine = adapter.CPUCompatibleSpecEdgeEngine(
            model=_FakeModel(), max_len=8, max_n_beams=4
        )

        engine.prefill(
            input_ids=_FakeTensor((1, 2)),
            position_ids=_FakeTensor((1, 2)),
            batch_idx=0,
            cache_seq_indices=_FakeIndices([0, 1]),
            attention_mask=_FakeTensor((1, 1, 2, 8)),
        )
        engine.forward(
            input_ids=_FakeTensor((1, 3)),
            position_ids=_FakeTensor((1, 3)),
            cache_batch_indices=_FakeIndices([0, 0, 0]),
            # These are tree destinations, not source positions in k_cache.
            cache_seq_indices=_FakeIndices([2, 4, 5]),
            attention_mask=_FakeTensor((1, 1, 3, 8)),
        )
        engine.gather(_FakeIndices([0, 2]), _FakeIndices([0, 1]))

        cache = _FakeKVCache.instance
        self.assertEqual(
            cache.update_calls,
            [([0, 1], [0, 1]), ([0, 1, 2], [2, 4, 5])],
        )
        self.assertEqual(cache.gather_calls, [(0, [0, 2], [0, 1])])
        self.assertEqual(cache.storage[0], 100)
        self.assertEqual(cache.storage[1], 200)


if __name__ == "__main__":
    unittest.main()
