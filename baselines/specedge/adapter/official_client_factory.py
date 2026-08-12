"""Explicit, lazy bridge from a saved trace to pinned Official SpecEdge.

Nothing in this module imports CUDA, torch, grpc, PyYAML, or the ``official``
submodule at import time.  The heavyweight runtime is loaded only when the
explicit factory is prepared by :mod:`poisson_client`, after the caller has
provided all paths and the exact gRPC endpoint.

Official SpecEdge keeps its client configuration in a process-global class.
Therefore one adapter process represents exactly one logical edge client and
one visible GPU (``CUDA_VISIBLE_DEVICES=<one physical GPU>`` and
``SPECEDGE_DEVICE=cuda:0``).  A two-client experiment must use two adapter
processes, each replaying its own ``--client-id`` trace partition; it must not
share the global Official configuration across two logical clients in one
Python process.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import operator
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from .poisson_client import AdapterConfigurationError, ClientInvoker, ReplayContext


class OfficialSpecEdgeConfigurationError(AdapterConfigurationError):
    """Raised before a real Official SpecEdge API is imported or called."""


@dataclass(frozen=True)
class EnvironmentSettings:
    """Explicit process inputs needed to build one Official client runtime."""

    official_root: Path
    effective_config_path: Path
    prompts_path: Path
    grpc_address: str
    client_id: int
    device: str
    sync_timeout_s: float
    client_result_path: str | None
    client_exp_name: str | None


@dataclass(frozen=True)
class EffectiveClientConfig:
    """The exact Official ``SpecEdgeClientConfig`` fields derived from YAML."""

    result_path: str
    exp_name: str
    process_name: str
    seed: int
    optimization: int
    max_len: int
    draft_model: str
    dtype: str
    reasoning: bool
    dataset: str
    max_n_beams: int
    max_beam_len: int
    max_branch_width: int
    max_budget: int
    proactive_type: str
    proactive_max_n_beams: int
    proactive_max_beam_len: int
    proactive_max_branch_width: int
    proactive_max_budget: int
    max_new_tokens: int
    max_request_num: int
    req_offset: int
    sample_req_cnt: int
    host: str

    def to_official_environment(self, settings: EnvironmentSettings) -> dict[str, str]:
        """Render the documented Official client environment without defaults."""

        # The server and its two edge clients live on different hosts.  The
        # algorithm settings must come from one effective YAML, while each
        # client must put Official's own log/result files under its local,
        # fresh component directory.  These two explicit overrides affect
        # only upstream logging paths, never the tree/speculation parameters.
        result_path = settings.client_result_path or self.result_path
        exp_name = settings.client_exp_name or self.exp_name

        return {
            "SPECEDGE_RESULT_PATH": result_path,
            "SPECEDGE_EXP_NAME": exp_name,
            # Official's client_host.py does the same suffixing.  Without it
            # the two process-global clients would concurrently overwrite one
            # upstream log/result filename.
            "SPECEDGE_PROCESS_NAME": f"{self.process_name}_{settings.client_id}",
            "SPECEDGE_SEED": str(self.seed),
            "SPECEDGE_OPTIMIZATION": str(self.optimization),
            "SPECEDGE_MAX_LEN": str(self.max_len),
            "SPECEDGE_DRAFT_MODEL": self.draft_model,
            "SPECEDGE_DEVICE": settings.device,
            "SPECEDGE_DTYPE": self.dtype,
            "SPECEDGE_REASONING": str(self.reasoning),
            "SPECEDGE_DATASET": self.dataset,
            "SPECEDGE_MAX_N_BEAMS": str(self.max_n_beams),
            "SPECEDGE_MAX_BEAM_LEN": str(self.max_beam_len),
            "SPECEDGE_MAX_BRANCH_WIDTH": str(self.max_branch_width),
            "SPECEDGE_MAX_BUDGET": str(self.max_budget),
            "SPECEDGE_PROACTIVE_TYPE": self.proactive_type,
            "SPECEDGE_PROACTIVE_MAX_N_BEAMS": str(self.proactive_max_n_beams),
            "SPECEDGE_PROACTIVE_MAX_BEAM_LEN": str(self.proactive_max_beam_len),
            "SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH": str(
                self.proactive_max_branch_width
            ),
            "SPECEDGE_PROACTIVE_MAX_BUDGET": str(self.proactive_max_budget),
            "SPECEDGE_MAX_NEW_TOKENS": str(self.max_new_tokens),
            "SPECEDGE_MAX_REQUEST_NUM": str(self.max_request_num),
            "SPECEDGE_REQ_OFFSET": str(self.req_offset),
            "SPECEDGE_SAMPLE_REQ_CNT": str(self.sample_req_cnt),
            # The endpoint is intentionally supplied explicitly and checked
            # against the YAML before it reaches the Official client.
            "SPECEDGE_HOST": settings.grpc_address,
            "SPECEDGE_CLIENT_IDX": str(settings.client_id),
        }


@dataclass(frozen=True)
class _OfficialComponents:
    grpc: ModuleType
    client_config: Any
    graph_engine_type: Any
    pb2: ModuleType
    pb2_grpc: ModuleType
    spec_exec_client_type: Any
    util: ModuleType


def _required_environment(environ: Mapping[str, str], key: str) -> str:
    value = environ.get(key)
    if value is None or not value.strip():
        raise OfficialSpecEdgeConfigurationError(
            f"{key} must be set explicitly for the Official SpecEdge adapter"
        )
    return value


def _read_existing_file(value: str, *, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise OfficialSpecEdgeConfigurationError(f"{name} is not a file: {path}")
    return path


def _read_existing_directory(value: str, *, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise OfficialSpecEdgeConfigurationError(f"{name} is not a directory: {path}")
    return path


def _parse_non_negative_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise OfficialSpecEdgeConfigurationError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise OfficialSpecEdgeConfigurationError(f"{name} must be >= 0")
    return parsed


def _parse_positive_float(value: str, *, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise OfficialSpecEdgeConfigurationError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise OfficialSpecEdgeConfigurationError(f"{name} must be > 0")
    return parsed


def read_environment_settings(
    environ: Mapping[str, str] | None = None,
) -> EnvironmentSettings:
    """Read every runtime path/endpoint explicitly; never infer a transport."""

    values = os.environ if environ is None else environ
    official_root = _read_existing_directory(
        _required_environment(values, "SPECEDGE_OFFICIAL_ROOT"),
        name="SPECEDGE_OFFICIAL_ROOT",
    )
    if not (official_root / "src" / "config.py").is_file():
        raise OfficialSpecEdgeConfigurationError(
            "SPECEDGE_OFFICIAL_ROOT does not contain the pinned Official "
            f"SpecEdge source tree: {official_root}"
        )

    effective_config_path = _read_existing_file(
        _required_environment(values, "SPECEDGE_EFFECTIVE_CONFIG"),
        name="SPECEDGE_EFFECTIVE_CONFIG",
    )
    prompts_path = _read_existing_file(
        _required_environment(values, "SPECEDGE_PROMPTS_PATH"),
        name="SPECEDGE_PROMPTS_PATH",
    )
    grpc_address = _required_environment(values, "SPECEDGE_GRPC_ADDRESS")
    if "://" in grpc_address or "/" in grpc_address:
        raise OfficialSpecEdgeConfigurationError(
            "SPECEDGE_GRPC_ADDRESS must be an explicit host:port, not a URL"
        )

    client_id = _parse_non_negative_int(
        _required_environment(values, "SPECEDGE_CLIENT_ID"), name="SPECEDGE_CLIENT_ID"
    )
    device = values.get("SPECEDGE_DEVICE", "cuda:0")
    if device != "cuda:0":
        raise OfficialSpecEdgeConfigurationError(
            "SPECEDGE_DEVICE must be cuda:0: one adapter process owns one "
            "CUDA_VISIBLE_DEVICES GPU"
        )
    visible_devices = values.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices or "," in visible_devices:
        raise OfficialSpecEdgeConfigurationError(
            "CUDA_VISIBLE_DEVICES must expose exactly one physical GPU for "
            "the Official SpecEdge client process"
        )
    sync_timeout_s = _parse_positive_float(
        values.get("SPECEDGE_SYNC_TIMEOUT_S", "10"), name="SPECEDGE_SYNC_TIMEOUT_S"
    )
    client_result_path = values.get("SPECEDGE_CLIENT_RESULT_PATH") or None
    if client_result_path is not None and not Path(client_result_path).is_absolute():
        raise OfficialSpecEdgeConfigurationError(
            "SPECEDGE_CLIENT_RESULT_PATH must be absolute when supplied"
        )
    client_exp_name = values.get("SPECEDGE_CLIENT_EXP_NAME") or None
    if client_exp_name is not None and (
        "/" in client_exp_name or "\\" in client_exp_name
    ):
        raise OfficialSpecEdgeConfigurationError(
            "SPECEDGE_CLIENT_EXP_NAME must be a non-empty path component"
        )
    return EnvironmentSettings(
        official_root=official_root,
        effective_config_path=effective_config_path,
        prompts_path=prompts_path,
        grpc_address=grpc_address,
        client_id=client_id,
        device=device,
        sync_timeout_s=sync_timeout_s,
        client_result_path=client_result_path,
        client_exp_name=client_exp_name,
    )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialSpecEdgeConfigurationError(f"effective config {name} must be a mapping")
    return value


def _config_value(mapping: Mapping[str, Any], key: str, *, location: str) -> Any:
    if key not in mapping:
        raise OfficialSpecEdgeConfigurationError(
            f"effective config is missing {location}.{key}"
        )
    return mapping[key]


def _string_value(mapping: Mapping[str, Any], key: str, *, location: str) -> str:
    value = _config_value(mapping, key, location=location)
    if not isinstance(value, str) or not value:
        raise OfficialSpecEdgeConfigurationError(
            f"effective config {location}.{key} must be a non-empty string"
        )
    return value


def _integer_value(
    mapping: Mapping[str, Any],
    key: str,
    *,
    location: str,
    minimum: int | None = None,
) -> int:
    value = _config_value(mapping, key, location=location)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OfficialSpecEdgeConfigurationError(
            f"effective config {location}.{key} must be an integer"
        )
    if minimum is not None and value < minimum:
        raise OfficialSpecEdgeConfigurationError(
            f"effective config {location}.{key} must be >= {minimum}"
        )
    return value


def _load_yaml_document(path: Path) -> Mapping[str, Any]:
    """Import PyYAML only when an explicit real factory is prepared."""

    try:
        yaml = importlib.import_module("yaml")
    except ImportError as exc:
        raise OfficialSpecEdgeConfigurationError(
            "PyYAML is required to read SPECEDGE_EFFECTIVE_CONFIG in the "
            "dedicated Official SpecEdge environment"
        ) from exc
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OfficialSpecEdgeConfigurationError(
            f"could not parse effective SpecEdge YAML {path}: {exc}"
        ) from exc
    return _mapping(loaded, name="root")


def load_effective_client_config(
    path: str | Path, *, grpc_address: str
) -> EffectiveClientConfig:
    """Validate an official-compatible YAML and render no hidden defaults."""

    config_path = Path(path)
    document = _load_yaml_document(config_path)
    base = _mapping(_config_value(document, "base", location="root"), name="base")
    client = _mapping(
        _config_value(document, "client", location="root"), name="client"
    )
    proactive = _mapping(
        _config_value(client, "proactive", location="client"), name="client.proactive"
    )
    config_host = _string_value(client, "host", location="client")
    if config_host != grpc_address:
        raise OfficialSpecEdgeConfigurationError(
            "SPECEDGE_GRPC_ADDRESS must exactly match effective config client.host; "
            "the adapter will not guess or rewrite an Official endpoint"
        )

    optimization_value = document.get("opt", base.get("optimization"))
    if isinstance(optimization_value, bool) or not isinstance(optimization_value, int):
        raise OfficialSpecEdgeConfigurationError(
            "effective config must provide integer root.opt or base.optimization"
        )
    reasoning_value = _config_value(client, "reasoning", location="client")
    if not isinstance(reasoning_value, bool):
        raise OfficialSpecEdgeConfigurationError(
            "effective config client.reasoning must be a boolean"
        )

    return EffectiveClientConfig(
        result_path=_string_value(base, "result_path", location="base"),
        exp_name=_string_value(base, "exp_name", location="base"),
        process_name=_string_value(client, "process_name", location="client"),
        seed=_integer_value(base, "seed", location="base", minimum=0),
        optimization=optimization_value,
        max_len=_integer_value(base, "max_len", location="base", minimum=1),
        draft_model=_string_value(client, "draft_model", location="client"),
        dtype=_string_value(base, "dtype", location="base"),
        reasoning=reasoning_value,
        dataset=_string_value(client, "dataset", location="client"),
        max_n_beams=_integer_value(client, "max_n_beams", location="client", minimum=1),
        max_beam_len=_integer_value(client, "max_beam_len", location="client", minimum=0),
        max_branch_width=_integer_value(
            client, "max_branch_width", location="client", minimum=1
        ),
        max_budget=_integer_value(client, "max_budget", location="client", minimum=1),
        proactive_type=_string_value(proactive, "type", location="client.proactive"),
        proactive_max_n_beams=_integer_value(
            proactive, "max_n_beams", location="client.proactive", minimum=1
        ),
        proactive_max_beam_len=_integer_value(
            proactive, "max_beam_len", location="client.proactive", minimum=0
        ),
        proactive_max_branch_width=_integer_value(
            proactive, "max_branch_width", location="client.proactive", minimum=1
        ),
        proactive_max_budget=_integer_value(
            proactive, "max_budget", location="client.proactive", minimum=1
        ),
        max_new_tokens=_integer_value(
            client, "max_new_tokens", location="client", minimum=1
        ),
        max_request_num=_integer_value(
            client, "max_request_num", location="client", minimum=-1
        ),
        req_offset=_integer_value(client, "req_offset", location="client", minimum=0),
        sample_req_cnt=_integer_value(
            client, "sample_req_cnt", location="client", minimum=1
        ),
        host=config_host,
    )


def _load_saved_prompts(path: Path) -> dict[int, Mapping[str, Any]]:
    prompts: dict[int, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise OfficialSpecEdgeConfigurationError(
                    f"saved prompts line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(row, Mapping):
                raise OfficialSpecEdgeConfigurationError(
                    f"saved prompts line {line_number} must be a JSON object"
                )
            arrival_index = row.get("arrival_index")
            if isinstance(arrival_index, bool) or not isinstance(arrival_index, int) or arrival_index < 0:
                raise OfficialSpecEdgeConfigurationError(
                    f"saved prompts line {line_number} arrival_index must be a non-negative integer"
                )
            if arrival_index in prompts:
                raise OfficialSpecEdgeConfigurationError(
                    f"saved prompts has duplicate arrival_index {arrival_index}"
                )
            prompt = row.get("prompt")
            if not isinstance(prompt, str):
                raise OfficialSpecEdgeConfigurationError(
                    f"saved prompts line {line_number} prompt must be a string"
                )
            prompts[arrival_index] = dict(row)
    if not prompts:
        raise OfficialSpecEdgeConfigurationError(f"saved prompts is empty: {path}")
    return prompts


def _ensure_official_module(module: ModuleType, *, source_root: Path, name: str) -> None:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return
    try:
        Path(module_file).resolve().relative_to(source_root)
    except ValueError as exc:
        raise OfficialSpecEdgeConfigurationError(
            f"refusing to mix a previously imported {name} module from {module_file} "
            f"with the pinned Official SpecEdge root {source_root.parent}"
        ) from exc


def _import_official_components(official_root: Path) -> _OfficialComponents:
    """Import the known upstream client components only after configuration."""

    source_root = official_root / "src"
    if not source_root.is_dir():
        raise OfficialSpecEdgeConfigurationError(
            f"Official SpecEdge source directory is missing: {source_root}"
        )
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    importlib.invalidate_caches()

    try:
        config_module = importlib.import_module("config")
        util_module = importlib.import_module("util")
        grpc_module = importlib.import_module("grpc")
        spec_exec_module = importlib.import_module("specedge.client.specexec")
        graph_module = importlib.import_module("specedge.engine.graph")
        pb2 = importlib.import_module("specedge_grpc.specedge_pb2")
        pb2_grpc = importlib.import_module("specedge_grpc.specedge_pb2_grpc")
    except Exception as exc:
        raise OfficialSpecEdgeConfigurationError(
            "could not import the pinned Official SpecEdge client dependencies; "
            "activate its dedicated environment and verify SPECEDGE_OFFICIAL_ROOT"
        ) from exc

    for module, name in ((config_module, "config"), (util_module, "util")):
        _ensure_official_module(module, source_root=source_root, name=name)
    return _OfficialComponents(
        grpc=grpc_module,
        client_config=config_module.SpecEdgeClientConfig,
        graph_engine_type=graph_module.GraphEngine,
        pb2=pb2,
        pb2_grpc=pb2_grpc,
        spec_exec_client_type=spec_exec_module.SpecExecClient,
        util=util_module,
    )


class OfficialSpecEdgeFactory:
    """One explicit Official SpecEdge Draft runtime for one edge process."""

    def __init__(
        self,
        settings: EnvironmentSettings,
        config: EffectiveClientConfig,
        prompts: Mapping[int, Mapping[str, Any]],
    ) -> None:
        self.settings = settings
        self.config = config
        self.prompts = prompts
        self._components: _OfficialComponents | None = None
        self._engine: Any = None
        self._tokenizer: Any = None
        self._generation_lock: asyncio.Lock | None = None

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "OfficialSpecEdgeFactory":
        settings = read_environment_settings(environ)
        config = load_effective_client_config(
            settings.effective_config_path, grpc_address=settings.grpc_address
        )
        prompts = _load_saved_prompts(settings.prompts_path)
        return cls(settings=settings, config=config, prompts=prompts)

    def prepare(self) -> None:
        """Load the one known client runtime before replay timing starts."""

        if self._components is not None:
            return
        os.environ.update(self.config.to_official_environment(self.settings))
        components = _import_official_components(self.settings.official_root)
        try:
            # It is global in upstream; reset makes inherited process state
            # explicit and identical to this factory's effective YAML.
            components.client_config.reset()
            draft_model = components.util.load_graph_model(
                name=components.client_config.draft_model,
                device=components.client_config.device,
                dtype=components.client_config.dtype,
            )
            engine = components.graph_engine_type(
                model=draft_model,
                max_len=components.client_config.max_len,
                max_n_beams=components.client_config.max_n_beams,
            )
            tokenizer = components.util.load_tokenizer(
                components.client_config.draft_model
            )
            with components.grpc.insecure_channel(self.settings.grpc_address) as channel:
                stub = components.pb2_grpc.SpecEdgeServiceStub(channel)
                stub.Sync(
                    components.pb2.SyncRequest(), timeout=self.settings.sync_timeout_s
                )
        except Exception as exc:
            raise OfficialSpecEdgeConfigurationError(
                "Official SpecEdge prepare failed before replay; verify the pinned "
                "server, model, and dedicated environment"
            ) from exc

        self._components = components
        self._engine = engine
        self._tokenizer = tokenizer

    @staticmethod
    def _trace_int(request: Mapping[str, Any], key: str) -> int:
        value = request.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OfficialSpecEdgeConfigurationError(
                f"canonical trace request field {key!r} must be a non-negative integer"
            )
        return value

    def _prompt_for_request(
        self, request: Mapping[str, Any], context: ReplayContext
    ) -> tuple[int, str]:
        if context.client_id != self.settings.client_id:
            raise OfficialSpecEdgeConfigurationError(
                f"replay context client_id {context.client_id!r} does not match "
                f"SPECEDGE_CLIENT_ID={self.settings.client_id}"
            )
        trace_client_id = self._trace_int(request, "client_id")
        if trace_client_id != self.settings.client_id:
            raise OfficialSpecEdgeConfigurationError(
                f"trace client_id {trace_client_id} does not match "
                f"SPECEDGE_CLIENT_ID={self.settings.client_id}"
            )
        arrival_index = self._trace_int(request, "arrival_index")
        if arrival_index != context.trace_index:
            raise OfficialSpecEdgeConfigurationError(
                "trace arrival_index does not match replay trace_index; refusing "
                "to map a saved prompt by a guessed index"
            )
        dataset_index = self._trace_int(request, "dataset_index")
        saved = self.prompts.get(arrival_index)
        if saved is None:
            raise OfficialSpecEdgeConfigurationError(
                f"saved prompts has no row for arrival_index {arrival_index}"
            )
        saved_dataset_index = saved.get("dataset_index")
        if saved_dataset_index != dataset_index:
            raise OfficialSpecEdgeConfigurationError(
                f"saved prompt dataset_index disagrees with trace at arrival_index {arrival_index}"
            )
        if "task_id" in saved and "task_id" in request and saved["task_id"] != request["task_id"]:
            raise OfficialSpecEdgeConfigurationError(
                f"saved prompt task_id disagrees with trace at arrival_index {arrival_index}"
            )
        prompt = saved["prompt"]
        if not isinstance(prompt, str):  # defended again for direct construction/tests.
            raise OfficialSpecEdgeConfigurationError(
                f"saved prompt at arrival_index {arrival_index} is not a string"
            )
        return dataset_index, prompt

    @staticmethod
    def _token_count(value: Any, *, field: str) -> int:
        """Validate a count exposed by the pinned Official client object."""

        if isinstance(value, bool):
            raise OfficialSpecEdgeConfigurationError(
                f"Official SpecEdge {field} must be a non-negative integer"
            )
        try:
            count = operator.index(value)
        except TypeError as exc:
            raise OfficialSpecEdgeConfigurationError(
                f"Official SpecEdge {field} must be a non-negative integer"
            ) from exc
        if count < 0:
            raise OfficialSpecEdgeConfigurationError(
                f"Official SpecEdge {field} must be a non-negative integer"
            )
        return count

    @classmethod
    def _tensor_token_count(cls, value: Any, *, field: str) -> int:
        """Read ``Tensor.numel`` without importing or naming a tensor type."""

        numel = getattr(value, "numel", None)
        if not callable(numel):
            raise OfficialSpecEdgeConfigurationError(
                f"Official SpecEdge {field} must expose callable numel()"
            )
        try:
            raw_count = numel()
        except Exception as exc:
            raise OfficialSpecEdgeConfigurationError(
                f"could not read Official SpecEdge {field}.numel()"
            ) from exc
        return cls._token_count(raw_count, field=field)

    @classmethod
    def _extract_official_completion(cls, client: Any) -> dict[str, Any]:
        """Extract a continuation using the fixed upstream client semantics.

        The pinned ``SpecExecClient.generate`` returns ``None`` but leaves its
        generated sequence in ``_prefix_tokens``.  Its own final log line is
        exactly ``tokenizer.decode(_prefix_tokens[0],
        skip_special_tokens=True)``.  We perform that identical full-sequence
        decode for provenance, then decode only the token suffix after the
        upstream-recorded ``_num_original_tokens`` for the scorer-facing
        completion.  This deliberately does not infer a completion by string
        slicing the prompt, which would be tokenizer-unsafe.
        """

        prefix_tokens = getattr(client, "_prefix_tokens", None)
        if prefix_tokens is None:
            raise OfficialSpecEdgeConfigurationError(
                "pinned Official SpecEdge client has no _prefix_tokens after generate"
            )
        try:
            token_ids = prefix_tokens[0]
        except Exception as exc:
            raise OfficialSpecEdgeConfigurationError(
                "pinned Official SpecEdge _prefix_tokens must contain one sequence"
            ) from exc

        full_sequence_token_count = cls._tensor_token_count(
            prefix_tokens, field="_prefix_tokens"
        )
        sequence_token_count = cls._tensor_token_count(
            token_ids, field="_prefix_tokens[0]"
        )
        if full_sequence_token_count != sequence_token_count:
            raise OfficialSpecEdgeConfigurationError(
                "pinned Official SpecEdge completion extraction expects exactly one "
                "token sequence"
            )
        prompt_token_count = cls._token_count(
            getattr(client, "_num_original_tokens", None),
            field="_num_original_tokens",
        )
        if prompt_token_count > full_sequence_token_count:
            raise OfficialSpecEdgeConfigurationError(
                "pinned Official SpecEdge _num_original_tokens exceeds generated "
                "_prefix_tokens length"
            )

        tokenizer = getattr(client, "_tokenizer", None)
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            raise OfficialSpecEdgeConfigurationError(
                "pinned Official SpecEdge client tokenizer has no callable decode()"
            )
        try:
            # Keep this call source-equivalent to the upstream completion log.
            full_sequence_text = decode(token_ids, skip_special_tokens=True)
            completion_text = decode(
                token_ids[prompt_token_count:], skip_special_tokens=True
            )
        except Exception as exc:
            raise OfficialSpecEdgeConfigurationError(
                "could not decode pinned Official SpecEdge generated prefix tokens"
            ) from exc
        if not isinstance(full_sequence_text, str) or not isinstance(completion_text, str):
            raise OfficialSpecEdgeConfigurationError(
                "pinned Official SpecEdge tokenizer.decode() must return text"
            )

        generated_token_count = full_sequence_token_count - prompt_token_count
        return {
            # ``text`` is intentionally the token-sliced continuation used by
            # materialize_specedge_completions.py and downstream quality
            # scorers.  The explicit flag prevents prompt-inclusive scoring.
            "text": completion_text,
            "output_includes_prompt": False,
            "generated_token_count": generated_token_count,
            "prompt_token_count": prompt_token_count,
            "full_sequence_token_count": full_sequence_token_count,
            # This is the exact text upstream logs after generate(), retained
            # for source-equivalence auditing rather than for scoring.
            "full_sequence_text": full_sequence_text,
            "full_sequence_includes_prompt": True,
            "completion_extraction": "official_prefix_tokens",
        }

    def create_client(
        self, request: Mapping[str, Any], context: ReplayContext
    ) -> ClientInvoker:
        """Return an invoker that calls known ``SpecExecClient.generate``."""

        self.prepare()
        dataset_index, prompt = self._prompt_for_request(request, context)
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        async def invoke() -> Mapping[str, Any]:
            if self._generation_lock is None:
                self._generation_lock = asyncio.Lock()
            # Official GraphEngine/config are process-global in practice.  The
            # wrapper keeps max-concurrency at one; this lock makes accidental
            # direct use safe rather than letting two clients reset one engine.
            async with self._generation_lock:
                if self._components is None:
                    raise OfficialSpecEdgeConfigurationError("Official runtime was not prepared")
                client = self._components.spec_exec_client_type(
                    engine=self._engine,
                    tokenizer=self._tokenizer,
                    prompt=prompt,
                    max_len=self._components.client_config.max_len,
                )
                await client.generate(dataset_index)
                # The extraction below is deliberately limited to the pinned
                # upstream client fields used by its own post-generate log.
                completion = self._extract_official_completion(client)
            return {
                "official_api": "SpecExecClient.generate",
                "request_id": context.request_id,
                "trace_index": context.trace_index,
                "dataset_index": dataset_index,
                "client_id": self.settings.client_id,
                "grpc_address": self.settings.grpc_address,
                "prompt_sha256": prompt_sha256,
                **completion,
            }

        return invoke


_FACTORY: OfficialSpecEdgeFactory | None = None


def _get_factory() -> OfficialSpecEdgeFactory:
    global _FACTORY
    if _FACTORY is None:
        # Configuration, prompts, and local-log overrides become an immutable
        # process snapshot.  Re-reading YAML/JSON for every scheduled request
        # would contaminate the latency being measured and would let ambient
        # environment changes redirect a live Official client.
        _FACTORY = OfficialSpecEdgeFactory.from_environment()
    return _FACTORY


def prepare() -> None:
    """Optional hook recognized by ``poisson_client`` before timing begins."""

    _get_factory().prepare()


def create_client(request: Mapping[str, Any], context: ReplayContext) -> ClientInvoker:
    """Public explicit factory for ``--client-factory ...:create_client``."""

    return _get_factory().create_client(request, context)


# ``poisson_client.prepare_client_factory`` looks for this opt-in hook on the
# callable, so model initialization and Sync happen before measured replay.
create_client.prepare = prepare  # type: ignore[attr-defined]
create_client.requires_client_id = True  # type: ignore[attr-defined]


__all__ = [
    "EffectiveClientConfig",
    "EnvironmentSettings",
    "OfficialSpecEdgeConfigurationError",
    "OfficialSpecEdgeFactory",
    "create_client",
    "load_effective_client_config",
    "prepare",
    "read_environment_settings",
]
