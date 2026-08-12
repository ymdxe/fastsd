#!/usr/bin/env python3
"""Transport-only entry point for the pinned Official SpecEdge batch server.

Official SpecEdge's ``src/script/batch_server.py`` always binds ``[::]:8000``.
This wrapper keeps that source untouched: it reuses the official config loader,
``SpecExecBatchServer`` controller, generated gRPC registration function, and
shutdown semantics, but supplies an explicit, non-wildcard bind address.

The module imports neither torch, grpc, nor the Official submodule until a live
run is requested.  ``--dry-run`` therefore validates the transport and YAML
contract without allocating a GPU or creating any run artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib
import inspect
import ipaddress
import json
import os
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_PORT = 18000
EFFECTIVE_CONFIG_NAME = "specedge-server-effective.yaml"
SERVER_EVENTS_FILE_NAME = "specedge_server_events.jsonl"
# Pinned Official ``InferenceController._check_batch_condition`` explicitly
# supports ``dynamic`` and ``static``.  A Poisson trace cannot assume that two
# independent edge clients reach every batch boundary simultaneously, so the
# unified experiment renders the former.  This is an exposed Official batch
# scheduling setting, not a modification of its tree/speculation algorithm.
OFFICIAL_POISSON_BATCH_TYPE = "dynamic"
SHUTDOWN_FILE_POLL_INTERVAL_S = 0.25


class ServerEntrypointError(RuntimeError):
    """Raised when the adapter cannot safely start an Official server."""


@dataclass(frozen=True)
class ValidatedConfig:
    """A parsed Official batch-server configuration and source provenance."""

    path: Path
    document: dict[str, Any]
    sha256: str
    source_format: str


@dataclass(frozen=True)
class OfficialBindings:
    """Only the public/used pieces of the pinned Official batch server."""

    batch_server: ModuleType
    controller_type: type
    grpc_aio: Any
    grpc_service: ModuleType


class ServerGrpcEventWriter:
    """Append-only server-side gRPC timing events owned by the adapter.

    This writer deliberately captures only method names, monotonic timestamps,
    and small scalar request identifiers.  It never serializes request bodies,
    prompts, token tensors, metadata, or gRPC contexts.
    """

    def __init__(self, path: Path, handle: Any, *, run_id: str | None) -> None:
        self.path = path
        self._handle = handle
        self._run_id = run_id
        self._next_rpc_id = 0

    @classmethod
    def open_new(cls, path: str | Path, *, run_id: str | None) -> "ServerGrpcEventWriter":
        output = Path(path)
        if output.exists() or output.is_symlink():
            raise ServerEntrypointError(
                "Refusing to overwrite existing server gRPC event artifact: "
                f"{output}"
            )
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            handle = output.open("x", encoding="utf-8", newline="\n")
        except FileExistsError as exc:
            raise ServerEntrypointError(
                "Refusing to overwrite existing server gRPC event artifact: "
                f"{output}"
            ) from exc
        except OSError as exc:
            raise ServerEntrypointError(
                f"Cannot create server gRPC event artifact {output}: {exc}"
            ) from exc
        return cls(output, handle, run_id=run_id)

    def next_rpc_id(self) -> int:
        rpc_id = self._next_rpc_id
        self._next_rpc_id += 1
        return rpc_id

    def write(
        self,
        *,
        phase: str,
        method: str,
        rpc_id: int,
        request_identity: Mapping[str, Any] | None,
        outcome: str | None = None,
        error_type: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "record_type": "grpc_request",
            "phase": phase,
            "server_monotonic_s": time.monotonic(),
            "method": method,
            "rpc_id": rpc_id,
            "run_id": self._run_id,
        }
        if request_identity:
            payload["request_identity"] = dict(request_identity)
        if outcome is not None:
            payload["outcome"] = outcome
        if error_type is not None:
            payload["error_type"] = error_type
        self._handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def _safe_request_identity(request: Any) -> dict[str, Any]:
    """Extract only stable scalar IDs exposed by Official ValidateRequest."""

    identity: dict[str, Any] = {}
    for name in ("client_idx", "req_idx"):
        value = getattr(request, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            identity[name] = value
    prefill = getattr(request, "prefill", None)
    if isinstance(prefill, bool):
        identity["prefill"] = prefill
    return identity


class EventRecordingSpecEdgeService:
    """Transparent adapter-owned proxy around the two pinned gRPC methods."""

    def __init__(self, controller: Any, event_writer: ServerGrpcEventWriter) -> None:
        self._controller = controller
        self._event_writer = event_writer

    async def _call(self, method: str, request: Any, context: Any) -> Any:
        rpc_id = self._event_writer.next_rpc_id()
        request_identity = _safe_request_identity(request)
        self._event_writer.write(
            phase="enter",
            method=method,
            rpc_id=rpc_id,
            request_identity=request_identity,
        )
        try:
            result = getattr(self._controller, method)(request, context)
            if inspect.isawaitable(result):
                result = await result
        except BaseException as exc:
            self._event_writer.write(
                phase="leave",
                method=method,
                rpc_id=rpc_id,
                request_identity=request_identity,
                outcome="error",
                error_type=type(exc).__name__,
            )
            raise
        self._event_writer.write(
            phase="leave",
            method=method,
            rpc_id=rpc_id,
            request_identity=request_identity,
            outcome="completed",
        )
        return result

    async def Validate(self, request: Any, context: Any) -> Any:
        return await self._call("Validate", request, context)

    async def Sync(self, request: Any, context: Any) -> Any:
        return await self._call("Sync", request, context)


# These are the exact dictionary reads in Official
# ``src/script/batch_server.py::_load_config``.  Validating them up front gives
# a readable error before importing CUDA/grpc or writing a run artifact.
_REQUIRED_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "base": ("result_path", "exp_name", "seed", "max_len", "dtype"),
    "server": (
        "target_model",
        "device",
        "temperature",
        "max_batch_size",
        "num_clients",
        "batch_type",
        "cache_prefill",
    ),
    "client": (
        "dataset",
        "sample_req_cnt",
        "req_offset",
        "max_n_beams",
        "max_budget",
    ),
}

_UNIFIED_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "experiment": ("name", "dataset", "run_root"),
    "models": ("draft_model", "target_model", "dtype"),
    "decoding": ("max_new_tokens", "temperature"),
    "workload": ("num_edge_clients", "max_requests"),
    "specedge": (
        "tree_budget",
        "max_beam_len",
        "max_branch_width",
        "max_batch_size",
        "num_clients",
        "cache_prefill",
    ),
}

_UNIFIED_TO_OFFICIAL_DTYPE = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "bf16": "bf16",
    "fp16": "fp16",
    "fp32": "fp32",
}

# FastSD names the MT-Bench workload ``mt_bench`` to match its local JSONL.
# The pinned Official SpecEdge package names the corresponding prompt artifact
# ``data/mtbench_prompts.json``.  Translate only at this adapter boundary.
_UNIFIED_TO_OFFICIAL_DATASET = {
    "mt_bench": "mtbench",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml_module() -> Any:
    try:
        return importlib.import_module("yaml")
    except ModuleNotFoundError as exc:
        raise ServerEntrypointError(
            "PyYAML is required to validate the Official SpecEdge YAML config. "
            "Activate the dedicated SpecEdge environment and install the pinned "
            "dependencies before starting the server."
        ) from exc


def validate_bind_host(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Accept an explicit literal IP while refusing network-wide wildcards."""

    try:
        host = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ServerEntrypointError(
            "--bind-host must be a literal local IP address, for example "
            "10.66.0.5 (IB) or 127.0.0.1 (local smoke test)."
        ) from exc

    if host.is_unspecified:
        raise ServerEntrypointError(
            "Refusing wildcard bind host. Use the node2 IB address 10.66.0.5 "
            "for a cloud-edge run, or 127.0.0.1 for a local smoke test."
        )
    if host.is_multicast:
        raise ServerEntrypointError("--bind-host must not be a multicast address")
    return host


def validate_port(value: int | str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ServerEntrypointError("--port must be an integer in the range 1..65535") from exc
    if not 1 <= port <= 65535:
        raise ServerEntrypointError("--port must be in the range 1..65535")
    return port


def resolve_shutdown_file(value: str | Path) -> Path:
    """Validate an external graceful-shutdown marker without creating it."""

    raw_path = Path(value).expanduser()
    if not raw_path.is_absolute():
        raise ServerEntrypointError(
            "--shutdown-file must be an absolute path owned by this run's control "
            "directory"
        )
    # Do not resolve here: callers must still be able to detect a broken
    # symlink at this exact control path before beginning a live run.
    return raw_path


def assert_new_shutdown_file(path: str | Path) -> Path:
    """Fail closed on a stale marker; the server never writes this path."""

    resolved = resolve_shutdown_file(path)
    if resolved.exists() or resolved.is_symlink():
        raise ServerEntrypointError(
            "--shutdown-file must not already exist for a live run; refusing to "
            f"reuse a stale graceful-shutdown marker: {resolved}"
        )
    return resolved


def shutdown_marker_matches(path: str | Path, *, run_id: str) -> bool:
    """Return whether a completed external JSON marker names this live run.

    A controller can observe a marker between exclusive creation and close, so
    malformed/partial JSON is deliberately treated as not-ready and retried by
    the async poller.  A symlink is never followed as a control marker.
    """

    marker_path = Path(path)
    if marker_path.is_symlink() or not marker_path.is_file():
        return False
    try:
        document = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(document, Mapping) and document.get("run_id") == run_id


async def watch_shutdown_file(
    shutdown_event: asyncio.Event,
    *,
    shutdown_file: str | Path,
    run_id: str,
    poll_interval_s: float = SHUTDOWN_FILE_POLL_INTERVAL_S,
) -> bool:
    """Poll a wrapper-created marker and trigger graceful shutdown on match."""

    if poll_interval_s <= 0:
        raise ServerEntrypointError("shutdown-file poll interval must be > 0")
    marker_path = resolve_shutdown_file(shutdown_file)
    while not shutdown_event.is_set():
        if shutdown_marker_matches(marker_path, run_id=run_id):
            print(
                json.dumps(
                    {
                        "event": "shutdown_marker_matched",
                        "run_id": run_id,
                        "shutdown_file": str(marker_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            shutdown_event.set()
            return True
        await asyncio.sleep(poll_interval_s)
    return False


def format_bind_address(
    host: ipaddress.IPv4Address | ipaddress.IPv6Address, port: int
) -> str:
    """Return the gRPC address spelling needed for IPv4 and IPv6 literals."""

    host_text = str(host)
    return f"[{host_text}]:{port}" if host.version == 6 else f"{host_text}:{port}"


def _require_sections(document: Mapping[str, Any], required: Mapping[str, Sequence[str]]) -> None:
    for section, keys in required.items():
        section_value = document.get(section)
        if not isinstance(section_value, dict):
            raise ServerEntrypointError(
                f"SpecEdge config is missing mapping section {section!r}"
            )
        missing = [key for key in keys if key not in section_value]
        if missing:
            joined = ", ".join(missing)
            raise ServerEntrypointError(f"SpecEdge config section {section!r} is missing: {joined}")


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ServerEntrypointError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ServerEntrypointError(f"{field} must be a positive integer") from exc
    if result <= 0:
        raise ServerEntrypointError(f"{field} must be a positive integer")
    return result


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ServerEntrypointError(f"{field} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ServerEntrypointError(f"{field} must be a non-negative integer") from exc
    if result < 0:
        raise ServerEntrypointError(f"{field} must be a non-negative integer")
    return result


def _render_official_from_unified(
    document: Mapping[str, Any],
    *,
    run_id: str | None,
    target_model: str | None,
    draft_model: str | None,
    client_host: str,
) -> dict[str, Any]:
    """Translate only configuration fields, never an Official model/client API.

    The resulting mapping is the documented input shape of Official
    ``script.batch_server.py`` plus the ordinary client fields needed by an
    explicit replay factory.  The pinned Official controller supports
    ``batch_type: dynamic``: it can dispatch an available partial batch rather
    than waiting for every one of two independently scheduled Poisson clients.
    ``max_batch_size`` remains the capacity limit and is deliberately kept
    equal to the two configured clients for this experiment.  Changing this
    public Official scheduler field does not alter its tree/speculation
    algorithm or source code.
    """

    _require_sections(document, _UNIFIED_REQUIRED_KEYS)
    experiment = document["experiment"]
    models = document["models"]
    decoding = document["decoding"]
    workload = document["workload"]
    specedge = document["specedge"]

    dtype_input = str(models["dtype"]).lower()
    dtype = _UNIFIED_TO_OFFICIAL_DTYPE.get(dtype_input)
    if dtype is None:
        accepted = ", ".join(sorted(_UNIFIED_TO_OFFICIAL_DTYPE))
        raise ServerEntrypointError(
            f"models.dtype={models['dtype']!r} cannot be mapped to Official SpecEdge; "
            f"accepted values: {accepted}"
        )

    max_batch_size = _positive_int(specedge["max_batch_size"], field="specedge.max_batch_size")
    num_clients = _positive_int(specedge["num_clients"], field="specedge.num_clients")
    workload_clients = _positive_int(
        workload["num_edge_clients"], field="workload.num_edge_clients"
    )
    if num_clients != workload_clients:
        raise ServerEntrypointError(
            "specedge.num_clients must equal workload.num_edge_clients so every "
            "Poisson replay client has one Official batch-server slot"
        )
    if max_batch_size != num_clients:
        raise ServerEntrypointError(
            "specedge.max_batch_size must equal specedge.num_clients for the "
            "two-client Official Poisson experiment"
        )

    resolved_target = str(target_model or models["target_model"])
    resolved_draft = str(draft_model or models["draft_model"])
    if not resolved_target or not resolved_draft:
        raise ServerEntrypointError("Both a Qwen3 target model and draft model are required")

    return {
        "version": 1,
        # Official's SpecExecClient reads this when a reviewed replay factory
        # imports it. It is not consumed by the batch-server controller itself.
        "opt": 2,
        "base": {
            "result_path": str(experiment["run_root"]),
            "exp_name": str(run_id or experiment["name"]),
            "dtype": dtype,
            "seed": 42,
            "max_len": 2048,
        },
        "server": {
            "process_name": "server",
            "target_model": resolved_target,
            # CUDA_VISIBLE_DEVICES maps the selected physical A6000 here.
            "device": "cuda:0",
            "temperature": float(decoding["temperature"]),
            "max_batch_size": max_batch_size,
            "num_clients": num_clients,
            "batch_type": OFFICIAL_POISSON_BATCH_TYPE,
            "cache_prefill": bool(specedge["cache_prefill"]),
        },
        "client": {
            "host": client_host,
            "process_name": "client",
            "draft_model": resolved_draft,
            # Official constructs ``data/{dataset}_prompts.json``.  Passing
            # FastSD's ``mt_bench`` through would look for a nonexistent
            # ``mt_bench_prompts.json`` instead of the pinned mtbench file.
            "dataset": _UNIFIED_TO_OFFICIAL_DATASET.get(
                str(experiment["dataset"]), str(experiment["dataset"])
            ),
            "reasoning": False,
            "sample_req_cnt": 1,
            "req_offset": 0,
            "max_n_beams": _positive_int(specedge["tree_budget"], field="specedge.tree_budget"),
            "max_beam_len": _positive_int(specedge["max_beam_len"], field="specedge.max_beam_len"),
            "max_branch_width": _positive_int(
                specedge["max_branch_width"], field="specedge.max_branch_width"
            ),
            "max_budget": _positive_int(specedge["tree_budget"], field="specedge.tree_budget"),
            "proactive": {
                "type": "excluded",
                "max_n_beams": _positive_int(
                    specedge["tree_budget"], field="specedge.tree_budget"
                ),
                "max_beam_len": _positive_int(
                    specedge["max_beam_len"], field="specedge.max_beam_len"
                ),
                "max_branch_width": _positive_int(
                    specedge["max_branch_width"], field="specedge.max_branch_width"
                ),
                "max_budget": _positive_int(
                    specedge["tree_budget"], field="specedge.tree_budget"
                ),
            },
            "max_new_tokens": _positive_int(
                decoding["max_new_tokens"], field="decoding.max_new_tokens"
            ),
            "max_request_num": _positive_int(
                workload["max_requests"], field="workload.max_requests"
            ),
        },
    }


def load_and_validate_config(
    config_path: str | Path,
    *,
    run_id: str | None = None,
    target_model: str | None = None,
    draft_model: str | None = None,
    client_host: str = "127.0.0.1:18000",
) -> ValidatedConfig:
    """Load an Official YAML or render an Official YAML from the unified config."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ServerEntrypointError(f"Official SpecEdge config does not exist: {path}")

    yaml = _load_yaml_module()
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except Exception as exc:
        raise ServerEntrypointError(f"Cannot parse Official SpecEdge config {path}: {exc}") from exc

    if not isinstance(document, dict):
        raise ServerEntrypointError(
            "Official SpecEdge config must contain a top-level YAML mapping"
        )

    if all(section in document for section in _REQUIRED_CONFIG_KEYS):
        _require_sections(document, _REQUIRED_CONFIG_KEYS)
        return ValidatedConfig(
            path=path,
            document=document,
            sha256=_sha256_file(path),
            source_format="official",
        )

    if all(section in document for section in _UNIFIED_REQUIRED_KEYS):
        return ValidatedConfig(
            path=path,
            document=_render_official_from_unified(
                document,
                run_id=run_id,
                target_model=target_model,
                draft_model=draft_model,
                client_host=client_host,
            ),
            sha256=_sha256_file(path),
            source_format="fastsd-unified",
        )

    raise ServerEntrypointError(
        "Config is neither an Official SpecEdge batch-server YAML nor the FastSD "
        "unified experiment YAML required by this adapter."
    )


def _resolve_existing_directory(value: str | Path, *, argument: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ServerEntrypointError(f"{argument} is not a directory: {path}")
    return path


def _resolve_output_directory(value: str | Path, *, argument: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_absolute():  # ``resolve`` makes this defensive check explicit.
        raise ServerEntrypointError(f"{argument} must be an absolute path")
    return path


def write_effective_config(
    config: ValidatedConfig,
    *,
    run_dir: str | Path,
    result_path: str | Path | None = None,
) -> Path:
    """Write a fresh runtime copy only under the caller's run directory.

    This does not touch the Official submodule or the supplied source YAML.
    ``--result-path`` is intentionally opt-in; without it the copy preserves
    every Official setting byte-for-data value (apart from YAML formatting).
    """

    resolved_run_dir = _resolve_output_directory(run_dir, argument="--run-dir")
    document = copy.deepcopy(config.document)
    if result_path is not None:
        document["base"]["result_path"] = str(
            _resolve_output_directory(result_path, argument="--result-path")
        )

    target_directory = resolved_run_dir / "config"
    try:
        target_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ServerEntrypointError(
            f"Cannot create the effective-config directory {target_directory}: {exc}"
        ) from exc

    target = target_directory / EFFECTIVE_CONFIG_NAME
    if target.exists():
        raise ServerEntrypointError(
            f"Refusing to overwrite existing effective config: {target}. "
            "Use a new run ID/run directory."
        )

    yaml = _load_yaml_module()
    try:
        rendered = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
        # Exclusive creation is intentional: a concurrent retry must never
        # silently replace another run's immutable configuration snapshot.
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    except FileExistsError as exc:
        raise ServerEntrypointError(
            f"Refusing to overwrite existing effective config: {target}. "
            "Use a new run ID/run directory."
        ) from exc
    except OSError as exc:
        raise ServerEntrypointError(f"Cannot write effective config {target}: {exc}") from exc
    return target


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_server_events_output(
    value: str | Path | None, *, run_dir: str | Path
) -> Path:
    """Resolve the adapter event artifact under its explicit component run."""

    raw_run_dir = Path(run_dir).expanduser()
    if not raw_run_dir.is_absolute():
        raise ServerEntrypointError("--run-dir must be an absolute path for server events")
    output = (
        raw_run_dir / "metrics" / SERVER_EVENTS_FILE_NAME
        if value is None
        else Path(value).expanduser()
    )
    if not output.is_absolute():
        raise ServerEntrypointError("--server-events-output must be an absolute path")
    if not _path_is_within(output, raw_run_dir):
        raise ServerEntrypointError(
            "--server-events-output must stay under --run-dir so event provenance "
            "cannot be redirected outside this component run"
        )
    return output


def assert_new_server_events_output(path: str | Path) -> Path:
    """Reserve a fresh path logically; writer later creates it exclusively."""

    output = Path(path)
    if output.exists() or output.is_symlink():
        raise ServerEntrypointError(
            "Refusing to overwrite existing server gRPC event artifact: "
            f"{output}"
        )
    return output


def _ensure_no_conflicting_official_imports(source_root: Path) -> None:
    """Avoid accidentally delegating to a different checkout's top-level modules."""

    for name in ("log", "util", "config"):
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        if module_file and not _path_is_within(Path(module_file), source_root):
            raise ServerEntrypointError(
                f"Cannot safely load Official SpecEdge: module {name!r} is already "
                f"imported from {module_file}, not {source_root}. Start a fresh Python "
                "process in the dedicated SpecEdge environment."
            )


def load_official_bindings(official_root: str | Path) -> OfficialBindings:
    """Load the pinned official server pieces lazily and verify their origin."""

    root = _resolve_existing_directory(official_root, argument="--official-root")
    source_root = root / "src"
    batch_server_path = source_root / "script" / "batch_server.py"
    if not batch_server_path.is_file():
        raise ServerEntrypointError(
            "Official SpecEdge batch server was not found at "
            f"{batch_server_path}. Initialize the pinned submodule before a live run."
        )

    _ensure_no_conflicting_official_imports(source_root)
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    try:
        batch_server = importlib.import_module("script.batch_server")
    except Exception as exc:
        raise ServerEntrypointError(
            "Could not import the pinned Official SpecEdge batch server. Activate the "
            "dedicated SpecEdge environment with its grpc/torch/transformers dependencies, "
            "then retry. The adapter did not change Official source."
        ) from exc

    module_file = getattr(batch_server, "__file__", None)
    if not module_file or Path(module_file).resolve() != batch_server_path.resolve():
        raise ServerEntrypointError(
            "The imported 'script.batch_server' is not the requested pinned Official "
            f"module ({module_file!r} != {batch_server_path}). Start a fresh process."
        )

    try:
        return OfficialBindings(
            batch_server=batch_server,
            controller_type=batch_server.SpecExecBatchServer,
            grpc_aio=batch_server.grpc.aio,
            grpc_service=batch_server.specedge_pb2_grpc,
        )
    except AttributeError as exc:
        raise ServerEntrypointError(
            "The pinned Official batch-server interface is not compatible with this "
            "transport adapter. Expected SpecExecBatchServer and generated gRPC service "
            "registration symbols are unavailable; do not edit Official source."
        ) from exc


def _install_shutdown_handlers(shutdown_event: asyncio.Event) -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def request_shutdown(_signum: int, _frame: Any) -> None:
        shutdown_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_shutdown)
        except (ValueError, OSError):
            # Signal handlers are only available in the main thread; the live
            # CLI runs there, while tests may invoke this coroutine differently.
            continue
    return previous


def _restore_shutdown_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (ValueError, OSError):
            continue


async def serve_official(
    bindings: OfficialBindings,
    *,
    bind_address: str,
    shutdown_file: str | Path | None = None,
    run_id: str | None = None,
    server_events_output: str | Path | None = None,
) -> None:
    """Run Official's controller with an explicit gRPC transport address."""

    if shutdown_file is not None and run_id is None:
        raise ServerEntrypointError(
            "run_id is required with shutdown_file for graceful shutdown"
        )
    shutdown_event = asyncio.Event()
    controller: Any = None
    server: Any = None
    event_writer: ServerGrpcEventWriter | None = None
    shutdown_watcher: asyncio.Task[bool] | None = None
    handlers: dict[int, Any] = {}
    started = False
    cleanup_completed = False
    try:
        if server_events_output is not None:
            event_writer = ServerGrpcEventWriter.open_new(
                server_events_output, run_id=run_id
            )
        controller = bindings.controller_type(shutdown_event=shutdown_event)
        server = bindings.grpc_aio.server()
        service = (
            EventRecordingSpecEdgeService(controller, event_writer)
            if event_writer is not None
            else controller
        )
        bindings.grpc_service.add_SpecEdgeServiceServicer_to_server(service, server)
        bound_port = server.add_insecure_port(bind_address)
        if bound_port == 0:
            raise ServerEntrypointError(
                f"gRPC could not bind {bind_address}. Check the selected IB address, "
                "exclusive port ownership, and node firewall."
            )

        handlers = _install_shutdown_handlers(shutdown_event)
        await server.start()
        started = True
        print(json.dumps({"event": "server_started", "bind_address": bind_address}), flush=True)
        if shutdown_file is not None and run_id is not None:
            shutdown_watcher = asyncio.create_task(
                watch_shutdown_file(
                    shutdown_event,
                    shutdown_file=shutdown_file,
                    run_id=run_id,
                ),
                name="specedge-shutdown-file-watcher",
            )
        await shutdown_event.wait()
    finally:
        try:
            _restore_shutdown_handlers(handlers)
            if shutdown_watcher is not None:
                shutdown_watcher.cancel()
                await asyncio.gather(shutdown_watcher, return_exceptions=True)
            if server is not None:
                await server.stop(grace=2.0 if started else 0.0)
            if controller is not None:
                await controller.cleanup()
            cleanup_completed = True
        finally:
            if event_writer is not None:
                event_writer.close()
            if started:
                print(
                    json.dumps(
                        {
                            "event": "server_stopped",
                            "cleanup_completed": cleanup_completed,
                            "shutdown_file": str(shutdown_file)
                            if shutdown_file is not None
                            else None,
                            "server_events_output": str(event_writer.path)
                            if event_writer is not None
                            else None,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )


def run_official_server(
    bindings: OfficialBindings,
    *,
    effective_config: Path,
    official_root: Path,
    bind_address: str,
    shutdown_file: Path | None = None,
    run_id: str | None = None,
    server_events_output: Path | None = None,
) -> None:
    """Invoke the same Official config loader and seed setup as batch_server.py."""

    # Official's shell wrapper changes into the submodule root before it loads
    # config. Preserve that relative-path behavior without modifying the source.
    os.chdir(official_root)
    try:
        bindings.batch_server._load_config(effective_config)
        bindings.batch_server.util.set_seed(bindings.batch_server.config.seed)
    except Exception as exc:
        raise ServerEntrypointError(
            "Official SpecEdge rejected the effective config before service startup. "
            f"Inspect {effective_config}; no Official source was modified."
        ) from exc
    asyncio.run(
        serve_official(
            bindings,
            bind_address=bind_address,
            shutdown_file=shutdown_file,
            run_id=run_id,
            server_events_output=server_events_output,
        )
    )


def _default_official_root() -> Path:
    return Path(__file__).resolve().parents[1] / "official"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned Official SpecEdge batch server on an explicit, "
            "non-wildcard gRPC bind address."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Official SpecEdge batch-server YAML or the FastSD unified experiment YAML "
            "to render into a runtime-only Official config"
        ),
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Run identifier recorded as exp_name when --config is a FastSD unified YAML"
        ),
    )
    parser.add_argument(
        "--bind-host",
        required=True,
        help="Literal local IP only; use 10.66.0.5 for the node2 IB experiment",
    )
    parser.add_argument(
        "--port", default=DEFAULT_PORT, help=f"gRPC port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--run-dir",
        help=(
            "Fresh run directory where config/specedge-server-effective.yaml is created; "
            "required for a live run"
        ),
    )
    parser.add_argument(
        "--result-path",
        help=(
            "Optional absolute replacement for base.result_path in only the effective "
            "runtime copy"
        ),
    )
    parser.add_argument(
        "--target-model",
        help="Optional target-model override used only when rendering a unified YAML",
    )
    parser.add_argument(
        "--draft-model",
        help="Optional draft-model provenance/client override used only for a unified YAML",
    )
    parser.add_argument(
        "--cloud-physical-gpu",
        help=(
            "Physical A6000 index recorded/validated by the launcher; CUDA_VISIBLE_DEVICES "
            "must map it to the effective config's cuda:0"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help=(
            "Optional positive generation cap for a smoke/quality run. It changes "
            "only the fresh effective config, never the unified source YAML or "
            "Official source."
        ),
    )
    parser.add_argument(
        "--shutdown-file",
        help=(
            "Optional absolute, wrapper-created JSON marker. A matching run_id "
            "requests graceful server shutdown; it must not exist before a live run."
        ),
    )
    parser.add_argument(
        "--server-events-output",
        help=(
            "Absolute new JSONL path under --run-dir for adapter-recorded server "
            "gRPC enter/leave events (default: <run-dir>/metrics/"
            f"{SERVER_EVENTS_FILE_NAME})."
        ),
    )
    parser.add_argument(
        "--official-root",
        default=str(_default_official_root()),
        help="Pinned Official SpecEdge submodule root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate YAML and transport inputs only; do not import Official/CUDA or write files"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        host = validate_bind_host(args.bind_host)
        port = validate_port(args.port)
        bind_address = format_bind_address(host, port)
        shutdown_file: Path | None = None
        if args.shutdown_file is not None:
            if not isinstance(args.run_id, str) or not args.run_id.strip():
                raise ServerEntrypointError(
                    "--run-id is required with --shutdown-file so a control marker "
                    "cannot stop a different run"
                )
            shutdown_file = resolve_shutdown_file(args.shutdown_file)
        if args.server_events_output is not None and not args.run_dir:
            raise ServerEntrypointError(
                "--server-events-output requires --run-dir so it remains inside "
                "the component run"
            )
        server_events_output: Path | None = None
        if args.run_dir:
            server_events_output = resolve_server_events_output(
                args.server_events_output, run_dir=args.run_dir
            )
        if args.cloud_physical_gpu is not None:
            _nonnegative_int(args.cloud_physical_gpu, field="--cloud-physical-gpu")
        if args.max_new_tokens is not None:
            _positive_int(args.max_new_tokens, field="--max-new-tokens")
        config = load_and_validate_config(
            args.config,
            run_id=args.run_id,
            target_model=args.target_model,
            draft_model=args.draft_model,
            client_host=bind_address,
        )
        if args.max_new_tokens is not None:
            # ``ValidatedConfig`` carries a run-local rendered dictionary.  A
            # controlled smoke cap is therefore visible in the immutable
            # effective YAML but leaves both source config trees untouched.
            config.document["client"]["max_new_tokens"] = int(args.max_new_tokens)

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "mode": "dry-run",
                        "official_config": str(config.path),
                        "official_config_sha256": config.sha256,
                        "config_source_format": config.source_format,
                        "bind_address": bind_address,
                        "shutdown_file": str(shutdown_file)
                        if shutdown_file is not None
                        else None,
                        "server_events_output": str(server_events_output)
                        if server_events_output is not None
                        else None,
                        "max_new_tokens": args.max_new_tokens,
                        "would_write_effective_config": bool(args.run_dir),
                    },
                    sort_keys=True,
                )
            )
            return 0

        if not args.run_dir:
            raise ServerEntrypointError("--run-dir is required for a live server run")
        if shutdown_file is not None:
            # The server only polls this external control path.  Verify before
            # loading CUDA/Official code that no stale marker can terminate a
            # newly started run, and do not create/touch it ourselves.
            shutdown_file = assert_new_shutdown_file(shutdown_file)
        if server_events_output is None:
            raise ServerEntrypointError(
                "live server requires --run-dir to create its server gRPC event artifact"
            )
        server_events_output = assert_new_server_events_output(server_events_output)

        official_root = _resolve_existing_directory(
            args.official_root, argument="--official-root"
        )
        # Import first so a missing/incompatible dedicated environment does not
        # leave an incomplete run directory behind.
        bindings = load_official_bindings(official_root)
        effective_config = write_effective_config(
            config, run_dir=args.run_dir, result_path=args.result_path
        )
        print(
            json.dumps(
                {
                    "event": "effective_config_created",
                    "effective_config": str(effective_config),
                    "source_config_sha256": config.sha256,
                    "bind_address": bind_address,
                    "server_events_output": str(server_events_output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        run_official_server(
            bindings,
            effective_config=effective_config,
            official_root=official_root,
            bind_address=bind_address,
            shutdown_file=shutdown_file,
            run_id=args.run_id,
            server_events_output=server_events_output,
        )
        return 0
    except ServerEntrypointError as exc:
        print(f"SpecEdge server adapter error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
