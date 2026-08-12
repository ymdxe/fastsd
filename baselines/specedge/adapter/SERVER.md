# Official SpecEdge server transport adapter

`server_entrypoint.py` starts the pinned Official SpecEdge **batch** server on
an explicit gRPC address without editing `baselines/specedge/official`.
Official `src/script/batch_server.py` fixes its listener to `[::]:8000`; this
adapter imports the same Official config loader, `SpecExecBatchServer`, and
generated gRPC registration function, then binds the resulting server itself.
It does not construct a model, alter the tree algorithm, or change the pinned
submodule.

## Validate before allocating a GPU

Run this on node2 in the dedicated SpecEdge environment.  The adapter accepts
either an Official batch-server YAML or the repository's unified
`configs/experiments/mtbench_poisson.yaml`; for the latter it validates the
FastSD fields and renders a fresh Official-shaped YAML only for a live run.
This command validates the literal address and port and writes nothing:

```bash
cd /home/hdd/zhangh/workspace/fastsd
source /home/hdd/zhangh/envs/specedge/bin/activate

python baselines/specedge/adapter/server_entrypoint.py \
  --config configs/experiments/mtbench_poisson.yaml \
  --run-id specedge-mtbench-smoke \
  --bind-host 10.66.0.5 \
  --port 18000 \
  --dry-run
```

The adapter accepts only literal IP addresses.  It rejects `0.0.0.0` and `::`;
use `10.66.0.5` for the node2 IB interface or `127.0.0.1` for a local test.
It does not check whether that address is configured locally; the experiment
preflight owns that check.

## Live server command

After the GPU preflight grants an A6000, map that physical GPU to the generated
Official configuration's `cuda:0` using `CUDA_VISIBLE_DEVICES`.  With the
unified config, the adapter maps `bfloat16` to Official's `bf16`, copies the
Qwen3 model paths, uses Official's public `dynamic` batch scheduling, and requires
`specedge.max_batch_size == specedge.num_clients == workload.num_edge_clients`.
The checked-in values remain 2.  `dynamic` lets the pinned server dispatch an
available partial batch when the two independent Poisson clients do not arrive
at exactly the same time; `max_batch_size=2` is still the capacity bound. This
is a documented/accepted Official scheduling configuration, not a change to
its tree-drafting or speculative-decoding algorithm or source code.  It writes
`cache_prefill`, tree budget, beam limits, temperature, and token cap from the
unified config without changing the Official source.  The client/replay address
is `10.66.0.5:18000`.

```bash
cd /home/hdd/zhangh/workspace/fastsd
source /home/hdd/zhangh/envs/specedge/bin/activate

export RUN_ID=specedge-mtbench-l0.8-s20260812
export RUN_ROOT=/home/hdd/zhangh/results/fastsd
export CLOUD_PHYSICAL_GPU=<preflight-selected-a6000-index>

CUDA_VISIBLE_DEVICES="$CLOUD_PHYSICAL_GPU" \
python baselines/specedge/adapter/server_entrypoint.py \
  --official-root baselines/specedge/official \
  --config configs/experiments/mtbench_poisson.yaml \
  --run-id "$RUN_ID" \
  --bind-host 10.66.0.5 \
  --port 18000 \
  --run-dir "$RUN_ROOT/$RUN_ID/cloud" \
  --result-path "$RUN_ROOT/$RUN_ID/cloud" \
  --shutdown-file "$RUN_ROOT/$RUN_ID/cloud/control/graceful-shutdown.json" \
  --target-model /home/hdd/zhangh/models/Qwen3-8B \
  --draft-model /home/hdd/zhangh/models/Qwen3-0.6B \
  --cloud-physical-gpu "$CLOUD_PHYSICAL_GPU"
```

For a live run the adapter creates exactly one immutable copy at
`<run-dir>/config/specedge-server-effective.yaml` using exclusive creation.
It never modifies the source YAML or any file under `official`; if that target
already exists it fails and requires a new run ID.  `--result-path` changes
only `base.result_path` in this copied YAML so Official log/result artifacts
remain inside the run directory.  Omit it to preserve the source config value.

The process remains in the foreground.  `Ctrl-C`/`SIGTERM` stops the gRPC
listener and invokes Official's own controller cleanup.  SpecEdge exposes gRPC,
not an HTTP `/health` endpoint; use the adapter's `server_started` JSON event
or a gRPC channel readiness check rather than `curl /health`.

The adapter also creates one fresh, append-only server event artifact at
`<run-dir>/metrics/specedge_server_events.jsonl` by default (override it with
absolute `--server-events-output` only under `<run-dir>`).  It wraps the two
generated gRPC service methods in the adapter layer, without changing the
Official controller.  Each line records `server_monotonic_s`, `phase` (`enter`
or `leave`), `method` (`Validate` or `Sync`), a local `rpc_id`, outcome, and
only safe scalar request identity fields (`client_idx`, `req_idx`, `prefill`)
when present. It never writes prompt text or token tensors.  For the formal
measurement window, finalization should use the first `Validate` enter and last
`Validate` leave; `Sync` is the pre-measurement readiness handshake.

## Auditable graceful shutdown

`--shutdown-file` is optional, but when supplied it must be an absolute path
and `--run-id` is required.  The marker must not exist when the live server is
started; the adapter never creates, truncates, replaces, or deletes it.  This
rejects stale control files before CUDA/Official startup.

After `server_started`, the adapter asynchronously polls that JSON path.  A
wrapper may exclusively create a marker such as:

```json
{"schema_version":1,"event":"graceful_shutdown","run_id":"specedge-mtbench-l0.8-s20260812"}
```

Only an object whose `run_id` exactly matches the live `--run-id` triggers
shutdown.  A partial, malformed, symlinked, or different-run marker is ignored
and cannot stop the server.  On match the adapter emits
`shutdown_marker_matched`, sets the same shutdown event used by SIGTERM, then
runs `server.stop(grace=2.0)` and Official controller cleanup before emitting
`server_stopped`.  The wrapper should retain this marker and the server log in
the run artifacts for provenance; it must not reuse the marker for another
run.

## Failure boundaries

- A missing pinned submodule, wrong `script.batch_server`, missing Official
  dependencies, or incompatible upstream interface fails before an effective
  config is written where possible, with an activation/configuration message.
- The adapter does not auto-select GPUs, probe or kill other processes, open a
  wildcard listener, or infer an Official client/model API.
- The Poisson client adapter remains separate.  Its explicit live factory must
  target `10.66.0.5:18000` and be verified in the same pinned environment.
- A formal replay with any request error or cancellation exits non-zero from
  the client adapter after preserving its request/summary artifacts.  The
  two-client launcher already propagates that child failure and skips merging,
  so a failed replay cannot be finalized as an analysis input.
