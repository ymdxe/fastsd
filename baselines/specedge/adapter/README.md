# Official SpecEdge Poisson replay adapter

`poisson_client.py` replays a shared, pre-generated JSONL arrival trace without
modifying or importing the pinned `official` submodule.  It is deliberately
standard-library-only so the trace, timing behaviour, and experiment artifacts
can be checked before the separate Official SpecEdge environment is installed.

## Trace contract

Each non-empty JSONL line must be an object with a non-negative,
non-decreasing `scheduled_offset_s`.  `request_id` is recommended and must be
unique when present.  Any other fields, such as `prompt`, `dataset_index`, and
`arrival_index`, are kept intact and passed to the explicit client factory.

```json
{"request_id":"mtbench-000","arrival_index":0,"scheduled_offset_s":0.0,"prompt":"..."}
{"request_id":"mtbench-001","arrival_index":1,"scheduled_offset_s":0.037,"prompt":"..."}
```

The adapter uses `time.monotonic()` and computes every deadline as
`run_start + scheduled_offset_s`.  It records ingress arrival before the
concurrency semaphore and dispatch after it.  A busy Draft client therefore
appears as `queue_wait_s`; it does not delay the next Poisson deadline.

## Safe smoke test

This checks trace parsing, absolute timing, unique-output protection, and
artifact creation without importing or calling Official SpecEdge:

```bash
python baselines/specedge/adapter/poisson_client.py \
  --trace /home/hdd/zhangh/results/fastsd/workloads/mtbench80/arrival_trace.jsonl \
  --output /home/hdd/zhangh/results/fastsd/smoke/metrics/requests.jsonl \
  --run-id specedge-smoke-001 \
  --max-concurrency 2 \
  --dry-run
```

The metrics path and its sibling `requests.summary.json` must be new.  The
adapter never removes or replaces existing artifacts.

## Live Official SpecEdge factory

The adapter includes one explicit factory for the pinned upstream client:
`baselines.specedge.adapter.official_client_factory:create_client`.  Its module
is import-safe on a machine without CUDA, gRPC, PyYAML, or Official SpecEdge;
those dependencies are loaded only when replay preparation begins.

There is intentionally no endpoint discovery or fallback.  Before a real run,
set every one of these values explicitly:

```bash
export SPECEDGE_OFFICIAL_ROOT=/home/hdd/zhangh/workspace/fastsd/baselines/specedge/official
export SPECEDGE_EFFECTIVE_CONFIG=/home/hdd/zhangh/results/fastsd/RUN/edge/config/official_client.yaml
export SPECEDGE_PROMPTS_PATH=/home/hdd/zhangh/results/fastsd/workloads/TRACE/prompts.jsonl
export SPECEDGE_GRPC_ADDRESS=10.66.0.5:18000
export SPECEDGE_CLIENT_ID=0
export CUDA_VISIBLE_DEVICES=<one-approved-physical-A5000>
export SPECEDGE_DEVICE=cuda:0
```

`SPECEDGE_EFFECTIVE_CONFIG` must be a run-snapshotted, Official-compatible YAML
such as [`official_client.example.yaml`](official_client.example.yaml).  Its
`client.host` must exactly equal `SPECEDGE_GRPC_ADDRESS`; the factory rejects a
mismatch rather than changing a transport endpoint.  `SPECEDGE_PROMPTS_PATH`
is the `prompts.jsonl` saved alongside the common arrival trace.  Prompts are
looked up by canonical `arrival_index`, then checked against the trace's
`dataset_index` and `task_id`; a prompt embedded in an ad-hoc request is never
silently substituted.

When the server is rendered from the unified Poisson experiment config, the
adapter writes the pinned Official-supported `server.batch_type: dynamic`.
This allows a partial batch to run when two independently replayed Poisson
partitions do not arrive at the same instant. It retains `max_batch_size: 2`
and `num_clients: 2`: the first remains the server's batch capacity and the
second remains the two logical edge clients. `batch_type` is an exposed
Official scheduler configuration; this adapter does not change the Official
tree-drafting/speculative algorithm or any source in `official/`.

Official SpecEdge has global client configuration and a single mutable graph
engine per process.  Use one physical A5000 and one adapter process per
logical client.  Pass the same global trace prefix to each process with a
different `--client-id`; do not expose `0,1` to one process or run this real
factory with `--max-concurrency > 1`.

Two processes on the same edge host must also share one monotonic origin.  The
wrapper starts both clients, each prepares its model and exclusively creates a
distinct ready JSON.  Once both ready files exist, the wrapper exclusively
creates one start JSON with a *future* `run_started_monotonic_s`; both clients
then replay their trace partition against that value.  This prevents model
loading or sequential shell startup from silently producing two different
Poisson time axes.

```bash
python baselines/specedge/adapter/poisson_client.py \
  --trace /home/hdd/zhangh/results/fastsd/workloads/mtbench80/arrival_trace.jsonl \
  --client-id "$SPECEDGE_CLIENT_ID" \
  --output /home/hdd/zhangh/results/fastsd/specedge-run/edge-client-0/metrics/requests.jsonl \
  --summary-output /home/hdd/zhangh/results/fastsd/specedge-run/edge-client-0/metrics/summary.json \
  --run-id specedge-mtbench-l0.8-s20260812-c0 \
  --max-concurrency 1 \
  --ready-file /home/hdd/zhangh/results/fastsd/specedge-run/sync/client-0.ready.json \
  --start-file /home/hdd/zhangh/results/fastsd/specedge-run/sync/common-start.json \
  --client-factory baselines.specedge.adapter.official_client_factory:create_client
```

The factory's `prepare` hook loads the draft model and executes the upstream
`Sync` RPC against that exact endpoint before the adapter starts its monotonic
measurement window.  Each request then constructs the known upstream
`SpecExecClient` and invokes `await client.generate(dataset_index)`.  Upstream
returns no completion object, but the pinned source stores the completed token
sequence in `client._prefix_tokens`, records the original prompt length in
`client._num_original_tokens`, and logs
`tokenizer.decode(client._prefix_tokens[0], skip_special_tokens=True)`.
Immediately after `generate()`, the adapter performs that same full-sequence
decode for audit and decodes the token suffix starting at
`_num_original_tokens` for the scorer-facing `client_result.text`.  It never
tries to remove the prompt with string matching.  The result records
`output_includes_prompt: false`, `generated_token_count`, prompt/full-sequence
token counts, and a prompt-inclusive `full_sequence_text` for source-equivalent
debugging.  Quality scorers must use `text`, not `full_sequence_text`.

The server's `specedge-server-effective.yaml` is safe to copy to node1 as the
immutable common algorithm/endpoint snapshot because it contains the full
client section that this factory validates.  Do **not** reuse its cloud result
directory on the edge: set `SPECEDGE_CLIENT_RESULT_PATH` to a fresh absolute
edge-local component directory (and optionally `SPECEDGE_CLIENT_EXP_NAME`) so
upstream log files cannot collide with cloud artifacts.  These overrides only
change upstream log destinations, not SpecEdge decoding parameters.

For a different reviewed integration, a caller may still provide its own
factory explicitly:

```python
# my_specedge_factory.py -- installed/available only in the dedicated SpecEdge env
async def _invoke_known_client(client, request_id: int):
    await client.generate(request_id)
    return {"request_id": request_id}

def create_client(request, context):
    # Build the known pinned official client here.  Do not change official code.
    client = build_client_from_my_pinned_environment(request, context)

    async def invoke():
        return await _invoke_known_client(client, int(request["dataset_index"]))

    return invoke
```

Run it only after separately verifying that factory against the pinned
Official SpecEdge revision:

```bash
PYTHONPATH=/path/containing/my_factory:$PYTHONPATH \
python baselines/specedge/adapter/poisson_client.py \
  --trace /home/hdd/zhangh/results/fastsd/workloads/mtbench80/arrival_trace.jsonl \
  --output /home/hdd/zhangh/results/fastsd/specedge-run/metrics/requests.jsonl \
  --summary-output /home/hdd/zhangh/results/fastsd/specedge-run/metrics/summary.json \
  --run-id specedge-mtbench-l0.8-s20260812 \
  --max-concurrency 2 \
  --client-factory my_specedge_factory:create_client
```

If `--dry-run` or `--client-factory` is not supplied, the program fails before
opening output files or attempting any Official SpecEdge API call.

## Artifacts

Each request record includes trace IDs/SHA linkage through the summary,
canonical `client_id`, scheduled and actual monotonic arrival times, arrival
lag, dispatch time, client queue wait, completion time, status, and structured
error metadata.  Official factory successes additionally include a tokenizer-
safe continuation under `client_result.text`; that field explicitly excludes
the prompt and has a corresponding `client_result.generated_token_count`.
The summary includes completion counts, P50/P95/P99 end-to-end latency, P95
arrival lag, trace SHA256, run ID, and output paths.  A per-client summary
sets `total_generated_tokens` only when every completed request supplied a
valid explicit count.  After the two partitions are merged, the canonical
summary computes `system_tok_per_s` from the measurement window (first actual
arrival to last completion), allowing the matrix analyzer to compute
J/generated-token when both edge and cloud GPU energy samples are present.
If any request record is `error` or `cancelled`, the client CLI prints the
preserved summary but exits non-zero; the standard two-client launcher then
does not merge or finalize it as a formal analysis run.
Monotonic timestamps are only comparable within one process; the corresponding
offsets are provided for analysis.
