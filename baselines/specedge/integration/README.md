# SpecEdge in the FastSD evaluation suite

Use `qwen3.eval.example.json` with `scripts/eval_suite.py prepare` and `plan`.
Set host-local model/tokenizer paths, interpreters, and `topology.instances`;
the number of `specedge_draft_devices` must match. Run the generated server
and client commands, then `normalize --method <exact method label> --input
<raw/run_id>`. Use `qwen3.comparison.example.json` for the 128-instance
performance run. Set the same absolute repository path on both hosts and
replace the model, host, SSH key, and interpreter paths. Keep separate
environments for FastSD, adapted SpecEdge, and native SpecEdge. For native,
run `UV_PROJECT_ENVIRONMENT=/path/to/native-venv uv sync --project baselines/specedge/official --python 3.14 --locked`
with the pinned checkout. Point the native interpreter settings to that environment.

| Label | Runtime |
| --- | --- |
| `specedge_official` | Native verifier, sampler, tree, wire format, and CUDA Graph. The formal suite supplies canonical prompts and observes request metrics. |
| `specedge_official_oom` | Native memory preflight or CUDA OOM terminated the run. |
| `specedge_adapted` | Canonical dataset/transport adaptation using the official tree search and verifier. |
| `specedge_single_chain_adapted_draft0` | Target-only verification RPCs with no draft checkpoint allocation. |
| `specedge_single_chain_adapted_draft4` | One chain of four proposals using official Tree nodes and rejection verification. |
| `specedge_offload_adapted_lru` | Eager single-GPU verifier with explicit LRU slots and CPU backing. |

The unified suite sets `max_n_beams=1`, `max_branch_width=1`, and
`tree_budget=gamma` in single-chain mode. `instances` controls how many draft
clients run, not the width of a chain. To run an adapted tree experiment,
set `specedge.draft_structure: tree` and explicitly provide
`max_n_beams`, `max_branch_width`, and `tree_budget` under `specedge`.
Use `SPECEDGE_DRAFT_STRUCTURE=single_chain` and `SPECEDGE_DRAFT_TOKENS=0`
or `4` to select the chain length. Set the same environment for `prepare`,
`plan`, and both launchers. Each variant needs a fresh `run_id`.

On a single target GPU, the adapted verifier uses the official CUDA Graph
engine. For TP, set `specedge.target_devices` to `["cuda:0", "cuda:1"]` (CUDA
ordinals after visibility filtering). The current two-card backend is eager
dense Qwen3: it shards attention heads (including K/V heads) and MLP channels
by layer, combines partial outputs on the primary GPU, and keeps both the
batch and session KV split. The primary GPU owns embeddings, final logits,
one verifier/sampler, and one gRPC endpoint. The full checkpoint is staged in
host RAM before sharding; no full-weight GPU replica is required. No CUDA
Graph is claimed for TP, and a requested TP run never falls back to one GPU.

The priority model pair is Qwen3-8B plus Qwen3-0.6B. The adapters check the
entire token-to-ID map, model vocab size, and effective EOS/PAD/BOS before
generation. Qwen3 uses chat formatting with thinking disabled. The Qwen3-32B
pair, Llama-2/TinyLlama and DeepSeek-Coder pairs are not yet hardware-validated.
Paths and BF16/FP16 are configurable; CPU draft defaults to FP32.

HumanEval and MBPP pass@1 require explicit isolated execution at normalization
(`--code-runner-image` with a pre-pulled image containing Python and `timeout`).
GSM8K/MGSM use numeric answers. MT-Bench uses observed prior replies and emits
per-turn output for judging; no judge score is inferred. For a first-turn-only
FastSD comparison, set `specedge.mt_bench_turn_policy` to `first_turn_only`.
For the performance comparison, set `dataset.fixed_history_path` to a JSON
map from MT-Bench sample ID to the list of previous assistant replies. Every
turn is an independent scheduled request whose prompt contains the same
preceding turns in both implementations. Normalization saves generated text
and token IDs for every dataset. Performance runs skip quality execution.

## No-offload and memory boundary

For the formal native variant, set `specedge.method=specedge_official`,
`specedge.draft_structure=tree`, and explicit positive `max_n_beams`,
`max_branch_width`, and `tree_budget`. Set FP16 for target and draft, provide
CUDA draft devices, and use the native interpreters. Allocate memory for a
draft engine per client on the configured device. Use `topology.fastsd_kv_budget_gb=0`
for the paired FastSD no-pressure run. Run the server and client commands
printed by `plan`, or call `scripts/zjh_ab/run_specedge.sh` with `budget`,
`server`, or `client` followed by the generated YAML. Set `SPECEDGE_PYTHON`
to the interpreter for that host and `SPECEDGE_BIND_HOST`/`SPECEDGE_PORT`
to the desired server listener when using the shell wrapper.

The formal preset fixes `temperature=0.7`, `top_p=0.8`, `top_k=0`, a shared
seed, and one Poisson schedule at 1.6 requests/s across all 128 clients.
Native SpecEdge reports `target_sample_match`, FastSD and adapted draft4
report `rejection`, and draft0 reports `target_only`. Native SpecEdge's
top-p implementation also differs from the adapted nucleus filter. The
comparison checks exact per-task prompt-token hashes, EOS, token caps,
workload hash, and arrival offsets before emitting a main table.

Per GPU, compute `KV_slot = 2 × layers × local_KV_heads × head_dim × max_len ×
dtype_bytes`. The conservative bound uses `batch_size × max_n_beams × KV_slot`,
plus `instances × KV_slot`, one prefill scratch slot, model weights, mask/logit
estimates and a reserve. The pinned cache's resident allocation is actually
`(batch_size + instances) × KV_slot`; the beam dimension is not separately
allocated. For example, 36 layers, 8 KV heads, head dim 128, BF16 and length
2048 give 288 MiB per slot (144 MiB per card with TP2).
For a single-chain example at that length, take 24 GiB free, a 2 GiB reserve,
about 15.26 GiB of Qwen3-8B weights, CPU draft instances, and `gamma=4`.
The verifier width is `gamma + 1 = 5`, not the beam width. Four instances
at batch 4 need nine KV slots including scratch (2.53 GiB), or about
17.84 GiB total before reserve. If batch size equals `instances`, the same
estimate puts 11 at about 21.86 GiB and 12 at about 22.44 GiB, on opposite
sides of the 22 GiB usable budget. These are planning estimates, not measured
OOM boundaries; repeat with the actual free memory and model configuration.

Run `integration/server.py --config <rendered-yaml> --budget-only` on the target
host. Confirm the last fitting batch passes, the next fails before loading
weights, and run a short workload below the boundary. The check uses both
total and free bytes and is conservative, not an exact CUDA OOM predictor.
`allow_memory_clamp: true` is limited to adapted no-offload configurations.
Native failure exits nonzero and saves `specedge_run.json` with the requested
and effective batch, workload hash, and OOM label. Pass that artifact to
`normalize --method specedge_official_oom`.

## Pressure comparison

`SPECEDGE_KV_OFFLOAD=0` is the default. Set `SPECEDGE_KV_OFFLOAD=1` for the
LRU adapter. `SPECEDGE_KV_OFFLOAD_POLICY` accepts `lru`. The adapter allocates
`gpu_slots` persistent GPU slots plus the verifier's batch scratch. It moves
evicted K/V to CPU, restores synchronously before forward, and records
`evictions`, `restores`, `h2d_bytes`, `d2h_bytes`, and exposed `io_stall`.
Artifacts include `cuda_graph=false` and `offload=true` for this path.
Use fewer slots than active clients to exercise restores. For a first GPU
smoke, force one GPU slot and overlapping arrivals from at least two clients.

Collect target `specedge_run.json` and the client `specedge/requests` directory
alongside the raw run. `compare` takes the normalized `summary.json` paths.
Provide FastSD no-pressure and native success for the fit case. For the
pressure case, provide FastSD pressure, LRU, and the normalized native OOM
artifact. Keep the draft-token sweep in the appendix.
