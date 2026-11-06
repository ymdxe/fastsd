# Heterogeneous draft experiment results

Status: **completed**.

node1: 8 Qwen3-1.7B drafts, fixed γ=100. node3: 8 Qwen3-1.7B drafts, fixed γ=20.
Requested output cap: 256 tokens.
Target: Qwen3-32B; dtype=float16; physical GPU=3; CPU weight offload=True; GPU weight budget=34 GiB.

| Method | Group | Requests | Output tokens | TTFT mean / p50 / p95 (s) | TPOT mean / p50 / p95 (ms) | E2E mean / p50 / p95 (s) |
| --- | --- | ---: | ---: | --- | --- | --- |
| standard | overall | 16 | 3723 | 163.792 / 154.588 / 257.718 | 5257.371 / 5445.642 / 7776.274 | 1368.654 / 1340.289 / 2077.786 |
| standard | node1 | 8 | 1997 | 110.773 / 107.974 / 148.534 | 5284.690 / 5553.560 / 7737.448 | 1408.228 / 1543.014 / 1948.940 |
| standard | node3 | 8 | 1726 | 216.811 / 220.027 / 263.594 | 5230.051 / 5227.777 / 7223.568 | 1329.080 / 1179.603 / 2086.346 |
| fastsd | overall | 16 | 3723 | 218.011 / 171.942 / 409.610 | 7461.861 / 7523.669 / 12218.379 | 1890.370 / 1790.460 / 3471.452 |
| fastsd | node1 | 8 | 1997 | 93.643 / 101.819 / 102.135 | 5349.174 / 5635.904 / 7858.397 | 1406.865 / 1539.001 / 1982.651 |
| fastsd | node3 | 8 | 1726 | 342.380 / 366.731 / 415.868 | 9574.547 / 8540.988 / 13622.614 | 2373.875 / 2260.562 / 3847.076 |
| specedge | overall | 16 | 3714 | 278.294 / 228.729 / 495.779 | 10832.311 / 11316.300 / 15972.118 | 2662.075 / 2701.187 / 3855.450 |
| specedge | node1 | 8 | 1997 | 125.171 / 123.902 / 134.672 | 8228.711 / 8478.942 / 11938.511 | 2143.961 / 2292.025 / 2857.336 |
| specedge | node3 | 8 | 1717 | 431.418 / 445.453 / 502.758 | 13435.911 / 12762.378 / 16798.215 | 3180.189 / 3382.775 / 4111.102 |

| Method | Output tokens | Draft acceptance | RPC errors | Coordinator window (s) | Aggregate throughput (tokens/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 3723 | 17.18% | 0 | 2229.623 | 1.670 |
| fastsd | 3723 | 17.20% | 0 | 4178.865 | 0.891 |
| specedge | 3714 | 15.79% | 0 | 4338.135 | 0.856 |

fastsd vs Standard SD: output token sequences match for **16/16** paired requests; mismatch details are in results.json.

specedge vs Standard SD: output token sequences match for **15/16** paired requests; mismatch details are in results.json.

## Observed host load

| Method | Node | Valid / total snapshots | Load1 min | Load1 mean | Load1 max |
| --- | --- | ---: | ---: | ---: | ---: |
| standard | node1 | 104 / 104 | 0.310 | 4.873 | 7.220 |
| standard | node3 | 114 / 114 | 0.330 | 4.255 | 7.570 |
| fastsd | node1 | 105 / 105 | 0.540 | 5.515 | 8.370 |
| fastsd | node3 | 212 / 212 | 12.260 | 111.550 | 172.940 |
| specedge | node1 | 148 / 148 | 0.530 | 5.974 | 8.280 |
| specedge | node3 | 220 / 220 | 1.100 | 33.681 | 171.280 |

These shared-host snapshots include experiment work and unrelated activity. They are not background-only load or CPU utilization. Methods were run sequentially under different host loads, so their latency and throughput differences cannot be attributed to the algorithm alone. The complete recorded load traces are plotted in `host_load.png` when the plotting script is run.

## Target scheduling and asynchronous prefetch

| Method | Counter scope | Plans | Verify slices | Prefill slices | Prefetch submitted | Prefetch completed | Prefetch failed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | Lifetime incl. smoke / polling | 0 | 0 | 0 | 0 | 0 | 0 |
| fastsd | Lifetime incl. smoke / polling | 308 | 364 | 18 | 0 | 0 | 0 |
| specedge | Formal phase; native counters NA | NA | NA | NA | NA | NA | NA |

Native counters come from archived `target/scheduler_metrics.json` and cover the Target process lifetime, including a separate 4-token smoke and idle polling. Plans are shown exactly as archived; no inferred smoke plan count is subtracted. Formal latency metrics exclude smoke. Missing values are NA.
- **standard formal workload only**: 363 verify RPCs and 16 prefill RPCs. Lifetime slices in the table also include smoke; chunked prefill may produce multiple slices per RPC.
- **fastsd formal workload only**: 362 verify RPCs and 16 prefill RPCs. Lifetime slices in the table also include smoke; chunked prefill may produce multiple slices per RPC.

SpecEdge resets its statistics and KV when entering the formal phase; any SpecEdge server counters shown use that phase's scope. FastSD-specific counters are not synthesized for it. Request completion alone does not demonstrate that all enabled scheduling/prefetch mechanisms worked.
- **fastsd**: No asynchronous prefetch task was submitted during the Target process lifetime (prefetch_async_submitted=0). This mechanism was not exercised; zero completed/failed counters are not evidence of successful prefetch execution.

## Metric definitions

- **ttft_seconds**: Native request_start (before session/init) to first verified output block; excludes model loading, startup clock calibration and warmup.
- **tpot_ms**: (request E2E - TTFT)/(generated_tokens - 1), including remaining executor work at request completion. Requests with fewer than two output tokens are excluded. This is amortized block-output latency, not a measured inter-token arrival distribution.
- **percentiles**: Linear interpolation at (n-1)*q; overall percentiles pool individual requests from both nodes, not node percentiles.
- **acceptance_rate_weighted**: sum(accepted_total)/sum(drafted_total); includes verified candidate tokens discarded at the final output cap.
- **throughput**: Total retained output tokens divided only by explicit coordinator wall_seconds. No subtraction of monotonic timestamps from different hosts.
- **startup_window**: The current coordinator starts its common window at the model-ready barrier, including subsequent clock calibration; native request TTFT/E2E start after calibration. Consult each method's timing.json for actual boundaries.
- **verify_rpc_ms**: Observed verification round-trip duration: HTTP for Standard SD/FastSD and gRPC for SpecEdge; protocol latency is not pure network RTT.
- **output_comparison**: Pair by (node, task_id), because task identifiers may overlap across nodes.
- **host_load1**: Whole-host Linux one-minute load average sampled from progress.jsonl, including this experiment and unrelated activity. Mean is the arithmetic mean of snapshots, not a time-weighted average. Snapshots cover launcher monitoring, including startup. Host load is not CPU utilization or a measurement of background load alone; values from different nodes are not pooled.
- **scheduler_counters**: Native Target counters cover process lifetime including a separate 4-token smoke and idle polling; formal request metrics exclude smoke. No inferred smoke plans are subtracted. SpecEdge server counters, if shown, start after its formal-phase statistics/KV reset. Request success and asynchronous prefetch execution are reported separately; zero submitted tasks means the mechanism was not exercised. No missing counter is assumed to be zero.

## Interpretation

- TTFT includes generating and verifying the first entire candidate block; γ=100 therefore directly affects time to first visible output.
- This is one run per method on shared hosts. Node hardware, runtime versions, prompt assignments and fixed γ differ; node differences cannot be attributed to γ alone.
- Host load changed between methods, particularly on node3. These observations are not a controlled algorithm speed comparison; all request measurements and host snapshots are retained.
- GPU/CPU weight offload, when enabled, is part of the measured Target configuration and should be reported with every performance comparison.
- Short capped outputs do not constitute a GSM8K accuracy evaluation; complete recorded output equality is separate from reference-answer correctness.
- SpecEdge is reported as `specedge_cpu_adapted_single_chain`: single-chain SpecExec with proactive drafting and gRPC, using experiment adapters for CPU execution, heterogeneous candidate lengths and Target weight offload. See the archived SpecEdge adapters and `specedge_plan.md` for implementation details.
- SpecEdge candidate acceptance excludes each cycle's Target bonus token; retained output tokens are counted after EOS and the output cap. Its gRPC round-trip timings are reported separately from the other methods' HTTP timings.
