# Third-round WRR and controlled KV prefetch

This implements the third-round plan against `7637595` on
`fix/wrr-kv-prefetch-20261101`.

## Scheduling behavior

The Prefill and Verify priority formulas remain independent. The compatibility
dispatcher calls the original formula for each task type. A seeded comparison
against the original Git object produced 6,000 exact floating-point matches
(3,000 Prefill and 3,000 Verify, including timing aliases, legacy requests,
cold starts and timing gates).

The persistent WRR sequence is six Short, three Mid, one Long. Each successful
slice consumes one visit. The visit chooses a queue before comparing scores
within that queue:

| Visit | Queue order |
| --- | --- |
| Short | SV, SP, MV, MP, LV, LP |
| Mid | MV, SV, SP, MP, LV, LP |
| Long | LV, SV, SP, MV, MP, LP |

An overdue Prefill in the visit's own class retains the existing protection.
Empty scans and insufficient tail budget consume no visits. A selected
candidate that cannot fit seals the batch and keeps that visit for the next
batch. An impossible slice under a full budget raises an explicit error.
Bridge tokens count toward the shared budget. Final commitment ages only the
plan's participating snapshot, once; later arrivals receive no retroactive age.
The optional `fcfs` and `progress` experiment policies remain available.

## Complete next-plan prediction

`predict_next_plan` copies scheduling metadata, projects current slices, applies
known outcomes, re-buckets both task types against frozen thresholds, and starts
from the current plan's next WRR cursor and cycle. Already applied phases are
not advanced twice. Unknown Verify results are recorded as `all_accept`
assumptions. Prediction leaves live queues, WorkItems and classifier history
unchanged.

The full next plan consumes its normal token and sequence budget, including
residual work on a currently protected device. Only afterward are protected
devices removed from the transfer intents. Cache readiness never changes
formal paper-WRR admission. The executor receives the current job before
lookahead runs, and duplicate prediction refreshes are suppressed for unchanged
scheduling metadata.

## Transfer resources and calibration

The manager prepares bounded GPU slabs and reusable copy events before
execution. Speculative preparation during execution borrows these resources
and requires pinned CPU input; pageable input waits for demand. The D2H pinned
buffer pool remains available for cache offload. Missing resources defer
prefetch. Storage
leases track aliasing by pointer ranges and retain regions through DMA,
consumption and formal cache ownership. Demand claims the same pending ticket
and continues its remaining chunks. Consumers wait only on their own ticket.

Speculative submissions have both a byte limit per group and a token-bucket
rate limit. Demand H2D and required D2H suppress new speculative groups. One
speculative group may remain in flight. Previously issued DMA is not preempted.
The requested mode stays visible; the controller records `overlap`,
`throttled`, `paused` and explicit calibration `probe` states and reasons.

Compute contexts include task phase, batch size, new tokens per sequence,
cached context lengths, dtype and attention backend. Unknown contexts pause
new compute-window speculative DMA; idle-window prefetch remains available.
Load accepted calibration evidence using `--kv_prefetch_calibration PATH`.
Calibration loading checks the physical GPU UUID, canonical model path, dtype,
attention backend, forward primitive and KV batching mode before authorizing
any shape.
Other controls are `--kv_prefetch_rate_mb_s`,
`--kv_prefetch_idle_rate_mb_s`, `--kv_prefetch_chunk_kb`, and
`--kv_prefetch_event_pool_size`.

## Verification and performance boundary

Run `python -B -m unittest discover -s tests -q`. The `test_round3_*` modules
exercise visit traces, successful-visit accounting, exact projection versus
commitment, applied phases, cancellation, storage ownership, controller limits,
submission ordering and paired-result acceptance.

`scripts/replay_prefetch_pair.py --help` describes a frozen-input GPU replay.
It compares A (normal demand), B (CPU preparation), C (the original Git transfer
implementation), and D (controlled prefetch), with A/A noise estimation and
randomized A/D pairs. It reports forward GPU time, launch delay, exposed H2D
wait, paired bootstrap intervals, output/KV equality, and profiler
memcpy/kernel intersections for speculative copies and the timed production
forward. The 256 KiB, 1 MiB and 4 MiB groups are experimental
settings. Only strictly accepted shapes are exported in `calibration.json`.
Prefill continuation can be represented with a nonzero cached context.

The timed forward and restored-KV verification both call production
`varlen_generate` with `KVCacheModel_batching` and bounded probability windows.
Acceptance also requires lower P50 and P95 total host wait from next-batch
readiness to usable KV, including the worker join, as well as a lower exposed
H2D stall. A smaller ticket wait alone cannot hide preparation or joining cost.
Profiler evidence includes only GPU kernels and speculative H2D activities
correlated with the labeled current forward and background transfer calls;
missing thread attribution produces an inconclusive result.

As of November 2, 2026, the selected GPU server's cards are occupied by other
users. No new GPU performance acceptance has been measured. Neither passing
CPU tests nor an enclosing CUDA-event overlap establishes the proposed P50
1% / P95 2% forward limits. Request TTFT, TPOT and E2E also require a separate
end-to-end workload; this replay does not report them as measured.

Deployment receipts and the scheduler's detailed traces live in the local
task directory `fastsd-wrr-prefetch-fix-20261101/round3/evidence`. Server sync
backs up the changed files, preserves existing branches and histories, verifies
SHA-256 per file, and appends each server's WORKLOG.
