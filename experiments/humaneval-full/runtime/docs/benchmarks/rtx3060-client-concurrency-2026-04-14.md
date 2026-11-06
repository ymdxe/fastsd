# RTX 3060 Client Concurrency Results (2026-04-14)

Model: `deepseek-coder-1.3b` GPTQ  
Server: `i-2.gpushare.com`  
GPU: `RTX 3060 12GB`  
Role under test: draft-side client replica density on a single GPU

## Measurement Setup

- Script: `scripts/benchmark_single_gpu_concurrency.py`
- Capacity probe: `scripts/test_single_gpu_model_capacity.py`
- Prompt: built-in benchmark prompt in the script
- Warmup runs: `1`
- Measure runs: `2`
- Max new tokens per run: `64`
- Concurrency levels: `1`, `2`, `4`

## Capacity Probe

The 12 GB RTX 3060 successfully loaded 4 replicas of `deepseek-coder-1.3b`.

| Replica count | Free GiB | Allocated GiB | Reserved GiB |
| --- | ---: | ---: | ---: |
| 1 | 8.70 | 1.20 | 1.26 |
| 2 | 7.46 | 2.39 | 2.49 |
| 3 | 6.23 | 3.59 | 3.73 |
| 4 | 4.99 | 4.78 | 4.96 |

## Throughput Results

Aggregate throughput is computed as the sum of worker `tokens_per_s`.

| Concurrency | Status | Loaded | Aggregate tok/s | Avg per-client tok/s | Min | Max | Avg peak GiB/worker | Approx total model GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ok | 1 | 36.00 | 36.00 | 36.00 | 36.00 | 1.22 | 1.22 |
| 2 | ok | 2 | 66.49 | 33.25 | 32.25 | 34.24 | 1.22 | 2.44 |
| 4 | ok | 4 | 130.29 | 32.57 | 31.57 | 33.22 | 1.22 | 4.89 |

## Takeaways

1. Four replicas fit comfortably on a 12 GB RTX 3060 with about `4.99 GiB` still free after load.
2. Per-client throughput drops only modestly as density rises:
   - `2` replicas: `-7.64%` vs. single replica
   - `4` replicas: `-9.51%` vs. single replica
3. Aggregate throughput scales close to linearly:
   - `1 -> 2`: `36.00 -> 66.49 tok/s`
   - `1 -> 4`: `36.00 -> 130.29 tok/s`

## Decision

For draft-side client simulation, `4` replicas per RTX 3060 is currently the best density among the tested points. It gives the highest aggregate throughput with only a small per-client penalty, so using `4` cards to simulate `16` clients is reasonable for the next experiment stage.

## Caveat

This benchmark measures concurrent local draft decoding on one GPU. It does not yet include cloud target contention, network delay, or end-to-end FastSD scheduling effects. Use it as a client-capacity baseline, not as the final multi-client system result.

## Longer-Sequence Stress Variant

To avoid the earlier short-sequence benchmark being too optimistic, the benchmark script was extended to support repeated prompts. A second pass was run with:

- `prompt_repeat=26`
- actual prompt length: `987` tokens
- `max_new_tokens=256`
- `warmup_runs=1`
- `measure_runs=2`

This setting is materially heavier than the original short-prompt run while still finishing in a reasonable amount of time on the RTX 3060.

### Longer-Sequence Results

| Concurrency | Status | Prompt tokens | Aggregate tok/s | Avg per-client tok/s | Min | Max | Avg peak GiB/worker | Approx total peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ok | 987 | 25.81 | 25.81 | 25.81 | 25.81 | 1.45 | 1.45 |
| 2 | ok | 987 | 66.04 | 33.02 | 32.39 | 33.65 | 1.46 | 2.91 |
| 4 | ok | 987 | 116.26 | 29.07 | 28.67 | 29.25 | 1.45 | 5.82 |

### Interpretation

1. Longer sequences do change the picture: single-instance throughput fell from `36.00 tok/s` in the short benchmark to `25.81 tok/s`.
2. Four replicas still fit and still give the highest aggregate throughput.
3. The decode-only metric used here can let `2` replicas beat `1` replica on per-client throughput, which likely means a single worker underutilizes the GPU at this prompt length. This should not be read as "adding replicas always makes each client faster" in full end-to-end serving.
4. Under this heavier setting, `4` replicas remain a reasonable client-simulation density on RTX 3060, but the stronger conclusion is about aggregate throughput, not latency fairness.
