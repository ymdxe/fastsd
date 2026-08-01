# RTX 3060 Client Concurrency Measurement Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Measure how a single RTX 3060 behaves when hosting 1, 2, and 4 concurrent `deepseek-coder-1.3b` client-model replicas, then infer a reasonable per-GPU client density for later multi-GPU simulation.

**Architecture:** Reuse the existing per-process throughput benchmark in `scripts/benchmark_single_gpu_concurrency.py` so each simulated client owns one full model instance and decodes independently on the same GPU. Pair those runs with the capacity probe in `scripts/test_single_gpu_model_capacity.py` and summarize both aggregate throughput and per-client slowdown.

**Tech Stack:** Python 3.11, PyTorch CUDA 12.1, AutoGPTQ, remote SSH execution on GPUShare.

---

### Task 1: Confirm the benchmark contract

**Files:**
- Read: `scripts/benchmark_single_gpu_concurrency.py`
- Read: `scripts/test_single_gpu_model_capacity.py`

**Step 1:** Verify that benchmark workers each load one replica and report per-worker throughput.

**Step 2:** Verify that the capacity probe can detect whether 4 replicas fit on 12 GB VRAM.

**Step 3:** Fix benchmark code only if the current scripts cannot answer aggregate throughput, per-worker throughput, or load-failure status.

### Task 2: Define the measurement envelope

**Files:**
- Read: `scripts/benchmark_single_gpu_concurrency.py`

**Step 1:** Fix the experiment constants for all runs:
- model: `/hy-tmp/models/deepseek-coder-1.3b`
- GPU: `0`
- concurrency: `1`, `2`, `4`
- prompt: built-in benchmark prompt
- warmup runs: `1`
- measure runs: `2`
- max new tokens: `64`

**Step 2:** Treat benchmark `avg_tokens_per_s` as per-worker throughput and compute aggregate throughput as the sum of worker `tokens_per_s`.

**Step 3:** Capture peak memory and any load/runtime failures per concurrency level.

### Task 3: Run the remote probe

**Files:**
- Execute: `scripts/test_single_gpu_model_capacity.py`
- Execute: `scripts/benchmark_single_gpu_concurrency.py`

**Step 1:** Run the capacity probe on the remote RTX 3060 to see whether at least 4 replicas can be loaded.

**Step 2:** Run concurrency benchmarks for `1`, `2`, and `4`.

**Step 3:** Save raw JSON outputs locally for later comparison.

### Task 4: Summarize the result

**Files:**
- Create: `docs/benchmarks/rtx3060-client-concurrency-2026-04-14.md` if durable notes are useful

**Step 1:** Report, for each concurrency level:
- status
- loaded replicas
- aggregate throughput
- average per-client throughput
- min/max per-client throughput
- average peak memory

**Step 2:** State the decision-relevant conclusion:
- whether 4 replicas fit
- whether 2 or 4 replicas are throughput-efficient
- recommended per-3060 simulated client count for the next experiment stage
