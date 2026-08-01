# FastSD Repository Guide

This document is the primary orientation guide for humans and future agents working in this repository. It focuses on how the codebase is organized today, how the main runtime path works, how experiments are launched, and what caveats matter before making changes.

## 1. What This Repository Is

FastSD is a research repository for speculative decoding in a cloud-edge setting. In the current form of the repo, there are two overlapping generations of code:

- an older benchmark-oriented path that runs decoding directly from `benchmark/*.py` and `comparison/*.py`
- a newer service-oriented path where:
  - the cloud side runs a FastAPI target service in `cloud/cloud_service.py`
  - the edge side runs the draft loop and HTTP coordination logic in `edge/edge.py`

The branch history suggests the newer path was added incrementally:

- `cloud`
- `communication ok`
- `draft while waiting`
- `pipeline`
- `fastsd`
- `energy meter`
- `fix: align FastSD scheduler with paper logic`

That means the repo is not a polished product package. It is an active research tree with legacy scripts, partially overlapping entrypoints, and a few environment-specific assumptions.

## 2. Paper Goals vs. Current Code

This distinction is critical:

- the paper describes the intended FastSD system design
- the repository contains the current implementation state

These are related, but they are not identical. Future agents should never infer "the paper says X" therefore "the code already does X".

### 2.1 Paper-Level System Goal

The paper frames FastSD as a cloud-edge speculative decoding system for large-scale heterogeneous deployments, where:

- edge devices run draft LLMs
- the cloud runs the target LLM
- one cloud target serves many edge draft clients over time
- the system is optimized for many-request production-style service rather than one edge paired with one cloud model

At the paper level, the system is motivated by four facts:

- decoder-only LLM inference is bottlenecked by sequential autoregressive decoding
- speculative decoding can reduce target-side steps by validating multiple draft tokens in one target forward pass
- cloud-edge deployment introduces communication delay, heterogeneous devices, and multi-client contention
- under many concurrent requests, KV cache movement between CPU or disk and GPU becomes a real bottleneck

### 2.2 Paper-Level Innovation Claims

The paper text you provided maps FastSD to three main design contributions.

#### A. Priority-based Multi-queue Task Scheduling

Target motivation:

- cloud receives a mixture of prefill and verify tasks
- tasks differ in sequence length, urgency, device speed, network latency, and draft acceptance quality
- a single FIFO queue would let long or expensive tasks block short ones and would waste batch efficiency

Paper design:

- split tasks by type:
  - prefill
  - verify
- split each type into three length queues:
  - short
  - medium
  - long
- update length boundaries online using quantiles over a sliding window
- assign priorities inside each queue using:
  - draft-side compute speed
  - communication condition
  - acceptance probability
  - draft length
  - waiting time

#### B. Dynamic Weighted Round-Robin Batching

Target motivation:

- batching improves GPU utilization
- mixing very different sequence lengths causes padding waste
- always prioritizing verify can starve prefill
- always processing prefill can explode verify latency

Paper design:

- serve short, medium, and long queues in a fixed 6:3:1 ratio
- prioritize verify tasks by default
- switch to prefill only when verify demand is insufficient
- use a conservative switching rule based on underutilized verify opportunities

#### C. Prediction-based KV Cache Preloading

Target motivation:

- in large-scale service, KV cache often cannot stay resident on GPU for all active requests
- verify rounds repeatedly need previously built KV cache
- loading KV cache from CPU memory or disk into GPU introduces non-trivial latency

Paper design:

- predict which verify tasks are most likely to run next
- proactively preload their KV cache into GPU memory while the current batch is still running
- reduce the I/O stall before the next verify batch

### 2.3 What the Paper Assumes That Matters for Reading the Code

The paper narrative assumes the following terms:

- prefill task:
  the first speculative round that initializes target-side KV cache
- verify task:
  every later speculative round that validates a new drafted suffix against the cached prefix
- speculative round:
  one draft phase on the edge plus one validation phase on the cloud

Those terms appear in the code, but the code should still be treated as an implementation attempt rather than a proof that the full paper system is finished.

## 3. Current Working State

As of the current local checkout:

- branch: `master`
- top-level tracked code directories: `benchmark/`, `cloud/`, `comparison/`, `data/`, `docs/`, `edge/`, `scripts/`, `src/`, `tests/`
- top-level output directory already present: `exp/`
- the working tree is not clean

Current local uncommitted state matches the remote development server and includes:

- modified:
  - `cloud/cloud_service.py`
  - `install.sh`
  - some `__pycache__` files
- untracked:
  - `requirements.clean.txt`
  - `tests/__pycache__/test_fastsd_scheduler.cpython-310.pyc`

Implication:

- do not assume this checkout is a pristine baseline
- inspect `git status --short` before editing
- if you run new experiments, use a fresh `--exp_name` so you do not mix outputs with existing runs

## 4. Mental Model of the Codebase

The fastest way to understand the repo is to think in four layers.

### 4.1 Core Decoding Layer

Location: `src/`

This is the real center of the project. The most important files are:

- `src/engine.py`
- `src/util.py`
- `src/kvcache.py`
- `src/kvcache_batching.py`
- `src/kvcache4RC.py`
- `src/energy_meter.py`

What they do:

- `src/engine.py`
  - defines the abstract `Decoding` base class
  - loads models and tokenizers
  - implements decoding variants
  - contains the FastSD scheduling helpers added more recently
  - integrates optional energy measurement
- `src/util.py`
  - central argument parsing
  - model alias to path mapping through `model_zoo`
  - sampling helpers like `norm_logits`, `sample`, `top_k_top_p_filter`
- `src/kvcache_batching.py`
  - batched KV-cache management keyed by process id
  - used by the cloud-side worker path
- `src/energy_meter.py`
  - power sampling and FastAPI control service for cloud-side energy measurement

Important design fact:

- most scripts are thin wrappers
- almost all substantive behavior eventually flows through `Decoding` in `src/engine.py`

### 4.2 Benchmark Layer

Location: `benchmark/`

These files are dataset-specific evaluation entrypoints:

- `benchmark/eval_humaneval.py`
- `benchmark/eval_gsm8k.py`
- `benchmark/eval_mgsm.py`
- `benchmark/eval_mt_bench.py`
- `benchmark/eval_SingleDraft.py`
- `benchmark/eval_MultiDraft.py`

Pattern:

- each file subclasses `Decoding`
- each file defines:
  - `load_data`
  - `preprocess`
  - `postprocess`
  - `eval`
- `eval()` selects the decoding implementation based on `args.eval_mode`

Common modes you will see:

- `small`: autoregressive decoding on the draft model only
- `large`: autoregressive decoding on the target model only
- `sd`: single-process speculative decoding
- `para_sd`: two-process or parallel speculative decoding
- `para_sd_wo_1` / `para_sd_wo_2`: ablations with one optimization removed
- `rc_para_sd`: a specialized variant used by the newer cloud-edge path

### 4.3 Cloud-Edge Service Layer

Locations:

- `cloud/cloud_service.py`
- `edge/edge.py`

This is the most important path for the current repository direction.

Cloud side:

- `cloud/cloud_service.py` starts a FastAPI service
- it exposes:
  - `GET /health`
  - `POST /session/init`
  - `POST /prefill`
  - `POST /verify`
  - `POST /exit`
- behind the API, it spins up a worker process that reuses `Decoding.run_target_process_batching(...)`
- request and response queues are multiprocessing queues
- the API thread hands requests to the worker and waits on futures

Edge side:

- `edge/edge.py` contains:
  - `EdgeClient`, an HTTP client for the cloud target service
  - `EdgeRunner`, which reuses `Decoding` and runs the local draft loop
- this file handles:
  - session creation
  - draft token generation
  - asynchronous verify requests to the cloud
  - overlap between drafting and waiting
  - pipeline-specific gamma adaptation
  - per-task metric logging into `exp/<exp_name>/edge_metrics_proc*.jsonl`

Conceptually:

1. edge generates draft tokens
2. edge sends a verify request to cloud
3. cloud verifies against the target model
4. edge merges accepted tokens and final token
5. edge repeats until stop conditions fire

### 4.4 Experiment Shell Layer

Location: `scripts/`

These files are the practical entrypoints most users will touch.

Relevant current scripts:

- `scripts/run_fastsd_profile.sh`
- `scripts/run_vanilla_profile.sh`
- `scripts/case_study.sh`
- `scripts/ablation_study.sh`
- `scripts/run_sd.sh`
- `scripts/run_para_sd.sh`
- `scripts/run_assist.sh`
- `scripts/run_comp.sh`

Important distinction:

- `run_fastsd_profile.sh` and `run_vanilla_profile.sh` target the newer `edge/edge.py` path
- `run_sd.sh` and `run_para_sd.sh` are older benchmark launch collections
- `run_assist.sh` and `run_comp.sh` are comparison scripts for external baselines

## 5. Current Implementation Status Relative to the Paper

This section is intentionally separate from the paper-goal section above.

### 5.1 Clearly Reflected in Current Code

These ideas are directly visible in the repository today.

- Cloud-edge split exists.
  - cloud service in `cloud/cloud_service.py`
  - edge runner in `edge/edge.py`
- Prefill and verify are explicit request types in the newer cloud path.
- Multi-queue scheduling exists in `src/engine.py`.
- Short / mid / long classification exists and is updated dynamically.
- The fixed weighted round-robin order exists.
- Verify-first scheduling with conditional switching to prefill exists.
- Target-side KV cache is explicitly managed across requests.
- KV cache offload between GPU and CPU is now explicitly part of the request-handling logic.
- Energy measurement hooks exist for cloud-side runs.

### 5.2 Partially Implemented or Still Research-Code Level

These are present in some form, but should not be treated as finished production implementations.

- Priority scoring exists, but it should be read as a current heuristic implementation, not as a finalized system abstraction.
- Pipeline adaptation exists, but it is still intertwined with experiment code and debug flags.
- Proactive drafting and verify overlap exist in `edge/edge.py`, but the control flow is still research-code style.
- KV cache preloading exists in code structure, but the surrounding system is still an evolving implementation rather than a hardened serving subsystem.
- The repository still mixes old benchmark paths and newer cloud-edge runtime paths.

### 5.3 Not Safe to Assume from the Code Alone

Future readers should avoid over-claiming the following based only on the current repository.

- that every paper contribution is fully implemented end to end
- that all baselines are normalized under the same runtime assumptions
- that all shell scripts correspond to the latest paper setup
- that the current repo is already a production-grade serving system
- that all heterogeneous-device factors described in the paper are fully modeled in the current code

### 5.4 Concrete Example of Goal vs. Implementation

The KV-cache story is a good example:

- paper goal:
  KV cache movement between CPU or disk and GPU is a production bottleneck, so FastSD should reduce latency through smarter preloading and scheduling
- current code:
  the repository now explicitly offloads target-side KV cache between GPU and CPU in the target request paths, and includes logic related to preloading / keeping selected caches on GPU

That means the code reflects the paper motivation, but it still does not automatically mean the full intended serving strategy is complete or fully evaluated.

## 6. Detailed Directory Map

### `src/`

Purpose:

- reusable decoding logic, cache logic, model loading, scheduler logic, utilities

Files worth reading first:

- `src/engine.py`
- `src/util.py`
- `src/kvcache_batching.py`
- `src/energy_meter.py`

Read order recommendation:

1. `src/util.py`
2. `src/engine.py`
3. `src/kvcache_batching.py`
4. `edge/edge.py`
5. `cloud/cloud_service.py`

### `cloud/`

Purpose:

- HTTP target-model service used by the edge runner

Main file:

- `cloud/cloud_service.py`

Operational note:

- the file uses multiprocessing with `spawn`
- this matters if you change worker initialization or move code that must be pickle-safe

### `edge/`

Purpose:

- edge-side speculative decoding loop and cloud communication

Main file:

- `edge/edge.py`

Why it matters:

- this is where the current FastSD runtime behavior is most visible
- pipeline and proactive drafting behavior are coordinated here

### `benchmark/`

Purpose:

- dataset-specific evaluation harnesses

Datasets covered:

- HumanEval
- GSM8K
- MGSM
- MT-Bench

Output pattern:

- writes `.jsonl` results into `exp/<exp_name>/`

### `comparison/`

Purpose:

- evaluation scripts for non-FastSD baselines like Assist and Ouroboros

Caveat:

- some shell scripts still refer to `comparation/...`, which does not match the actual directory name `comparison/`
- these scripts should be treated as historical and verified before reuse

### `data/`

Purpose:

- evaluation datasets packaged as `.jsonl`

Current files:

- `humaneval.jsonl`
- `gsm8k.jsonl`
- `mgsm.jsonl`
- `mt_bench.jsonl`

### `exp/`

Purpose:

- experiment outputs and edge metrics summaries

Current directories include:

- `exp/fastsd`
- `exp/vanilla_run`
- `exp/proactive_run`
- `exp/pipeline_run`
- `exp/both_run`
- `exp/test`

Example summary file shape:

- `profile`
- `enable_pipeline`
- `enable_proactive_draft`
- `num_drafts`
- `num_tasks`
- `wallclock_s`
- `total_generated_tokens`
- `system_tok_per_s`
- latency percentiles

### `tests/`

Purpose:

- lightweight unit tests for newer logic

Current test coverage:

- `tests/test_energy_meter.py`
  - validates `EnergyAccumulator` integration logic
- `tests/test_fastsd_scheduler.py`
  - validates:
    - fixed  round robin order
    - prefill switch threshold
    - runtime quantile thresholds
    - priority score calculation

Important limitation:

- there are no end-to-end integration tests for cloud/edge interaction
- most runtime behavior is still validated by manual experiments

## 7. Core Runtime Architecture

The current practical architecture is:

```text
edge/edge.py
  -> EdgeClient sends HTTP requests
  -> cloud/cloud_service.py receives requests
  -> worker process calls Decoding target-side batching path
  -> edge merges verify response and continues drafting
  -> metrics written to exp/<exp_name>/
```

The main abstractions are:

- `Decoding`
  - owns model loading and decoding algorithms
- `KVCacheModel_batching`
  - keeps per-request cache state for batched verification on the cloud side
- `EdgeClient`
  - cloud API wrapper
- `CloudTargetWorker`
  - target-side worker built on `Decoding`

### FastSD scheduler pieces in `src/engine.py`

The newer scheduling logic includes helper functions such as:

- `_build_fixed_wrr_order`
- `_update_length_thresholds`
- `_compute_priority_score`
- `_should_switch_to_prefill`

What that indicates:

- requests are categorized by prompt length
- the scheduler uses dynamic quantile thresholds after warmup
- verify priority is adjusted using lag, transport RTT, wait time, and acceptance statistics
- the cloud can opportunistically switch to prefill when verify utilization is low enough

This is one of the main places where the paper design is visible in implementation form.

## 8. Configuration and Arguments

Most runtime flags are defined in `src/util.py` via `parse_arguments()`.

Important arguments:

- dataset:
  - `humaneval`
  - `gsm8k`
  - `mt_bench`
- model selection:
  - `--draft_model`
  - `--target_model`
- decoding:
  - `--eval_mode`
  - `--gamma`
  - `--num_drafts`
  - `--batch_size`
- output:
  - `--exp_name`
- FastSD behavior:
  - `--server_sched_mode`
  - `--profile`
  - `--enable_proactive_draft`
  - `--enable_pipeline`
  - `--pipeline_gamma_adapt`
- diagnostics:
  - `--debug_pipeline`
  - `--debug_verify_tokens`
- energy:
  - `--measure_energy`
  - `--energy_api_host`
  - `--energy_api_port`

Important caveat:

- model names in scripts are aliases, not direct filesystem paths
- `src/util.py:model_zoo()` maps aliases to local paths
- this mapping still contains environment-specific placeholders and hard-coded local paths

Practical implication:

- if a script fails on model loading, `model_zoo()` is the first place to inspect

## 9. How Experiments Are Usually Run

### 7.1 Install

```bash
bash install.sh
```

This installs:

- PyTorch 2.1.2 CUDA 12.1 wheels
- Transformers
- Accelerate
- AutoGPTQ
- FastAPI / Uvicorn
- `nvidia-ml-py3` for power measurement

### 7.2 Newer cloud-edge runtime

Cloud side:

```bash
python cloud/cloud_service.py --exp_name fastsd
```

Edge side, FastSD profile:

```bash
bash scripts/run_fastsd_profile.sh fastsd --num_drafts 4 --dataset humaneval --max_tokens 256
```

Edge side, baseline profile:

```bash
bash scripts/run_vanilla_profile.sh vanilla vanilla_run --num_drafts 4 --dataset humaneval --max_tokens 256
```

If cloud is not local, set:

```bash
export SERVER_URL=http://<cloud-host>:8001
```

Important note:

- `scripts/run_fastsd_profile.sh` points edge to `http://127.0.0.1:8001` by default
- `cloud/cloud_service.py` currently starts Uvicorn on port `8001`

That mismatch means one of these must be changed before use. This is an active repo caveat, not a user error.

### 7.3 Older benchmark path

Single-process speculative decoding examples live in:

- `scripts/run_sd.sh`

Parallel speculative decoding examples live in:

- `scripts/run_para_sd.sh`

These scripts mainly call:

```bash
accelerate launch benchmark/<dataset-script>.py ...
```

### 7.4 Comparison baselines

Examples:

- `scripts/run_assist.sh`
- `scripts/run_comp.sh`

Treat these scripts as references, not guaranteed-clean production entrypoints.

## 10. Outputs and Metrics

Most experiment outputs go under `exp/<exp_name>/`.

Typical files:

- task-level metrics:
  - `edge_metrics_proc0.jsonl`
  - `edge_metrics_proc1.jsonl`
  - ...
- aggregate metrics:
  - `edge_metrics_summary.json`
- benchmark results:
  - `<eval_mode>_<dataset>.jsonl`

Metrics already present in summaries include:

- wallclock time
- total generated tokens
- system throughput in tokens/s
- average task latency
- p50/p90/p95 latency

Task-level metrics in `edge/edge.py` also record:

- per-round draft time
- verify wait time
- cloud service time
- transport RTT estimate
- accept rate
- proactive reuse statistics

## 11. Test and Verification Commands

Current unit tests:

```bash
python -m unittest tests.test_energy_meter tests.test_fastsd_scheduler -v
```

What they cover well:

- deterministic helper logic
- energy integration arithmetic

What they do not cover:

- end-to-end cloud/edge interaction
- model loading
- benchmark correctness against full datasets
- script compatibility

If you change cloud-edge behavior, add at least one more unit test around scheduler or metrics logic before trusting manual runs.

## 12. Known Caveats and Fragile Spots

This section matters more than most for future agents.

### Environment-specific model paths

`src/util.py` still contains local path assumptions and placeholders in `model_zoo()`.

### Port mismatch between scripts and service

- `scripts/run_fastsd_profile.sh` and `scripts/run_vanilla_profile.sh` default to `http://127.0.0.1:8001`
- `cloud/cloud_service.py` runs on port `8001`

This should be reconciled before assuming the shell scripts are plug-and-play.

### Historical scripts may be stale

Some scripts still reference names that do not exist now, such as `comparation/...`.

### Mixed generations of workflow

The repo contains:

- direct benchmark runs
- comparison baselines
- newer cloud-edge service orchestration

These are related, but not fully normalized under a single CLI.

### Paper narrative and code narrative are not identical

This repository must be read with two separate questions:

1. What is FastSD trying to be according to the paper?
2. What does this code actually implement today?

If you do not keep those separate, it is easy to write incorrect docs, incorrect experiment descriptions, or incorrect assumptions for later agents.

### Unclean working tree

The local repository intentionally mirrors an unclean remote development tree.

### `README.md` was previously empty

This guide is currently the more reliable orientation document.

## 13. Recommended Reading Order for New Agents

If you are a future agent and need to become productive quickly, read in this order:

1. `README.md`
2. `docs/REPOSITORY_GUIDE.md`
3. `src/util.py`
4. `src/engine.py`
5. `edge/edge.py`
6. `cloud/cloud_service.py`
7. `tests/test_fastsd_scheduler.py`
8. one launch script:
   - `scripts/run_fastsd_profile.sh`
   - or `scripts/run_vanilla_profile.sh`

Then inspect:

- `git status --short`
- the target `exp/<exp_name>/` directory
- any command the current user actually intends to run

## 14. Recommended Next Cleanup Tasks

These are not required to use the repo, but they would sharply improve maintainability:

1. Unify the cloud service port and shell script defaults.
2. Move model path mapping out of source code into config.
3. Separate legacy benchmark entrypoints from current cloud-edge runtime docs.
4. Add an end-to-end smoke test for `cloud/cloud_service.py` plus `edge/edge.py`.
5. Clean or ignore transient `__pycache__` artifacts in version control.
6. Replace stale shell scripts that reference `comparation/`.
7. Add one explicit document that maps each paper claim to concrete code locations and experiment scripts.

## 15. Minimal Operator Checklist

Before running anything:

1. Check `git status --short`.
2. Confirm `src/util.py:model_zoo()` points to real local models.
3. Confirm cloud URL and port match.
4. Use a fresh `--exp_name`.
5. Decide whether you are running:
   - older benchmark scripts
   - newer cloud-edge profile scripts
6. If measuring energy, ensure `nvidia-ml-py3` is installed and the selected GPU index is correct.

## 16. Summary

The repo is best understood as a research worktree centered on `Decoding` in `src/engine.py`, with a newer cloud-edge runtime layered on top through `cloud/cloud_service.py` and `edge/edge.py`. The shell scripts are useful, but not all of them are equally current. Just as importantly, the paper goal and the current implementation are not the same thing. The safest default for future work is:

- treat `src/`, `edge/`, and `cloud/` as the canonical runtime path
- treat `benchmark/` as dataset harnesses
- treat `comparison/` and some shell scripts as historical or semi-manual
- verify ports, model paths, and working tree state before running experiments
