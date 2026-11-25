"""Shared configuration and provenance for SpecEdge experiment variants."""

import math
import os
import subprocess
from pathlib import Path


OFFICIAL_REVISION = "1edcaf02ffc41a7b57726450c5357ed216a3b9bc"
OFFICIAL = "specedge_official"
OFFICIAL_OOM = "specedge_official_oom"
ADAPTED = "specedge_adapted"
SINGLE_CHAIN = "specedge_single_chain_adapted"
OFFLOAD = "specedge_offload_adapted_lru"
METHODS = (OFFICIAL, OFFICIAL_OOM, ADAPTED, SINGLE_CHAIN, OFFLOAD)
SAMPLING = (0.7, 0.8, 0)
RESULT_FIELDS = (
    "method", "instances", "model_pair", "dataset", "draft_structure", "gamma",
    "max_n_beams", "max_len", "batch_size", "prealloc_kv_bytes", "prealloc_clamped",
    "temperature", "top_p", "top_k", "seed", "ttft_fallback_to_e2e", "target_tp_size",
    "target_devices", "target_gpu_uuids", "target_model", "draft_model", "tokenizer", "draft_tokenizer",
    "target_dtype", "draft_dtype", "draft_device", "accept_rate", "accept_length",
    "draft_tokens", "accepted_tokens", "verify_steps", "acceptance_rate", "verify_method",
    "client_count", "arrival_distribution", "arrival_rate", "arrival_seed", "workload_hash",
    "max_tokens", "offload", "cuda_graph", "status", "oom", "requested_batch", "effective_batch", "gpu_slots",
    "draft_top_k",
)


def method_for(structure, offload=False, draft_tokens=None):
    if structure not in {"tree", "single_chain"}:
        raise ValueError(f"invalid draft_structure: {structure}")
    if offload:
        return OFFLOAD
    if structure == "single_chain":
        return SINGLE_CHAIN if draft_tokens is None else f"{SINGLE_CHAIN}_draft{draft_tokens}"
    return ADAPTED


def check_official_checkout(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if (Path(git("rev-parse", "--show-toplevel")).resolve() != root.resolve()
            or git("rev-parse", "HEAD") != OFFICIAL_REVISION
            or git("status", "--porcelain", "--untracked-files=no")):
        raise RuntimeError(f"official/ must be clean and pinned to {OFFICIAL_REVISION}")


def is_method(method):
    return method in METHODS or (isinstance(method, str) and method.startswith(f"{SINGLE_CHAIN}_draft")
                                 and method.removeprefix(f"{SINGLE_CHAIN}_draft").isdigit())


def runtime_options(structure, draft_tokens, *, offload=False, policy="lru", environ=None):
    env = os.environ if environ is None else environ
    value = env.get("SPECEDGE_KV_OFFLOAD", "1" if offload is True else "0")
    if value not in {"0", "1"} or not isinstance(offload, bool):
        raise ValueError("SPECEDGE_KV_OFFLOAD must be 0 or 1")
    if env.get("SPECEDGE_KV_OFFLOAD_POLICY", policy) != "lru":
        raise ValueError("SpecEdge supports only LRU offload")
    structure = env.get("SPECEDGE_DRAFT_STRUCTURE", structure)
    tokens = int(env.get("SPECEDGE_DRAFT_TOKENS", draft_tokens))
    if structure not in {"tree", "single_chain"} or tokens < 0 or (structure == "tree" and tokens == 0):
        raise ValueError("draft_tokens must be nonnegative and draft0 requires single_chain")
    return structure, tokens, value == "1"


def check_comparison_sampling(temperature, top_p, top_k, verify_method):
    if (float(temperature), float(top_p), int(top_k)) != SAMPLING or verify_method != "rejection":
        raise ValueError("SpecEdge comparison requires temperature=0.7, top_p=0.8, top_k=0, verify_method=rejection")


def apply_runtime_options(config):
    names = {"SPECEDGE_KV_OFFLOAD", "SPECEDGE_KV_OFFLOAD_POLICY", "SPECEDGE_DRAFT_STRUCTURE", "SPECEDGE_DRAFT_TOKENS"}
    if not names.intersection(os.environ):
        return config
    from copy import deepcopy

    config = deepcopy(config)
    client, server = config["client"], config["server"]
    original_method = validate_config(config)[0]
    structure, tokens, offload = runtime_options(
        client.get("draft_structure", "tree"), client["max_beam_len"], offload=server.get("offload", False),
        policy=server.get("offload_policy", "lru"),
    )
    server.update(offload=offload, offload_policy="lru")
    client.update(draft_structure=structure, max_beam_len=tokens, draft_tokens=tokens)
    if structure == "single_chain":
        client.update(max_n_beams=1, max_branch_width=1, max_budget=tokens)
        client["proactive"]["type"] = "disabled"
    elif "SPECEDGE_DRAFT_TOKENS" in os.environ:
        raise ValueError("SPECEDGE_DRAFT_TOKENS controls single_chain only")
    method = method_for(structure, offload, tokens)
    if original_method == OFFICIAL and not offload and structure == "tree":
        method = OFFICIAL
    if "integration" in config:
        config["integration"]["method"] = method
    elif method != OFFICIAL:
        raise ValueError("adapted modes require integration.dataset_file and completion_dir")
    return config


def check_sampling(temperature, top_p, top_k):
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("sampling requires temperature >= 0, 0 < top_p <= 1, and top_k >= 0")


def validate_config(config):
    server, client, base = (config[name] for name in ("server", "client", "base"))
    integration = config.get("integration", {})
    structure = client.get("draft_structure", "tree")
    tokens = int(client.get("draft_tokens", client["max_beam_len"]))
    offload = server.get("offload", False)
    runtime_options(structure, tokens, offload=offload, policy=server.get("offload_policy", "lru"), environ={})
    method = method_for(structure, offload, tokens if "draft_tokens" in client else None)
    expected = integration.get("method", OFFICIAL if not integration and not offload else method)
    if expected != method and not (expected == OFFICIAL and not offload and structure == "tree"):
        raise ValueError(f"configured SpecEdge method {expected!r} does not match {method!r}")
    check_sampling(float(server["temperature"]), float(server.get("top_p", 0.8)), int(server.get("top_k", 0)))
    devices = list(server.get("target_devices", [server["device"]]))
    if len(devices) not in {1, 2} or len(set(devices)) != len(devices) or devices[0] != server["device"]:
        raise ValueError("target_devices must contain one or two distinct CUDA devices, primary first")
    if any(not str(device).startswith("cuda:") or not str(device)[5:].isdigit()
           or str(device) != f"cuda:{int(str(device)[5:])}" for device in devices):
        raise ValueError("SpecEdge target requires explicit cuda:N devices")
    if expected == OFFICIAL and (len(devices) != 1 or server.get("allow_memory_clamp", False)
                                 or (float(server['temperature']), float(server.get("top_p", 0.8)), int(server.get("top_k", 0))) != SAMPLING):
        raise ValueError("native SpecEdge uses one GPU, official sampling, and no memory clamp")
    if offload and (len(devices) != 1 or server.get("allow_memory_clamp", False)):
        raise ValueError("SpecEdge LRU requires a single GPU and a fixed requested batch")
    if offload and not 1 <= gpu_slots(config) <= int(server["num_clients"]):
        raise ValueError("gpu_slots must be between 1 and num_clients")
    if isinstance(server.get("allow_memory_clamp", False), bool) is False:
        raise ValueError("allow_memory_clamp must be a boolean")
    if base["dtype"] not in {"bf16", "fp16"} or client.get("dtype", base["dtype"]) not in {"bf16", "fp16", "fp32"}:
        raise ValueError("unsupported SpecEdge dtype")
    if int(client["max_new_tokens"]) < 1 or int(base["max_len"]) < 1:
        raise ValueError("max_new_tokens and max_len must be positive")
    if int(client.get("draft_top_k", 0)) < 0:
        raise ValueError("draft_top_k must be nonnegative")
    if min(int(client[key]) for key in ("max_n_beams", "max_branch_width")) < 1:
        raise ValueError("beam width and branch width must be positive")
    if structure == "tree" and min(int(client["max_budget"]), int(client["max_beam_len"])) < 1:
        raise ValueError("tree beam length and verifier budget must be positive")
    if structure == "single_chain" and (
        client["max_n_beams"] != 1 or client["max_branch_width"] != 1
        or client["max_budget"] != tokens or client["max_beam_len"] != tokens or client["proactive"]["type"] != "disabled"
    ):
        raise ValueError("single_chain requires one beam, one child, budget=gamma, and proactive disabled")
    if integration and server.get("cache_prefill", False):
        raise ValueError("adapted SpecEdge requires runtime prefill")
    if integration and server.get("batch_type") != "dynamic":
        raise ValueError("adapted SpecEdge requires dynamic batching")
    if expected == OFFICIAL and server.get("cache_prefill", False):
        raise ValueError("specedge_official no-offload baseline requires cache_prefill=false")
    if expected == OFFICIAL and (base["dtype"] != "fp16" or client.get("dtype", "fp16") != "fp16"):
        raise ValueError("native SpecEdge raw NumPy wire encoding requires fp16; use the adapter for bf16")
    return expected, devices


def gpu_slots(config):
    server = config["server"]
    instances = int(server["num_clients"])
    return int(server.get("gpu_slots", min(int(server["max_batch_size"]), max(1, instances // 2)))) if server.get("offload") else instances


def verify_method_for(config, method):
    if config["client"].get("draft_structure") != "single_chain" or method == OFFICIAL:
        return "target_sample_match"
    return "rejection" if int(config["client"]["max_beam_len"]) else "target_only"


def result_metadata(config, budget):
    server, client, base = (config[name] for name in ("server", "client", "base"))
    method, devices = validate_config(config)
    integration = config.get("integration", {})
    return {
        "method": method,
        "status": "ok", "oom": False,
        "requested_batch": budget["requested_batch_size"], "effective_batch": budget["batch_size"],
        "offload": server.get("offload", False),
        "cuda_graph": not server.get("offload", False) and len(devices) == 1,
        "gpu_slots": gpu_slots(config),
        "instances": int(server["num_clients"]),
        "client_count": int(server["num_clients"]),
        "arrival_distribution": integration.get("arrival_distribution"),
        "arrival_rate": integration.get("arrival_rate"),
        "arrival_seed": integration.get("arrival_seed"),
        "workload_hash": integration.get("workload_hash"),
        "max_tokens": int(client["max_new_tokens"]),
        "draft_tokens": int(client["max_beam_len"]) if client.get("draft_structure") == "single_chain" else None,
        "draft_top_k": client.get("draft_top_k") if method != OFFICIAL else None,
        "verify_method": verify_method_for(config, method),
        "model_pair": config.get("integration", {}).get("model_pair") or {
            "target": server["target_model"], "draft": client["draft_model"],
        },
        "draft_structure": client.get("draft_structure", "tree"),
        "gamma": int(client["max_beam_len"]),
        "max_n_beams": int(client["max_n_beams"]),
        "tree_budget": int(client["max_budget"]),
        "max_len": int(base["max_len"]),
        "batch_size": budget["batch_size"],
        "prealloc_kv_bytes": budget["prealloc_kv_bytes"],
        "prealloc_clamped": budget["prealloc_clamped"],
        "temperature": float(server["temperature"]),
        "top_p": float(server.get("top_p", 0.8)),
        "top_k": int(server.get("top_k", 0)),
        "seed": int(base["seed"]),
        "target_tp_size": len(devices),
        "target_devices": devices,
        "target_model": server["target_model"],
        "draft_model": client["draft_model"],
        "tokenizer": server.get("tokenizer", server["target_model"]),
        "draft_tokenizer": client.get("tokenizer", client["draft_model"]),
        "target_dtype": base["dtype"],
        "draft_dtype": client.get("dtype", base["dtype"]),
    }
