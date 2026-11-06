"""Launch one method's GPU Target, preserving exact arguments and lifecycle."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
method = sys.argv[1]
assert method in {"standard", "fastsd"}
out = ROOT / method / "target"
out.mkdir(parents=True, exist_ok=True)
model = str(ROOT / "model-view" / "Qwen3-32B")
args = [sys.executable, "-u", str(ROOT / "target_adapter.py"),
        "--draft_model", model, "--target_model", model,
        "--target_device", "cuda:0", "--target_dtype", "float16", "--draft_dtype", "float16",
        "--kv_batch_mode", "padded", "--dataset", "gsm8k",
        "--profile", "custom" if method == "fastsd" else "vanilla",
        "--server_sched_mode", "fastsd" if method == "fastsd" else "vanilla",
        "--no-pipeline_gamma_adapt", "--gamma", "100", "--max_tokens", "256",
        "--batch_size", "8", "--token_budget", "512", "--num_drafts", "16",
        "--verify_method", "greedy", "--temp", "0", "--top_k", "0", "--top_p", "1",
        "--no-measure_energy", "--exp_name", str(out)]
for feature in ["enable_pipeline", "enable_proactive_draft", "enable_latency_priority"]:
    args.append("--" + ("" if method == "fastsd" else "no-") + feature)
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="3", CLOUD_SERVICE_HOST="127.0.0.1", CLOUD_SERVICE_PORT="18625",
           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1",
           OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false")
(out / "launch.json").write_text(json.dumps({"command": args, "environment": {k: env[k] for k in ["CUDA_VISIBLE_DEVICES", "CLOUD_SERVICE_HOST", "CLOUD_SERVICE_PORT", "OMP_NUM_THREADS"]}}, indent=2))
with (out / "target.log").open("w") as log:
    child = subprocess.Popen(args, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    (out / "target.pid").write_text(str(child.pid))
    code = child.wait()
(out / "exit.json").write_text(json.dumps({"exit_status": code, "ended_epoch": time.time()}))
sys.exit(code)
