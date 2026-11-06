"""Launch adapted SpecEdge clients locally or over SSH from one config."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _find_repo_root() -> Path:
    """Find the checkout root when this absolute script is run externally."""

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "src" / "run_artifacts.py").is_file():
            return candidate
    raise RuntimeError(
        "cannot locate FastSD repository root containing src/run_artifacts.py "
        f"from {Path(__file__).resolve()}"
    )


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.run_artifacts import append_command, append_status


INTEGRATION_ROOT = Path(__file__).resolve().parent
SPECEDGE_ROOT = INTEGRATION_ROOT.parent / "official"


def _client_environment(config: dict, client_idx: int, device: str, start_epoch: float) -> dict[str, str]:
    base = config["base"]
    client = config["client"]
    proactive = client["proactive"]
    integration = config["integration"]
    client_dtype = client.get(
        "dtype",
        "fp32" if str(device).split(":", 1)[0] == "cpu" else base["dtype"],
    )
    values = {
        "SPECEDGE_OPTIMIZATION": config["opt"],
        "SPECEDGE_RESULT_PATH": base["result_path"],
        "SPECEDGE_EXP_NAME": base["exp_name"],
        "SPECEDGE_PROCESS_NAME": f"{client['process_name']}_{client_idx}",
        "SPECEDGE_SEED": base["seed"],
        "SPECEDGE_MAX_LEN": base["max_len"],
        "SPECEDGE_DRAFT_MODEL": client["draft_model"],
        "SPECEDGE_DEVICE": device,
        "SPECEDGE_DTYPE": client_dtype,
        "SPECEDGE_ENGINE": client.get("engine", "official_graph_engine"),
        "SPECEDGE_THREADS": client.get("threads", 1),
        "SPECEDGE_DATASET": client["dataset"],
        "SPECEDGE_MAX_N_BEAMS": client["max_n_beams"],
        "SPECEDGE_MAX_BEAM_LEN": client["max_beam_len"],
        "SPECEDGE_MAX_BRANCH_WIDTH": client["max_branch_width"],
        "SPECEDGE_MAX_BUDGET": client["max_budget"],
        "SPECEDGE_PROACTIVE_TYPE": proactive["type"],
        "SPECEDGE_PROACTIVE_MAX_N_BEAMS": proactive["max_n_beams"],
        "SPECEDGE_PROACTIVE_MAX_BEAM_LEN": proactive["max_beam_len"],
        "SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH": proactive["max_branch_width"],
        "SPECEDGE_PROACTIVE_MAX_BUDGET": proactive["max_budget"],
        "SPECEDGE_MAX_NEW_TOKENS": client["max_new_tokens"],
        "SPECEDGE_MAX_REQUEST_NUM": -1,
        "SPECEDGE_REQ_OFFSET": 0,
        "SPECEDGE_SAMPLE_REQ_CNT": 1,
        "SPECEDGE_HOST": client["host"],
        "SPECEDGE_CLIENT_IDX": client_idx,
        "SPECEDGE_REASONING": client.get("reasoning", False),
        "FASTSD_EVAL_DATASET_FILE": integration["dataset_file"],
        "FASTSD_EVAL_COMPLETION_DIR": integration["completion_dir"],
        "FASTSD_EVAL_NUM_CLIENTS": config["server"]["num_clients"],
        "FASTSD_EVAL_START_EPOCH": start_epoch,
        "FASTSD_EVAL_WORKLOAD_HASH": integration["workload_hash"],
        "FASTSD_EVAL_METHOD": integration.get("method", "specedge_cpu_adapted"),
        "FASTSD_EVAL_ARRIVAL_DISTRIBUTION": integration.get(
            "arrival_distribution", "immediate"
        ),
        "FASTSD_EVAL_WARMUP_REQUESTS": integration.get("warmup_requests", 0),
    }
    return {key: str(value) for key, value in values.items()}


def main(config_file: str) -> int:
    import yaml

    with Path(config_file).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    start_epoch = time.time() + float(config["integration"].get("startup_delay_s", 15.0))
    python = Path(config["integration"].get("python", sys.executable))
    client_script = INTEGRATION_ROOT / "client.py"
    host_script = Path(__file__).resolve()
    processes: list[subprocess.Popen] = []
    client_idx = 0

    for node_name, clients in config["node"].items():
        for client_info in clients:
            env = _client_environment(config, client_idx, client_info["device"], start_epoch)
            if node_name in {"local", "localhost"}:
                process_env = os.environ.copy()
                process_env.update(env)
                process = subprocess.Popen(
                    [str(python), str(client_script)],
                    cwd=SPECEDGE_ROOT,
                    env=process_env,
                )
            else:
                exports = " && ".join(
                    f"export {key}={shlex.quote(value)}" for key, value in env.items()
                )
                command = (
                    f"{exports} && cd {shlex.quote(str(SPECEDGE_ROOT))} && "
                    f"{shlex.quote(str(python))} {shlex.quote(str(client_script))}"
                )
                process = subprocess.Popen(
                    ["ssh", "-i", str(config["base"]["ssh_key"]), node_name, command]
                )
            processes.append(process)
            client_idx += 1

    if client_idx != int(config["server"]["num_clients"]):
        raise ValueError(
            f"configured {client_idx} clients but server.num_clients="
            f"{config['server']['num_clients']}"
        )
    exit_code = max((process.wait() for process in processes), default=0)
    integration = config["integration"]
    # The recorded command must identify this orchestration entry point.  The
    # child processes intentionally still execute client.py, but rerunning the
    # recorded host command is what reproduces the fan-out behavior.
    command = shlex.join([str(python), str(host_script), "--config", str(config_file)])
    status_path = integration.get("status_path")
    if status_path:
        append_status(
            status_path,
            method=integration.get("method", "specedge_cpu_adapted"),
            phase="run",
            exit_code=exit_code,
            command=command,
        )
    commands_path = integration.get("commands_path")
    if commands_path:
        append_command(
            commands_path,
            command,
            config=config_file,
            status=exit_code,
            note=integration.get("method", "specedge_cpu_adapted"),
        )
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.config))
