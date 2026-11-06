"""Generate the four auditable dataset configs from one experiment template.

The generated files share the template's generation/topology/seed verbatim;
only the dataset name, request cap, run ID, and MT-Bench turn-policy metadata
change.  No data or model is downloaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))
from src.run_artifacts import write_json_once


DATASET_MATRIX = {
    "humaneval": {"max_requests": 164, "mt_bench_turn_policy": "not_applicable"},
    "mgsm": {"max_requests": 110, "mt_bench_turn_policy": "not_applicable"},
    "gsm8k": {"max_requests": 1319, "mt_bench_turn_policy": "not_applicable"},
    "mt_bench": {"max_requests": 80, "mt_bench_turn_policy": "first_turn_only"},
}
DATASET_SUFFIXES = tuple(f"_{name}" for name in DATASET_MATRIX)


def stable_run_id(base_run_id: str, dataset_name: str) -> str:
    """Append exactly one canonical dataset suffix to a base run ID."""

    stem = str(base_run_id)
    for suffix in sorted(DATASET_SUFFIXES, key=len, reverse=True):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return f"{stem}_{dataset_name}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_matrix(base_path: str | Path, output_dir: str | Path, data_root: str) -> dict[str, Any]:
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base_run_id = str(base["run_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    for dataset_name, details in DATASET_MATRIX.items():
        config = copy.deepcopy(base)
        config["run_id"] = stable_run_id(base_run_id, dataset_name)
        config["dataset"] = {
            **config.get("dataset", {}),
            "name": dataset_name,
            "data_path": data_root,
            "max_requests": details["max_requests"],
        }
        config["metadata"] = {
            **config.get("metadata", {}),
            "matrix_template": str(base_path),
            "matrix_dataset": dataset_name,
            "mt_bench_turn_policy": details["mt_bench_turn_policy"],
        }
        output_path = output_dir / f"{config['run_id']}.json"
        write_json_once(output_path, config)
        generated.append(
            {
                "dataset": dataset_name,
                "max_requests": details["max_requests"],
                "run_id": config["run_id"],
                "config": str(output_path),
                "mt_bench_turn_policy": details["mt_bench_turn_policy"],
            }
        )

    manifest = {
        "schema_version": 1,
        "base_config": str(base_path),
        "base_config_sha256": _sha256(base_path),
        "shared_generation": base["generation"],
        "shared_topology": base["topology"],
        "configs": generated,
    }
    write_json_once(output_dir / "matrix_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--data-root",
        required=True,
        help="existing repository data directory; no download is attempted",
    )
    args = parser.parse_args(argv)
    manifest = generate_matrix(args.base_config, args.output_dir, args.data_root)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
