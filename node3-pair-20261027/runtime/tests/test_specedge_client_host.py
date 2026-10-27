import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from scripts import eval_suite


REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENT_HOST = REPO_ROOT / "baselines" / "specedge" / "integration" / "client_host.py"
CONFIG = REPO_ROOT / "configs" / "evaluation" / "llama2_7b_tinyllama_four_method_cpu.json"


class SpecEdgeClientHostTests(unittest.TestCase):
    def test_absolute_client_host_runs_outside_repo_without_pythonpath(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "-S", str(CLIENT_HOST), "--help"],
            cwd=Path(__file__).resolve().anchor,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_node3_specedge_plan_exposes_repo_pythonpath(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        execution = config["execution"]

        def fake_layout(current_config, execution=None):
            resolved = execution or current_config["execution"]
            node3_root = (
                PurePosixPath(resolved["node3_repo"])
                / "exp"
                / "comparison"
                / current_config["run_id"]
            )
            node2_root = (
                PurePosixPath(resolved["node2_repo"])
                / "exp"
                / "comparison"
                / current_config["run_id"]
            )
            return {
                "linux_root": node3_root,
                "node3_root": node3_root,
                "node2_root": node2_root,
                "node3_canonical": node3_root / "inputs" / "canonical.jsonl",
                "node2_canonical": node2_root / "inputs" / "canonical.jsonl",
                "node3_specedge_config_linux": node3_root / "specedge" / "specedge.yaml",
                "node2_specedge_config_linux": node2_root / "specedge" / "node2.yaml",
                "commands": node3_root / "commands.txt",
                "status": node3_root / "run_status.jsonl",
            }

        node3_integration = f"{execution['node3_repo']}/baselines/specedge/integration"
        node3_official = f"{execution['node3_repo']}/baselines/specedge/official"
        expected_pythonpath = ":".join(
            (execution["node3_repo"], node3_integration, f"{node3_official}/src")
        )
        output = io.StringIO()
        with patch.object(eval_suite, "load_config", return_value=config), patch.object(
            eval_suite, "output_layout", side_effect=fake_layout
        ), patch.object(
            eval_suite,
            "_read_manifest",
            return_value={"execution": execution, "workload_hash": "test-hash"},
        ), patch.object(eval_suite, "_append_command_record"), patch.object(
            eval_suite, "append_status"
        ), contextlib.redirect_stdout(output):
            self.assertEqual(eval_suite.print_plan(str(CONFIG)), 0)

        plan = output.getvalue()
        self.assertIn(
            f"PYTHONPATH={shlex.quote(expected_pythonpath)}",
            plan,
        )
        self.assertIn(
            f"{execution['node3_repo']}/baselines/specedge/integration/client_host.py",
            plan,
        )
        draft_section = plan.split(
            "\n[node3：Draft-only（同一 CPU runtime；不访问 target）]", 1
        )[1].split("\n[node2：target-only", 1)[0]
        self.assertIn(
            f"PYTHON_BIN={shlex.quote(execution['node3_edge_python'])}",
            draft_section,
        )
        fastsd_section = plan.split(
            "\n[node3：FastSD stateful EdgeClient", 1
        )[1].split("\n[node2：SpecEdge target server", 1)[0]
        self.assertIn("--warmup_requests 10 --max_tasks_per_draft 0", fastsd_section)
        self.assertIn("edge/edge.py", fastsd_section)
        self.assertIn("--kv_batch_mode varlen", fastsd_section)
        self.assertIn("--token_budget 512", fastsd_section)
        self.assertNotIn("run_fastsd_profile.sh", fastsd_section)

    def test_node2_target_only_plan_exposes_repo_pythonpath_and_preserves_existing(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        execution = config["execution"]

        def fake_layout(current_config, execution=None):
            resolved = execution or current_config["execution"]
            node3_root = (
                PurePosixPath(resolved["node3_repo"])
                / "exp"
                / "comparison"
                / current_config["run_id"]
            )
            node2_root = (
                PurePosixPath(resolved["node2_repo"])
                / "exp"
                / "comparison"
                / current_config["run_id"]
            )
            return {
                "linux_root": node3_root,
                "node3_root": node3_root,
                "node2_root": node2_root,
                "node3_canonical": node3_root / "inputs" / "canonical.jsonl",
                "node2_canonical": node2_root / "inputs" / "canonical.jsonl",
                "node3_specedge_config_linux": node3_root / "specedge" / "specedge.yaml",
                "node2_specedge_config_linux": node2_root / "specedge" / "node2.yaml",
                "commands": node3_root / "commands.txt",
                "status": node3_root / "run_status.jsonl",
            }

        output = io.StringIO()
        with patch.object(eval_suite, "load_config", return_value=config), patch.object(
            eval_suite, "output_layout", side_effect=fake_layout
        ), patch.object(
            eval_suite,
            "_read_manifest",
            return_value={"execution": execution, "workload_hash": "test-hash"},
        ), patch.object(eval_suite, "_append_command_record"), patch.object(
            eval_suite, "append_status"
        ), contextlib.redirect_stdout(output):
            self.assertEqual(eval_suite.print_plan(str(CONFIG)), 0)

        node2_repo = execution["node2_repo"]
        expected_pythonpath = (
            f"PYTHONPATH={shlex.quote(node2_repo)}${{PYTHONPATH:+:$PYTHONPATH}}"
        )
        expected_command = (
            f"cd {shlex.quote(node2_repo)} && {expected_pythonpath} "
            f"{shlex.quote(execution['node2_target_python'])} "
            "benchmark/eval_target_only.py"
        )
        self.assertIn(expected_command, output.getvalue())


if __name__ == "__main__":
    unittest.main()
