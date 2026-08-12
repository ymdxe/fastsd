import argparse
import concurrent.futures
import glob
import json
import multiprocessing as mp
import os
import re
import statistics
import sys
import time
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List

import requests
import torch

sys.path.append(os.path.join(sys.path[0], "../"))

from src.engine import Decoding
from src.arrival import read_trace_jsonl, sha256_file, validate_trace_rows
from src.kvcache import KVCacheModel
from src.util import parse_arguments, seed_everything


class EdgeClient:
    """边缘端 HTTP 客户端，负责与云端 target 服务通信。"""

    def __init__(self, server_url: str, timeout: float = 30.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        # 避免 127.0.0.1 请求被环境变量代理劫持到其他服务。
        self.session.trust_env = False

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.server_url}{path}"
        response = self.session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def health(self) -> Dict[str, Any]:
        url = f"{self.server_url}/health"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def init_session(self) -> str:
        resp = self._post("/session/init", {})
        return resp["session_id"]

    def prefill(
        self,
        session_id: str,
        task_id: str,
        draft_output: List[int],
        prefix_len: int,
        lag: float,
        current_time: float,
    ) -> Dict[str, Any]:
        payload = {
            "session_id": session_id,
            "task_id": task_id,
            "draft_output": draft_output,
            "prefix_len": prefix_len,
            "lag": lag,
            "current_time": current_time,
        }
        return self._post("/prefill", payload)

    def verify(
        self,
        session_id: str,
        task_id: str,
        draft_output: List[int],
        prefix_len: int,
        lag: float,
        current_time: float,
        gamma: int,
        transport_rtt: float = 0.0,
        tail_only: bool = False,
        has_bridge_token: bool = False,
    ) -> tuple[Dict[str, Any], float]:
        payload = {
            "session_id": session_id,
            "task_id": task_id,
            "draft_output": draft_output,
            "prefix_len": prefix_len,
            "lag": lag,
            "current_time": current_time,
            "gamma": gamma,
            "transport_rtt": transport_rtt,
            "tail_only": tail_only,
            "has_bridge_token": has_bridge_token,
        }
        transport_start = time.monotonic()
        resp = self._post("/verify", payload)
        measured_http_total = max(0.0, time.monotonic() - transport_start)
        return resp, measured_http_total


class EdgeRunner(Decoding):
    """边缘端运行器：复用 Decoding 基类并执行 draft 侧循环。"""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.load_tokenizer()
        self.answer_trigger = "The answer is"
        self.gsm8k_prompt = self._create_gsm8k_demo_text(
            n_shot=8,
            cot_flag=True,
            answer_trigger=self.answer_trigger,
        )

    def load_data(self):
        return

    def preprocess(self, input_text):
        return input_text.strip()

    def postprocess(self, input_text, output_text):
        return output_text

    def _create_gsm8k_demo_text(self, n_shot: int = 8, cot_flag: bool = True, answer_trigger: str = "The answer is") -> str:
        """Build GSM8K few-shot prompt text aligned with benchmark/eval_gsm8k.py."""
        question = []
        chain = []
        answer = []

        question.append(
            "There are 15 trees in the grove. "
            "Grove workers will plant trees in the grove today. "
            "After they are done, there will be 21 trees. "
            "How many trees did the grove workers plant today?"
        )
        chain.append(
            "There are 15 trees originally. "
            "Then there were 21 trees after some more were planted. "
            "So there must have been 21 - 15 = 6."
        )
        answer.append("6")

        question.append(
            "If there are 3 cars in the parking lot and 2 more cars arrive, "
            "how many cars are in the parking lot?"
        )
        chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        answer.append("5")

        question.append(
            "Leah had 32 chocolates and her sister had 42. If they ate 35, "
            "how many pieces do they have left in total?"
        )
        chain.append(
            "Originally, Leah had 32 chocolates. "
            "Her sister had 42. So in total they had 32 + 42 = 74. "
            "After eating 35, they had 74 - 35 = 39."
        )
        answer.append("39")

        question.append(
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
            "has 12 lollipops. How many lollipops did Jason give to Denny?"
        )
        chain.append(
            "Jason started with 20 lollipops. Then he had 12 after giving some "
            "to Denny. So he gave Denny 20 - 12 = 8."
        )
        answer.append("8")

        question.append(
            "Shawn has five toys. For Christmas, he got two toys each from his "
            "mom and dad. How many toys does he have now?"
        )
        chain.append(
            "Shawn started with 5 toys. If he got 2 toys each from his mom and "
            "dad, then that is 4 more toys. 5 + 4 = 9."
        )
        answer.append("9")

        question.append(
            "There were nine computers in the server room. Five more computers "
            "were installed each day, from monday to thursday. "
            "How many computers are now in the server room?"
        )
        chain.append(
            "There were originally 9 computers. For each of 4 days, 5 more "
            "computers were added. So 5 * 4 = 20 computers were added. "
            "9 + 20 is 29."
        )
        answer.append("29")

        question.append(
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
            "wednesday, he lost 2 more. "
            "How many golf balls did he have at the end of wednesday?"
        )
        chain.append(
            "Michael started with 58 golf balls. After losing 23 on tuesday, "
            "he had 58 - 23 = 35. After losing 2 more, "
            "he had 35 - 2 = 33 golf balls."
        )
        answer.append("33")

        question.append(
            "Olivia has $23. She bought five bagels for $3 each. "
            "How much money does she have left?"
        )
        chain.append(
            "Olivia had 23 dollars. "
            "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
            "So she has 23 - 15 dollars left. 23 - 15 is 8."
        )
        answer.append("8")

        demo_text = ""
        for i in range(min(n_shot, len(question))):
            if cot_flag:
                demo_text += (
                    "Q: "
                    + question[i]
                    + "\nA: "
                    + chain[i]
                    + " "
                    + answer_trigger
                    + " "
                    + answer[i]
                    + ".\n\n"
                )
            else:
                demo_text += (
                    "Question: "
                    + question[i]
                    + "\nAnswer: "
                    + answer_trigger
                    + " "
                    + answer[i]
                    + ".\n\n"
                )
        return demo_text

    def _preprocess_gsm8k(self, question_text: str) -> str:
        """Format GSM8K input in the same few-shot style as benchmark script."""
        return self.gsm8k_prompt + "Q: " + question_text + "\nA:"

    @staticmethod
    def _percentile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        ordered = sorted(float(v) for v in values)
        idx = int(round((len(ordered) - 1) * q))
        idx = max(0, min(len(ordered) - 1, idx))
        return ordered[idx]

    def _metrics_path(self, proc_id: int) -> str:
        if getattr(self.args, "run_dir", None):
            return os.path.join(self.args.run_dir, "metrics", f"requests_proc{proc_id}.jsonl")
        return os.path.join(self.args.exp_name, f"edge_metrics_proc{proc_id}.jsonl")

    def _outputs_path(self, proc_id: int) -> str:
        if getattr(self.args, "run_dir", None):
            return os.path.join(self.args.run_dir, "outputs", f"completions_proc{proc_id}.jsonl")
        return os.path.join(self.args.exp_name, f"completions_proc{proc_id}.jsonl")

    def _metric_input_paths(self) -> List[str]:
        if getattr(self.args, "run_dir", None):
            return sorted(glob.glob(os.path.join(self.args.run_dir, "metrics", "requests_proc*.jsonl")))
        return sorted(glob.glob(os.path.join(self.args.exp_name, "edge_metrics_proc*.jsonl")))

    def _prepare_worker_outputs(self, proc_id: int) -> None:
        """Reserve unique worker artifacts before it becomes ready for replay."""
        for path in (self._metrics_path(proc_id), self._outputs_path(proc_id)):
            if os.path.exists(path) or os.path.islink(path):
                raise FileExistsError(
                    f"refusing to reuse worker artifact: {path}; choose a new run_id"
                )
            with open(path, "x", encoding="utf-8"):
                pass

    @staticmethod
    def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _truncate_at_eos(prefix: torch.Tensor, eos_token_id: int, start_idx: int) -> tuple[torch.Tensor, bool]:
        """Truncate newly generated suffix at the first EOS token if present."""
        if eos_token_id is None:
            return prefix, False
        if start_idx >= prefix.shape[1]:
            return prefix, False
        suffix = prefix[0, start_idx:].tolist()
        if eos_token_id not in suffix:
            return prefix, False
        eos_offset = suffix.index(eos_token_id)
        keep_len = start_idx + eos_offset + 1
        return prefix[:, :keep_len], True

    @staticmethod
    def _truncate_on_humaneval_markers(generated_text: str) -> tuple[str, bool]:
        """
        HumanEval samples should stop at function solution.
        If model starts generating tests/runner blocks, truncate before them.
        """
        markers = [
            "\nif __name__ == \"__main__\":",
            "\n#tests",
            "\nimport unittest",
            "\nimport pytest",
            "\nclass Test",
        ]
        cut = -1
        for marker in markers:
            idx = generated_text.find(marker)
            if idx != -1 and (cut == -1 or idx < cut):
                cut = idx
        if cut == -1:
            return generated_text, False
        return generated_text[:cut].rstrip(), True

    @staticmethod
    def _truncate_on_gsm8k_markers(generated_text: str) -> tuple[str, bool]:
        """
        GSM8K few-shot prompt uses `Q:`/`A:` pattern.
        Stop once model starts generating the next `Q:` block.
        """
        match = re.search(r"\n\s*Q:", generated_text)
        if match is None:
            return generated_text, False
        cut = match.start()
        return generated_text[:cut].rstrip(), True

    @staticmethod
    def _is_degenerate_repeat(prefix: torch.Tensor, input_len: int, window: int = 64) -> bool:
        """
        Stop on obvious collapse mode: a long run of the same token.
        This prevents runaway outputs like '111111...' or repeated whitespace.
        """
        if prefix.shape[1] - input_len < window:
            return False
        tail = prefix[0, -window:]
        return bool((tail == tail[-1]).all().item())

    def _write_summary(self, orchestration_wallclock_s: float) -> None:
        records: List[Dict[str, Any]] = []
        for path in self._metric_input_paths():
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))

        task_e2e = [float(r.get("task_e2e_ms", 0.0)) for r in records]
        prefill_http = [float(r.get("prefill_http_ms", 0.0)) for r in records]
        prefill_queue = [float(r.get("prefill_queue_ms", 0.0)) for r in records]
        prefill_service = [float(r.get("prefill_service_ms", 0.0)) for r in records]
        verify_queue = [float(r.get("avg_verify_queue_ms", 0.0)) for r in records]
        verify_service = [float(r.get("avg_verify_service_ms", 0.0)) for r in records]
        prefill_chunks = [int(r.get("prefill_chunks", 1)) for r in records]
        ttft = [float(r.get("ttft_ms", 0.0)) for r in records if r.get("ttft_ms") is not None]
        edge_queue = [float(r.get("edge_queue_ms", 0.0)) for r in records]
        arrival_lag = [float(r.get("arrival_lag_ms", 0.0)) for r in records]
        itl = []
        for record in records:
            commits = [float(value) for value in record.get("token_commit_offsets_ms", [])]
            itl.extend(max(0.0, right - left) for left, right in zip(commits, commits[1:]))
        total_tokens = sum(int(r.get("generated_tokens", 0)) for r in records)
        completed = [r for r in records if r.get("status", "completed") == "completed"]
        arrival_monotonic = [
            float(r["actual_arrival_monotonic_s"])
            for r in records
            if r.get("actual_arrival_monotonic_s") is not None
        ]
        completion_monotonic = [
            float(r["completion_monotonic_s"])
            for r in completed
            if r.get("completion_monotonic_s") is not None
        ]
        # The formal measurement window deliberately excludes worker/model
        # preparation.  It begins at the first request's actual dispatcher
        # arrival and ends when the final request completes.
        if arrival_monotonic and completion_monotonic:
            measurement_window_s = max(
                0.0, max(completion_monotonic) - min(arrival_monotonic)
            )
        else:
            measurement_window_s = 0.0
        summary = {
            "profile": self.args.profile,
            "enable_pipeline": bool(getattr(self.args, "enable_pipeline", True)),
            "enable_proactive_draft": bool(getattr(self.args, "enable_proactive_draft", True)),
            "num_drafts": int(self.args.num_drafts),
            "num_tasks": len(records),
            "completed_requests": len(completed),
            "completion_rate": float(len(completed) / len(records)) if records else 0.0,
            "measurement_window_s": float(measurement_window_s),
            # Retained as a backwards-compatible alias for downstream readers.
            "wallclock_s": float(measurement_window_s),
            "orchestration_wallclock_s": float(orchestration_wallclock_s),
            "total_generated_tokens": int(total_tokens),
            "system_tok_per_s": (
                float(total_tokens / measurement_window_s)
                if measurement_window_s > 0
                else 0.0
            ),
            "system_req_per_s": (
                float(len(completed) / measurement_window_s)
                if measurement_window_s > 0
                else 0.0
            ),
            "task_e2e_ms_avg": float(statistics.mean(task_e2e)) if task_e2e else 0.0,
            "task_e2e_ms_p50": self._percentile(task_e2e, 0.50),
            "task_e2e_ms_p90": self._percentile(task_e2e, 0.90),
            "task_e2e_ms_p95": self._percentile(task_e2e, 0.95),
            "task_e2e_ms_p99": self._percentile(task_e2e, 0.99),
            "ttft_ms_p50": self._percentile(ttft, 0.50),
            "ttft_ms_p95": self._percentile(ttft, 0.95),
            "ttft_ms_p99": self._percentile(ttft, 0.99),
            "itl_ms_p50": self._percentile(itl, 0.50),
            "itl_ms_p95": self._percentile(itl, 0.95),
            "itl_ms_p99": self._percentile(itl, 0.99),
            "edge_queue_ms_p95": self._percentile(edge_queue, 0.95),
            "arrival_lag_ms_p95": self._percentile(arrival_lag, 0.95),
            "prefill_http_ms_avg": float(statistics.mean(prefill_http)) if prefill_http else 0.0,
            "prefill_http_ms_p95": self._percentile(prefill_http, 0.95),
            "prefill_queue_ms_avg": float(statistics.mean(prefill_queue)) if prefill_queue else 0.0,
            "prefill_queue_ms_p95": self._percentile(prefill_queue, 0.95),
            "prefill_service_ms_avg": float(statistics.mean(prefill_service)) if prefill_service else 0.0,
            "prefill_service_ms_p95": self._percentile(prefill_service, 0.95),
            "prefill_chunks_avg": float(statistics.mean(prefill_chunks)) if prefill_chunks else 0.0,
            "verify_queue_ms_avg": float(statistics.mean(verify_queue)) if verify_queue else 0.0,
            "verify_queue_ms_p95": self._percentile(verify_queue, 0.95),
            "verify_service_ms_avg": float(statistics.mean(verify_service)) if verify_service else 0.0,
            "verify_service_ms_p95": self._percentile(verify_service, 0.95),
        }
        if getattr(self.args, "run_dir", None):
            summary_path = os.path.join(self.args.run_dir, "metrics", "summary.json")
        else:
            summary_path = os.path.join(self.args.exp_name, "edge_metrics_summary.json")
        if os.path.exists(summary_path) or os.path.islink(summary_path):
            raise FileExistsError(f"refusing to overwrite summary: {summary_path}")
        with open(summary_path, "x", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        self.color_print(f"[METRICS] wrote summary: {summary_path}", 2)

    def eval(self):
        torch.cuda.init()
        torch.multiprocessing.set_start_method("spawn", force=True)
        # Formal runs never delete prior metrics.  A fresh --run-dir has been
        # reserved by the metadata tool, and each worker creates a unique file.
        wallclock_start = time.monotonic()
        if self.args.arrival_mode == "poisson":
            self._eval_poisson()
        else:
            processes = []
            for proc_id in range(self.args.num_drafts):
                proc = Process(target=self.run_draft_process_http, args=(self.tokenizer, proc_id))
                proc.start()
                processes.append(proc)
            for proc in processes:
                proc.join()
            failed = [proc.pid for proc in processes if proc.exitcode not in (0, None)]
            if failed:
                raise RuntimeError(f"closed-loop edge workers failed: {failed}")
        self._merge_worker_artifacts()
        self._write_summary(max(0.0, time.monotonic() - wallclock_start))

    def _load_trace_prompts(self, trace: List[Dict[str, Any]]) -> Dict[int, str]:
        """Load the immutable rendered prompts paired with a formal trace.

        The workload builder saves prompts separately from the scheduling trace
        so every method uses identical input bytes.  Re-rendering MT-Bench at
        replay time would make a tokenizer/template upgrade silently alter the
        workload, even if the arrival-trace SHA remained unchanged.
        """
        prompt_path = Path(self.args.arrival_trace_in).parent / "prompts.jsonl"
        workload_manifest_path = prompt_path.parent / "manifest.json"
        if not prompt_path.is_file():
            raise ValueError(
                "Poisson replay requires the immutable prompts.jsonl next to "
                f"the arrival trace: {prompt_path}"
            )
        if not workload_manifest_path.is_file():
            raise ValueError(
                "Poisson replay requires the immutable workload manifest next to "
                f"the arrival trace: {workload_manifest_path}"
            )
        try:
            workload_manifest = json.loads(
                workload_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid workload manifest: {workload_manifest_path}"
            ) from exc
        actual_prompt_sha = sha256_file(prompt_path)
        expected_prompt_sha = workload_manifest.get("prompt_sha256")
        if expected_prompt_sha != actual_prompt_sha:
            raise ValueError(
                "prompt artifact SHA does not match the workload manifest: "
                f"expected={expected_prompt_sha!r}, actual={actual_prompt_sha}"
            )
        expected_trace_sha = workload_manifest.get("trace_sha256")
        actual_trace_sha = sha256_file(self.args.arrival_trace_in)
        if expected_trace_sha != actual_trace_sha:
            raise ValueError(
                "arrival trace SHA does not match the workload manifest: "
                f"expected={expected_trace_sha!r}, actual={actual_trace_sha}"
            )

        prompts: Dict[int, str] = {}
        with prompt_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in prompt artifact {prompt_path} line {line_number}"
                    ) from exc
                try:
                    arrival_index = int(payload["arrival_index"])
                    prompt = payload["prompt"]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid prompt record in {prompt_path} line {line_number}"
                    ) from exc
                if not isinstance(prompt, str):
                    raise ValueError(
                        f"prompt artifact {prompt_path} line {line_number} has a non-string prompt"
                    )
                if arrival_index in prompts:
                    raise ValueError(
                        f"prompt artifact {prompt_path} has duplicate arrival_index {arrival_index}"
                    )
                prompts[arrival_index] = prompt

        expected = {int(row["arrival_index"]) for row in trace}
        if not expected.issubset(prompts):
            missing = sorted(expected - set(prompts))
            raise ValueError(
                "prompt artifact does not match the trace "
                f"(missing={missing[:5]})"
            )
        return {arrival_index: prompts[arrival_index] for arrival_index in expected}

    def _eval_poisson(self) -> None:
        if not self.args.arrival_trace_in:
            raise ValueError("Poisson replay requires --arrival-trace-in; generate it before starting workers")
        trace = read_trace_jsonl(self.args.arrival_trace_in)
        if self.args.max_requests is not None:
            trace = validate_trace_rows(trace[: int(self.args.max_requests)])
        rendered_prompts = self._load_trace_prompts(trace)
        samples = self._load_samples()
        for row in trace:
            dataset_index = int(row["dataset_index"])
            client_id = int(row["client_id"])
            if dataset_index >= len(samples):
                raise ValueError(f"trace dataset_index {dataset_index} exceeds dataset size {len(samples)}")
            if client_id >= self.args.num_drafts:
                raise ValueError(
                    f"trace client_id {client_id} requires more than num_drafts={self.args.num_drafts}"
                )

        context = mp.get_context("spawn")
        task_queues = [context.Queue() for _ in range(self.args.num_drafts)]
        ready_queue = context.Queue()
        start_event = context.Event()
        run_start = context.Value("d", 0.0)
        processes = []
        for proc_id in range(self.args.num_drafts):
            proc = Process(
                target=self.run_draft_process_http,
                args=(self.tokenizer, proc_id, task_queues[proc_id], ready_queue, start_event, run_start),
            )
            proc.start()
            processes.append(proc)

        ready = set()
        deadline = time.monotonic() + float(self.args.worker_ready_timeout_s)
        while len(ready) < self.args.num_drafts:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("not all edge workers became ready before --worker-ready-timeout-s")
            try:
                message = ready_queue.get(timeout=min(remaining, 1.0))
            except Exception:
                failed = [proc.pid for proc in processes if proc.exitcode not in (0, None)]
                if failed:
                    raise RuntimeError(f"edge worker failed before readiness: {failed}")
                continue
            if message.get("state") == "error":
                raise RuntimeError(f"edge worker {message.get('proc_id')} failed: {message.get('error')}")
            if message.get("state") == "ready":
                ready.add(int(message["proc_id"]))

        with run_start.get_lock():
            run_start.value = time.monotonic()
        start_event.set()
        base = float(run_start.value)
        for row in trace:
            deadline = base + float(row["scheduled_offset_s"])
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(remaining, 0.05))
            actual = time.monotonic()
            client_id = int(row["client_id"])
            task_queues[client_id].put(
                {
                    "sample": samples[int(row["dataset_index"])],
                    "trace": row,
                    "prompt": rendered_prompts[int(row["arrival_index"])],
                    "actual_arrival_offset_s": max(0.0, actual - base),
                }
            )
        last_actual_arrival_monotonic = base
        for task_queue in task_queues:
            task_queue.put(None)
        if trace:
            # The last dispatcher timestamp is updated after each absolute
            # deadline.  It is the local definition of post-arrival drain,
            # not the earlier worker/model preparation time.
            last_actual_arrival_monotonic = actual
        drain_deadline = (
            last_actual_arrival_monotonic
            + float(self.args.post_arrival_drain_timeout_s)
        )
        while True:
            alive = [proc for proc in processes if proc.is_alive()]
            if not alive:
                break
            remaining = drain_deadline - time.monotonic()
            if remaining <= 0.0:
                # These are exactly the worker children spawned above for
                # this run.  Preserve their partial artifacts, stop them, and
                # fail closed so a timed-out run cannot be finalized.
                for proc in alive:
                    proc.terminate()
                for proc in alive:
                    proc.join(timeout=5.0)
                raise TimeoutError(
                    "Poisson post-arrival drain exceeded "
                    f"--post-arrival-drain-timeout-s={self.args.post_arrival_drain_timeout_s}"
                )
            for proc in alive:
                proc.join(timeout=min(0.1, remaining))
        failed = [proc.pid for proc in processes if proc.exitcode not in (0, None)]
        if failed:
            raise RuntimeError(f"Poisson edge workers failed: {failed}")

    def _merge_worker_artifacts(self) -> None:
        if not getattr(self.args, "run_dir", None):
            return
        pairs = (
            (self._metric_input_paths(), os.path.join(self.args.run_dir, "metrics", "requests.jsonl")),
            (
                sorted(glob.glob(os.path.join(self.args.run_dir, "outputs", "completions_proc*.jsonl"))),
                os.path.join(self.args.run_dir, "outputs", "completions.jsonl"),
            ),
        )
        for source_paths, destination in pairs:
            if os.path.exists(destination) or os.path.islink(destination):
                raise FileExistsError(f"refusing to overwrite canonical artifact: {destination}")
            records: List[Dict[str, Any]] = []
            for source_path in source_paths:
                with open(source_path, "r", encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"invalid worker JSONL {source_path}:{line_number}"
                            ) from exc
                        if not isinstance(record, dict):
                            raise ValueError(
                                f"worker JSONL record must be an object: {source_path}:{line_number}"
                            )
                        records.append(record)

            # Open-loop workers finish independently.  Canonical artifacts
            # must still line up with the shared trace rather than merely with
            # the arbitrary worker-file concatenation order.
            has_arrival_index = ["arrival_index" in record for record in records]
            if any(has_arrival_index):
                if not all(has_arrival_index):
                    raise ValueError("cannot merge a mix of traced and untraced worker records")
                by_arrival: Dict[int, Dict[str, Any]] = {}
                for record in records:
                    value = record["arrival_index"]
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError(f"invalid arrival_index in worker artifact: {value!r}")
                    if value in by_arrival:
                        raise ValueError(f"duplicate arrival_index in worker artifacts: {value}")
                    by_arrival[value] = record
                expected = list(range(len(by_arrival)))
                if sorted(by_arrival) != expected:
                    raise ValueError("worker artifacts do not cover a contiguous leading arrival trace")
                records = [by_arrival[index] for index in expected]
            else:
                # A closed-loop pilot has no offered-arrival order, but its
                # static round-robin data partition is deterministic.  Keep
                # canonical artifacts in that order rather than worker-file
                # order, and reject duplicates rather than silently choosing.
                has_dataset_index = ["dataset_index" in record for record in records]
                if any(has_dataset_index):
                    if not all(has_dataset_index):
                        raise ValueError(
                            "cannot merge a mix of indexed and unindexed closed-loop records"
                        )
                    by_dataset: Dict[int, Dict[str, Any]] = {}
                    for record in records:
                        value = record["dataset_index"]
                        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                            raise ValueError(
                                f"invalid dataset_index in worker artifact: {value!r}"
                            )
                        if value in by_dataset:
                            raise ValueError(
                                f"duplicate dataset_index in worker artifacts: {value}"
                            )
                        by_dataset[value] = record
                    records = [by_dataset[index] for index in sorted(by_dataset)]
            with open(destination, "x", encoding="utf-8") as output:
                for record in records:
                    output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                    output.write("\n")

    def _resolve_data_file(self) -> str:
        if self.args.dataset == "humaneval":
            default_file = "humaneval.jsonl"
        elif self.args.dataset == "gsm8k":
            default_file = "gsm8k.jsonl"
        elif self.args.dataset == "mt_bench":
            default_file = "mt_bench.jsonl"
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        if os.path.isdir(self.args.data_path):
            return os.path.join(self.args.data_path, default_file)
        return self.args.data_path

    def _load_samples(self) -> List[Dict[str, Any]]:
        data_file = self._resolve_data_file()
        with open(data_file, "r", encoding="utf-8") as handle:
            samples = [json.loads(line) for line in handle if line.strip()]
        if not samples:
            raise ValueError(f"dataset is empty: {data_file}")
        return samples

    def _use_chat_template(self) -> bool:
        configured = getattr(self.args, "use_chat_template", None)
        if configured is not None:
            return bool(configured)
        identity = " ".join(
            str(value).lower()
            for value in (self.args.draft_model, self.args.target_model, self.args.tokenizer_model)
        )
        return "qwen" in identity

    def _render_mt_bench_prompt(self, tokenizer, turn: str) -> str:
        if not self._use_chat_template():
            return turn.strip()
        messages = [{"role": "user", "content": turn.strip()}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    @torch.no_grad()
    def run_draft_process_http(
        self,
        tokenizer,
        proc_id: int,
        work_queue=None,
        ready_queue=None,
        start_event=None,
        run_start=None,
    ):
        gpu_id = (proc_id % max(1, self.args.edge_gpus)) + self.args.edge_gpu_start
        device = f"cuda:{gpu_id}"
        self.color_print(f"[Edge {proc_id}] loading draft model on {device}", 3)

        draft_model = self._load_draft_model_for_service(
            self.args.draft_model,
            device,
        )

        try:
            client = EdgeClient(self.args.server_url, timeout=self.args.request_timeout)
            health = client.health()
            if health.get("status") not in {"ok", "healthy"}:
                raise RuntimeError(f"Cloud service health check failed: {health}")
            self._prepare_worker_outputs(proc_id)
        except Exception as exc:
            if ready_queue is not None:
                ready_queue.put({"state": "error", "proc_id": proc_id, "error": repr(exc)})
            raise

        seed_everything(int(self.args.seed) + proc_id)
        if work_queue is None:
            all_samples = self._load_samples()
            static_items = [
                {
                    "sample": sample,
                    "local_index": index,
                    "dataset_index": index,
                    "trace": None,
                }
                for index, sample in enumerate(all_samples)
                if index % self.args.num_drafts == proc_id
            ][:int(self.args.max_tasks_per_draft)]
            work_items = iter(static_items)
        else:
            assert ready_queue is not None and start_event is not None and run_start is not None
            ready_queue.put({"state": "ready", "proc_id": proc_id})
            if not start_event.wait(timeout=float(self.args.worker_ready_timeout_s)):
                raise TimeoutError("Poisson replay start signal timed out")

            def queued_items():
                local_index = 0
                while True:
                    item = work_queue.get()
                    if item is None:
                        return
                    item["local_index"] = local_index
                    local_index += 1
                    yield item

            work_items = queued_items()

        for work_item in work_items:
            idx = int(work_item["local_index"])
            sample = work_item["sample"]
            arrival = work_item.get("trace")
            saved_prompt = work_item.get("prompt")
            actual_arrival_offset_s = work_item.get("actual_arrival_offset_s")
            if actual_arrival_offset_s is None:
                actual_arrival_offset_s = max(0.0, time.monotonic() - float(run_start.value)) if run_start is not None else 0.0
            service_start_monotonic = time.monotonic()
            if run_start is not None and work_queue is not None:
                task_arrival_monotonic = float(run_start.value) + float(actual_arrival_offset_s)
            else:
                task_arrival_monotonic = service_start_monotonic
            session_id = client.init_session()
            approx_model_cache = KVCacheModel(
                draft_model, self.args.temp, self.args.top_k, self.args.top_p
            )
            approx_model_cache.vocab_size = len(tokenizer)

            if saved_prompt is not None:
                input_text = str(saved_prompt)
                task_id = str(arrival["task_id"])
            elif self.args.dataset == "gsm8k":
                input_text = self._preprocess_gsm8k(sample["question"].strip())
                task_id = sample.get("task_id", f"gsm8k-{proc_id}-{idx}")
            elif self.args.dataset == "humaneval":
                input_text = sample["prompt"].strip()
                task_id = sample.get("task_id", f"humaneval-{proc_id}-{idx}")
            else:
                input_text = self._render_mt_bench_prompt(tokenizer, sample["turns"][0])
                task_id = str(sample.get("task_id", f"mtbench-{proc_id}-{idx}"))

            request_id = (
                f"arrival-{int(arrival['arrival_index'])}"
                if arrival is not None
                else f"closed-{proc_id}-{idx}-{task_id}"
            )
            arrival_fields: Dict[str, Any] = {}
            if arrival is not None:
                arrival_fields = {
                    "arrival_index": int(arrival["arrival_index"]),
                    "dataset_index": int(arrival["dataset_index"]),
                    "client_id": int(arrival["client_id"]),
                }
            elif work_item.get("dataset_index") is not None:
                # Closed-loop calibration is not a Poisson arrival trace, but
                # its deterministic static dataset partition must still be
                # auditable and sortable in canonical artifacts.
                arrival_fields = {"dataset_index": int(work_item["dataset_index"])}

            # input_text = 'def fib(n'  # for debug
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(draft_model.device)
            prefix = input_ids.clone()
            max_len = input_ids.shape[1] + self.args.max_tokens
            cloud_measurement_start_monotonic = None
            cloud_measurement_end_monotonic = None

            def observe_cloud_measurement_window(response: Dict[str, Any]) -> None:
                """Collect cloud-host-local bounds without cross-host math."""
                nonlocal cloud_measurement_start_monotonic
                nonlocal cloud_measurement_end_monotonic
                start = response.get("server_enqueue_monotonic_s")
                end = response.get("server_completed_monotonic_s")
                if start is not None:
                    start = float(start)
                    cloud_measurement_start_monotonic = (
                        start
                        if cloud_measurement_start_monotonic is None
                        else min(cloud_measurement_start_monotonic, start)
                    )
                if end is not None:
                    end = float(end)
                    cloud_measurement_end_monotonic = (
                        end
                        if cloud_measurement_end_monotonic is None
                        else max(cloud_measurement_end_monotonic, end)
                    )

            prefill_http_start = time.perf_counter()
            prefill_resp = client.prefill(
                session_id=session_id,
                task_id=task_id,
                draft_output=prefix[0].tolist(),
                prefix_len=prefix.shape[1],
                lag=0.0,
                current_time=time.monotonic(),
            )
            prefill_http_ms = max(
                0.0, (time.perf_counter() - prefill_http_start) * 1000.0
            )
            if "error" in prefill_resp:
                raise RuntimeError(
                    f"cloud prefill rejected: {prefill_resp['error']}: "
                    f"{prefill_resp.get('detail', '')}"
                )
            if prefill_resp.get("status") != "prefill_ok":
                raise RuntimeError(f"prefill failed: {prefill_resp}")
            observe_cloud_measurement_window(prefill_resp)

            task_start = task_arrival_monotonic
            pipeline_enabled = bool(getattr(self.args, "enable_pipeline", True))
            proactive_enabled = bool(getattr(self.args, "enable_proactive_draft", True))
            final_token = None
            reused_pending_tokens: List[int] = []
            current_gamma = int(self.args.gamma)
            verify_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            last_transport_rtt = 0.0
            reuse_hit_rounds = 0
            reuse_miss_rounds = 0
            reuse_miss_not_full_accept_rounds = 0
            sum_round_ms = 0.0
            sum_draft_ms = 0.0
            sum_wait_ms = 0.0
            sum_http_total_ms = 0.0
            sum_cloud_total_ms = 0.0
            sum_verify_ms = 0.0
            sum_verify_queue_ms = 0.0
            sum_verify_service_ms = 0.0
            sum_transport_rtt_ms = 0.0
            accepted_total = 0
            drafted_total = 0
            rounds = 0
            first_token_monotonic = None
            token_commit_offsets_ms: List[float] = []
            while prefix.shape[1] < max_len:
                prefix_len = prefix.shape[1]
                round_start = time.monotonic()
                req_gamma = current_gamma if pipeline_enabled else int(self.args.gamma)

                # 若上一轮复用 token 不足 gamma，这里补齐到 gamma 后再发起验证。
                if reused_pending_tokens:
                    reuse_count = min(len(reused_pending_tokens), req_gamma)
                    reuse_tensor = torch.tensor(
                        [reused_pending_tokens[:reuse_count]],
                        device=prefix.device,
                        dtype=prefix.dtype,
                    )
                    x = torch.cat((prefix, reuse_tensor), dim=1)
                    if reuse_count < req_gamma:
                        x = approx_model_cache.generate(x, req_gamma - reuse_count)
                else:
                    x = approx_model_cache.generate(prefix, req_gamma)

                has_bridge_token = pipeline_enabled and (final_token is not None)
                if pipeline_enabled:
                    pending_tokens = x[0, prefix_len:prefix_len + req_gamma].tolist()
                    if has_bridge_token:
                        # 发送：上一轮 final_token + 本轮 gamma 个 draft token
                        payload_tokens = [final_token] + pending_tokens
                    else:
                        # 首轮无上一轮 final_token，仅发送本轮 gamma 个 draft token
                        payload_tokens = pending_tokens
                else:
                    # vanilla verify: send full prefix + gamma draft tokens.
                    payload_tokens = x[0, :prefix_len + req_gamma].tolist()
                if getattr(self.args, "debug_verify_tokens", False):
                    debug_tail = 16
                    self.color_print(
                        f"[VERIFY-EDGE-SEND][pid={proc_id}][session={session_id}] "
                        f"prefix_len={prefix_len} has_bridge_token={has_bridge_token} "
                        f"tail_ids={payload_tokens[-debug_tail:]}",
                        3,
                    )

                verify_future = verify_executor.submit(
                    client.verify,
                    session_id=session_id,
                    task_id=task_id,
                    draft_output=payload_tokens,
                    prefix_len=prefix_len,
                    lag=time.monotonic() - round_start,
                    current_time=time.monotonic(),
                    gamma=req_gamma,
                    transport_rtt=last_transport_rtt,
                    tail_only=pipeline_enabled,
                    has_bridge_token=has_bridge_token,
                )
                draft_elapsed = max(0.0, time.monotonic() - round_start)
                if getattr(self.args, "debug_pipeline", False):
                    self.color_print(
                        f"[PIPELINE-EDGE][pid={proc_id}][session={session_id}] "
                        f"send req_gamma={req_gamma} draft_ms={draft_elapsed*1000:.2f} "
                        f"transport_rtt_prev={last_transport_rtt*1000:.2f}ms "
                        f"prefix_len={prefix_len} payload_tokens={len(payload_tokens)}",
                        3,
                    )

                # 验证等待期间持续 draft，最多缓存 gamma+1（bridge + gamma）个 token。
                overlap_tokens: List[int] = []
                overlap_prefix = x
                max_overlap_tokens = req_gamma + 1
                wait_start = time.monotonic()
                wait_draft_steps = 0
                if getattr(self.args, "debug_verify_tokens", False):
                    self.color_print(
                        f"[VERIFY-EDGE-WAIT-START][pid={proc_id}][session={session_id}] "
                        f"prefix_len={prefix_len} max_overlap_tokens={max_overlap_tokens}",
                        3,
                    )
                if proactive_enabled:
                    while len(overlap_tokens) < max_overlap_tokens and not verify_future.done():
                        overlap_prefix = approx_model_cache.generate(overlap_prefix, 1)
                        overlap_tokens.append(int(overlap_prefix[0, -1].item()))
                        wait_draft_steps += 1
                        if getattr(self.args, "debug_verify_tokens", False) and (
                            wait_draft_steps == 1
                            or wait_draft_steps % 8 == 0
                            or wait_draft_steps == max_overlap_tokens
                        ):
                            self.color_print(
                                f"[VERIFY-EDGE-WAIT-DRAFT][pid={proc_id}][session={session_id}] "
                                f"drafted_while_wait={wait_draft_steps} latest_token={overlap_tokens[-1]} "
                                f"verify_done={verify_future.done()}",
                                3,
                            )
                else:
                    while not verify_future.done():
                        time.sleep(0.0005)

                if getattr(self.args, "debug_verify_tokens", False):
                    wait_ms = (time.monotonic() - wait_start) * 1000.0
                    debug_tail = 8
                    self.color_print(
                        f"[VERIFY-EDGE-WAIT-END][pid={proc_id}][session={session_id}] "
                        f"wait_ms={wait_ms:.2f} drafted_while_wait={wait_draft_steps} "
                        f"verify_done={verify_future.done()} overlap_tail={overlap_tokens[-debug_tail:]}",
                        3,
                    )

                verify_resp, measured_http_total = verify_future.result()
                if "error" in verify_resp:
                    raise RuntimeError(
                        f"cloud verify rejected: {verify_resp['error']}: "
                        f"{verify_resp.get('detail', '')}"
                    )
                observe_cloud_measurement_window(verify_resp)
                verify_ms = float(verify_resp.get("verify_ms", 0.0))
                verify_queue_ms = float(verify_resp.get("verify_queue_ms", 0.0))
                verify_service_ms = float(verify_resp.get("verify_service_ms", 0.0))
                cloud_total_ms = float(verify_resp.get("cloud_total_ms", 0.0))
                # A purer transport estimate: subtract cloud-side service time from end-to-end HTTP time.
                last_transport_rtt = max(0.0, measured_http_total - cloud_total_ms / 1000.0)
                round_elapsed = max(0.0, time.monotonic() - round_start)
                wait_elapsed_ms = max(0.0, (time.monotonic() - wait_start) * 1000.0)
                sum_round_ms += round_elapsed * 1000.0
                sum_draft_ms += draft_elapsed * 1000.0
                sum_wait_ms += wait_elapsed_ms
                sum_http_total_ms += measured_http_total * 1000.0
                sum_cloud_total_ms += cloud_total_ms
                sum_verify_ms += verify_ms
                sum_verify_queue_ms += verify_queue_ms
                sum_verify_service_ms += verify_service_ms
                sum_transport_rtt_ms += last_transport_rtt * 1000.0
                rounds += 1

                accepted = int(verify_resp["accepted"])
                accepted_cnt = accepted - prefix_len
                accepted_total += max(0, accepted_cnt)
                drafted_total += max(1, req_gamma)
                final_token = int(verify_resp["final_token"])
                committed_now = max(0, accepted_cnt) + 1
                commit_time = time.monotonic()
                if committed_now > 0:
                    if first_token_monotonic is None:
                        first_token_monotonic = commit_time
                    token_commit_offsets_ms.extend(
                        [max(0.0, (commit_time - task_arrival_monotonic) * 1000.0)]
                        * committed_now
                    )
                if pipeline_enabled and "suggested_gamma" in verify_resp:
                    current_gamma = int(verify_resp["suggested_gamma"])
                if getattr(self.args, "debug_pipeline", False):
                    self.color_print(
                        f"[PIPELINE-EDGE][pid={proc_id}][session={session_id}] "
                        f"recv accepted={accepted_cnt}/{req_gamma} round_ms={round_elapsed*1000:.2f} "
                        f"transport_rtt={last_transport_rtt*1000:.2f}ms "
                        f"http_total_ms={measured_http_total*1000:.2f} cloud_total_ms={cloud_total_ms:.2f} "
                        f"verify_ms={verify_ms:.2f} "
                        f"suggested_gamma={current_gamma}",
                        3,
                    )
                if getattr(self.args, "debug_verify_tokens", False):
                    self.color_print(
                        f"[VERIFY-EDGE-RESP][pid={proc_id}][session={session_id}] "
                        f"accepted={accepted} final_token={final_token} req_gamma={req_gamma} suggested_gamma={current_gamma} "
                        f"drafted_while_wait={wait_draft_steps}",
                        3,
                    )
                final_token_tensor = torch.tensor([[final_token]], device=x.device, dtype=x.dtype)

                prefix = torch.cat((x[:, :accepted], final_token_tensor), dim=1)
                prefix, hit_eos = self._truncate_at_eos(prefix, tokenizer.eos_token_id, prefix_len)
                approx_model_cache.rollback(accepted)
                reused_pending_tokens = []

                if proactive_enabled and accepted_cnt == req_gamma and overlap_tokens:
                    if overlap_tokens[0] == final_token:
                        reused_pending_tokens = overlap_tokens[1 : 1 + req_gamma]
                        reuse_hit_rounds += 1
                        if getattr(self.args, "debug_verify_tokens", False):
                            self.color_print(
                                f"[VERIFY-EDGE-REUSE-HIT][pid={proc_id}][session={session_id}] "
                                f"final_token={final_token} overlap_first={overlap_tokens[0]} "
                                f"reused_count={len(reused_pending_tokens)}",
                                2,
                            )
                    elif getattr(self.args, "debug_verify_tokens", False):
                        reuse_miss_rounds += 1
                        self.color_print(
                            f"[VERIFY-EDGE-DROP][pid={proc_id}][session={session_id}] "
                            f"final_token={final_token} overlap_first={overlap_tokens[0]}",
                            3,
                        )
                elif proactive_enabled and accepted_cnt == req_gamma:
                    reuse_miss_rounds += 1
                    if getattr(self.args, "debug_verify_tokens", False):
                        self.color_print(
                            f"[VERIFY-EDGE-REUSE-MISS][pid={proc_id}][session={session_id}] "
                            f"reason=overlap_empty final_token={final_token}",
                            3,
                        )
                elif proactive_enabled:
                    reuse_miss_not_full_accept_rounds += 1
                    if getattr(self.args, "debug_verify_tokens", False):
                        self.color_print(
                            f"[VERIFY-EDGE-REUSE-SKIP][pid={proc_id}][session={session_id}] "
                            f"reason=not_full_accept accepted_cnt={accepted_cnt} gamma={req_gamma}",
                            3,
                        )

                if hit_eos:
                    if getattr(self.args, "debug_verify_tokens", False):
                        self.color_print(
                            f"[VERIFY-EDGE-STOP][pid={proc_id}][session={session_id}] reason=eos_token",
                            2,
                        )
                    break
                if self._is_degenerate_repeat(prefix, input_ids.shape[1]):
                    self.color_print(
                        f"[VERIFY-EDGE-STOP][pid={proc_id}][session={session_id}] reason=degenerate_repeat",
                        2,
                    )
                    break
                if self.args.dataset == "humaneval":
                    current_text = tokenizer.decode(
                        prefix[0, input_ids.shape[1]:], skip_special_tokens=True
                    )
                    truncated_text, marker_hit = self._truncate_on_humaneval_markers(current_text)
                    if marker_hit:
                        trunc_ids = tokenizer.encode(truncated_text, add_special_tokens=False)
                        trunc_tensor = torch.tensor(
                            [trunc_ids], device=prefix.device, dtype=prefix.dtype
                        )
                        prefix = torch.cat((input_ids, trunc_tensor), dim=1)
                        self.color_print(
                            f"[VERIFY-EDGE-STOP][pid={proc_id}][session={session_id}] reason=humaneval_stop_marker",
                            2,
                        )
                        break
                if self.args.dataset == "gsm8k":
                    current_text = tokenizer.decode(
                        prefix[0, input_ids.shape[1]:], skip_special_tokens=True
                    )
                    truncated_text, marker_hit = self._truncate_on_gsm8k_markers(current_text)
                    if marker_hit:
                        trunc_ids = tokenizer.encode(truncated_text, add_special_tokens=False)
                        trunc_tensor = torch.tensor(
                            [trunc_ids], device=prefix.device, dtype=prefix.dtype
                        )
                        prefix = torch.cat((input_ids, trunc_tensor), dim=1)
                        self.color_print(
                            f"[VERIFY-EDGE-STOP][pid={proc_id}][session={session_id}] reason=gsm8k_next_question_marker",
                            2,
                        )
                        break

            verify_executor.shutdown(wait=True)

            generated_text = tokenizer.decode(
                prefix[0, input_ids.shape[1]:], skip_special_tokens=True
            )
            total_reuse_rounds = reuse_hit_rounds + reuse_miss_rounds
            reuse_hit_rate = (reuse_hit_rounds / total_reuse_rounds) if total_reuse_rounds > 0 else 0.0
            self.color_print(
                f"[Edge {proc_id}] task {task_id} reuse stats: "
                f"hit={reuse_hit_rounds} miss={reuse_miss_rounds} "
                f"skip_not_full_accept={reuse_miss_not_full_accept_rounds} "
                f"hit_rate={reuse_hit_rate:.3f}",
                2,
            )
            self.color_print(
                f"[Edge {proc_id}] finished task {task_id}, generated {prefix.shape[1] - input_ids.shape[1]} tokens",
                2,
            )
            self.color_print(f"[Edge {proc_id}] task {task_id} output:\n{generated_text}", 2)

            generated_tokens = int(prefix.shape[1] - input_ids.shape[1])
            completion_monotonic = time.monotonic()
            task_e2e_ms = max(0.0, (completion_monotonic - task_start) * 1000.0)
            per_task = {
                "schema_version": 1,
                "status": "completed",
                "request_id": request_id,
                "task_id": task_id,
                "proc_id": int(proc_id),
                **arrival_fields,
                "profile": self.args.profile,
                "enable_pipeline": pipeline_enabled,
                "enable_proactive_draft": proactive_enabled,
                "generated_tokens": generated_tokens,
                "prefill_http_ms": prefill_http_ms,
                "prefill_queue_ms": float(prefill_resp.get("prefill_queue_ms", 0.0)),
                "prefill_service_ms": float(prefill_resp.get("prefill_service_ms", 0.0)),
                "prefill_chunks": int(prefill_resp.get("prefill_chunks", 1)),
                "scheduled_arrival_offset_s": float(arrival["scheduled_offset_s"]) if arrival is not None else None,
                "actual_arrival_offset_s": float(actual_arrival_offset_s),
                "arrival_lag_ms": (
                    max(0.0, (float(actual_arrival_offset_s) - float(arrival["scheduled_offset_s"])) * 1000.0)
                    if arrival is not None
                    else 0.0
                ),
                "edge_queue_ms": max(0.0, (service_start_monotonic - task_arrival_monotonic) * 1000.0),
                "actual_arrival_monotonic_s": float(task_arrival_monotonic),
                "service_start_monotonic_s": float(service_start_monotonic),
                "ttft_ms": (
                    max(0.0, (first_token_monotonic - task_arrival_monotonic) * 1000.0)
                    if first_token_monotonic is not None
                    else None
                ),
                "token_commit_offsets_ms": token_commit_offsets_ms,
                "task_e2e_ms": task_e2e_ms,
                "completion_monotonic_s": float(completion_monotonic),
                # Local to node2, for cloud GPU sample cropping only.  E2E
                # remains calculated entirely in node1's monotonic domain.
                "cloud_measurement_start_monotonic_s": cloud_measurement_start_monotonic,
                "cloud_measurement_end_monotonic_s": cloud_measurement_end_monotonic,
                "tok_per_s_task": float(generated_tokens / (task_e2e_ms / 1000.0)) if task_e2e_ms > 0 else 0.0,
                "rounds": int(rounds),
                "avg_round_ms": float(sum_round_ms / rounds) if rounds > 0 else 0.0,
                "avg_draft_ms": float(sum_draft_ms / rounds) if rounds > 0 else 0.0,
                "avg_wait_ms": float(sum_wait_ms / rounds) if rounds > 0 else 0.0,
                "avg_http_total_ms": float(sum_http_total_ms / rounds) if rounds > 0 else 0.0,
                "avg_cloud_total_ms": float(sum_cloud_total_ms / rounds) if rounds > 0 else 0.0,
                "avg_verify_ms": float(sum_verify_ms / rounds) if rounds > 0 else 0.0,
                "avg_verify_queue_ms": float(sum_verify_queue_ms / rounds) if rounds > 0 else 0.0,
                "avg_verify_service_ms": float(sum_verify_service_ms / rounds) if rounds > 0 else 0.0,
                "avg_transport_rtt_ms": float(sum_transport_rtt_ms / rounds) if rounds > 0 else 0.0,
                "accepted_total": int(accepted_total),
                "drafted_total": int(drafted_total),
                "accept_rate": float(accepted_total / drafted_total) if drafted_total > 0 else 0.0,
                "reuse_hit_rounds": int(reuse_hit_rounds),
                "reuse_miss_rounds": int(reuse_miss_rounds),
                "reuse_skip_not_full_accept_rounds": int(reuse_miss_not_full_accept_rounds),
            }
            self._append_jsonl(self._metrics_path(proc_id), per_task)
            self._append_jsonl(
                self._outputs_path(proc_id),
                {
                    "schema_version": 1,
                    "status": "completed",
                    "request_id": request_id,
                    "task_id": task_id,
                    "proc_id": int(proc_id),
                    **arrival_fields,
                    "text": generated_text,
                    "generated_tokens": generated_tokens,
                },
            )


def parse_edge_arguments() -> argparse.Namespace:
    edge_parser = argparse.ArgumentParser(add_help=False)
    edge_parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8001")
    edge_parser.add_argument("--request_timeout", type=float, default=30.0)
    edge_parser.add_argument("--edge_gpu_start", type=int, default=0)
    edge_parser.add_argument("--edge_gpus", type=int, default=1)

    edge_args, remaining = edge_parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        base_args = parse_arguments()
    finally:
        sys.argv = original_argv

    base_args.server_url = edge_args.server_url
    base_args.request_timeout = edge_args.request_timeout
    base_args.edge_gpu_start = edge_args.edge_gpu_start
    base_args.edge_gpus = edge_args.edge_gpus
    return base_args


if __name__ == "__main__":
    args = parse_edge_arguments()
    runner = EdgeRunner(args)
    runner.eval()
