"""Production control-loop ordering for third-round prediction."""
from collections import defaultdict
import ast
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import unittest

from src.fastsd_scheduler import (
    TaskLengthClassifier, WorkItem, length_category, plan_iteration, predict_next_plan,
)
from src.scheduler_runtime import (
    SchedulerQueues, ShutdownRequested, TargetSchedulerRuntime,
)


class SubmissionOrderingTests(unittest.TestCase):
    def test_ingress_then_submit_current_batch_before_one_prediction(self):
        events = []

        class Executor:
            def poll_result(self):
                return None

            def busy(self):
                return False

            def abort(self):
                events.append("abort")

        polls = [0]
        builds = [0]

        def poll(_timeout):
            polls[0] += 1
            events.append("ingress")
            if polls[0] > 1:
                raise ShutdownRequested
            return 1

        def build():
            builds[0] += 1
            events.append("build")
            return SimpleNamespace(selected_work_ids=["A"] if builds[0] == 1 else [])

        metrics = defaultdict(int)
        runtime = TargetSchedulerRuntime(
            executor=Executor(), queues=SchedulerQueues(metrics),
            poll_ingress=poll, build_plan=build,
            submit_plan=lambda _plan: events.append("submit"),
            apply_outcome=lambda _outcome: None,
            refresh_prefetch=lambda: events.append("prediction"), metrics=metrics,
        )
        runtime.run()
        self.assertEqual(events[:4], ["ingress", "build", "submit", "prediction"])
        self.assertEqual(events.count("prediction"), 1)
        self.assertNotIn("abort", events)


def engine_closure(name, namespace):
    """Execute an actual engine closure with lightweight dependency bindings.

    Importing the complete model engine would require CUDA/model packages.
    AST extraction keeps these integration checks on the production body.
    """
    source = Path(__file__).resolve().parents[1] / "src" / "engine.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


class EnginePredictionIntegrationTests(unittest.TestCase):
    def environment(self, *items):
        queues = SchedulerQueues(defaultdict(int))
        for item in items:
            queues.enqueue(item)
        sorted_calls, submitted, diagnostics = [], [], []
        classifier = TaskLengthClassifier()
        metrics = defaultdict(int)
        namespace = dict(
            self=SimpleNamespace(args=SimpleNamespace(token_budget=8, batch_size=1,
                min_prefill_chunk_tokens=4, prefill_chunk_quantum=16,
                scheduler_policy="legacy", scheduler_max_wait_s=1, kv_prefetch=True)),
            time=SimpleNamespace(perf_counter=lambda: 1., monotonic=lambda: 50.,
                                 monotonic_ns=lambda: 50000),
            task_queues=queues, length_classifier=classifier,
            _length_category=length_category, scheduler_state={"wrr_cursor": 5, "current_cycle": 0},
            accept_stats_snapshot=lambda: {}, candidate_value=lambda *_args: 0,
            in_flight={"job": None, "plan": None}, scheduler_metrics=metrics,
            DIAG=SimpleNamespace(context=SimpleNamespace(),
                                 emit=lambda event, **data: diagnostics.append((event, data))),
            _sort_queues_if_dirty=lambda *args: sorted_calls.append(args),
            plan_iteration=plan_iteration, predict_next_plan=predict_next_plan,
            prefetch_enabled=True, last_prediction={"signature": None},
            residency=SimpleNamespace(protected_snapshot=lambda: set()),
            prefetch_coordinator=SimpleNamespace(refresh=lambda pids: submitted.append(list(pids))),
        )
        engine_closure("build_plan", namespace)
        engine_closure("refresh_prefetch", namespace)
        return namespace, sorted_calls, submitted, diagnostics

    @staticmethod
    def item(pid, tokens=8, category="short"):
        return WorkItem(pid, pid, "verify", category, total_tokens=tokens,
                        request={"task_type": "verify", "proc_id": pid, "W_i": 0})

    def test_idle_lookahead_does_not_sort_or_change_real_metadata(self):
        stale = self.item("stale", tokens=40, category="short")
        env, sorted_calls, _, _ = self.environment(stale)
        before = asdict(stale)
        prediction = env["build_plan"](diagnostic_kind="lookahead", planning_time=50)
        self.assertEqual(sorted_calls, [])
        self.assertEqual(asdict(stale), before)
        self.assertTrue(env["task_queues"].dirty)
        self.assertEqual(prediction.verify_slices[0].category, "mid")

    def test_complete_prediction_uses_next_cursor_before_filter_and_dedups(self):
        a, b, c = self.item("A"), self.item("B"), self.item("C", tokens=40, category="mid")
        env, sorted_calls, submitted, diagnostics = self.environment(a, b, c)
        current = plan_iteration(env["task_queues"].snapshot(), 8, wrr_cursor=5,
                                 max_num_seqs=1, min_prefill_chunk_tokens=4)
        a.state = "reserved"
        env["in_flight"].update(plan=current,
            job=SimpleNamespace(plan=current, verify_applied=False, progress={}, completed=set()))
        env["residency"] = SimpleNamespace(protected_snapshot=lambda: {"A"})
        before = [asdict(item) for item in (a, b, c)]
        env["refresh_prefetch"]()
        env["refresh_prefetch"]()
        self.assertEqual(submitted, [["C"]])
        self.assertEqual(sorted_calls, [])
        self.assertEqual([asdict(item) for item in (a, b, c)], before)
        prediction_events = [row for name, row in diagnostics if name == "prefetch_prediction"]
        self.assertEqual(len(prediction_events), 1)
        self.assertEqual(prediction_events[0]["ordered_verify_pids"], ["C"])
        self.assertEqual(env["scheduler_metrics"]["prefetch_prediction_versions"], 1)

    def test_protected_residual_spends_slot_and_phase_result_refreshes(self):
        a, b = self.item("A", tokens=40), self.item("B")
        env, _, submitted, diagnostics = self.environment(a, b)
        current = plan_iteration(env["task_queues"].snapshot(), 8, wrr_cursor=5,
                                 max_num_seqs=1, min_prefill_chunk_tokens=4)
        a.state = "reserved"
        job = SimpleNamespace(plan=current, verify_applied=False, progress={}, completed=set())
        env["in_flight"].update(plan=current, job=job)
        env["residency"] = SimpleNamespace(protected_snapshot=lambda: {"A"})
        env["refresh_prefetch"]()
        self.assertEqual(submitted, [[]])
        first = [row for name, row in diagnostics if name == "prefetch_prediction"][-1]
        self.assertEqual(first["ordered_verify_pids"], ["A"])
        a.cursor = a.accepted_so_far = 8
        job.verify_applied, job.progress = True, {"A": 8}
        env["refresh_prefetch"]()
        second = [row for name, row in diagnostics if name == "prefetch_prediction"][-1]
        self.assertEqual(second["ordered_verify_pids"], ["A"])
        self.assertEqual(second["assumptions"], {})
        self.assertEqual(a.cursor, 8)
        self.assertEqual(env["scheduler_metrics"]["prefetch_prediction_versions"], 2)


if __name__ == "__main__":
    unittest.main()
