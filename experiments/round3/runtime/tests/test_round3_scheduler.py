"""Third-round WRR visits, commit accounting and next-state prediction."""

import copy
from dataclasses import asdict
import math
import unittest

from src.fastsd_scheduler import (
    WorkItem, TaskLengthClassifier, compute_priority_score, prefill_priority_score,
    verify_priority_score, plan_iteration, project_after_plan, predict_next_plan,
    commit_admission_plan, reserve_admission_plan,
)


class Round3SchedulerTests(unittest.TestCase):
    def item(self, pid, kind="verify", category="short", tokens=8, **kwargs):
        request = {"proc_id": pid, "task_type": kind, "W_i": 0, "gamma": tokens}
        request.update(kwargs.pop("request", {}))
        return WorkItem(pid, pid, kind, category, request=request,
                        total_tokens=tokens, **kwargs)

    def queues(self, *items):
        result = {kind: {category: [] for category in ("short", "mid", "long")}
                  for kind in ("verify", "prefill")}
        for item in items:
            result[item.task_type][item.category].append(item)
        return result

    def plan(self, queues, budget=16, **kwargs):
        options = dict(token_budget=budget, max_num_seqs=1,
                       min_prefill_chunk_tokens=4, prefill_chunk_quantum=16)
        options.update(kwargs)
        return plan_iteration(queues, **options)

    def test_split_priority_preserves_timing_legacy_and_cold_start_values(self):
        wait = math.exp(.01 * 5)
        prefill_cases = [
            ({}, wait),
            ({"timing_required": True}, -math.inf),
            ({"local_prefill_ms": 200, "local_decode_per_token_ms": 30,
              "push_ms": 10, "prefill_gamma": 4, "P_n": 100}, wait - .30),
            ({"timing_priority": True, "T_i_p": .2, "T_i_d": .03,
              "T_i_push": .01, "m_i": 4}, wait - .30),
        ]
        for fields, expected in prefill_cases:
            req = {"task_type": "prefill", "proc_id": "p", "W_i": 5, **fields}
            with self.subTest(req=req):
                self.assertEqual(prefill_priority_score(req, now=9),
                                 compute_priority_score(req, {}, now=9))
                self.assertAlmostEqual(prefill_priority_score(req, now=9), expected)
        verify_cases = [
            ({"lag": .4, "transport_rtt": .1}, {}, wait - .5),
            ({"lag": .4, "transport_rtt": .1}, {"v": (0, 1)}, wait - 1),
            ({"draft_time_per_token": .1, "edge_rtt": .1}, {"v": (3, 4)}, wait - .5 / .8),
            ({"local_decode_per_token_ms": 30, "push_ms": 10, "pull_ms": 20},
             {"v": (3, 4)}, wait - .15 / .8),
            ({"T_i_d": .03, "T_i_push": .01, "T_i_pull": .02, "P_n": .5},
             {}, wait - .15 / .5),
        ]
        for fields, stats, expected in verify_cases:
            req = {"task_type": "verify", "proc_id": "v", "gamma": 4, "W_i": 5, **fields}
            with self.subTest(req=req):
                self.assertEqual(verify_priority_score(req, stats, now=9),
                                 compute_priority_score(req, stats, now=9))
                self.assertAlmostEqual(verify_priority_score(req, stats, now=9), expected)

    def test_explicit_visit_table_chooses_queue_before_score(self):
        cases = [
            (0, ("sp", "mv"), "sp"),
            (6, ("sv", "mp"), "sv"),
            (6, ("mv", "sv"), "mv"),
            (9, ("lv", "sv"), "lv"),
            (0, ("mv", "lp"), "mv"),
            (9, ("sv", "sp", "mv", "mp", "lp"), "sv"),
        ]
        for cursor, available, expected in cases:
            items = {name: self.item(name, "verify" if name[1] == "v" else "prefill",
                                    {"s": "short", "m": "mid", "l": "long"}[name[0]],
                                    request={"W_i": 100 if name == "sv" else 0})
                     for name in available}
            with self.subTest(cursor=cursor, available=available):
                plan = self.plan(self.queues(*items.values()), wrr_cursor=cursor)
                self.assertEqual(plan.selected_work_ids, [expected])
                self.assertEqual(plan.slots_consumed, 1)

    def test_queue_internal_order_uses_original_score(self):
        low = self.item("low", request={"lag": 4})
        high = self.item("high", request={"lag": 1})
        self.assertEqual(self.plan(self.queues(low, high)).selected_work_ids, ["high"])

    def test_overdue_prefill_protection_is_only_current_category(self):
        sv = self.item("sv")
        mp = self.item("mp", "prefill", "mid", missed_cycles=2)
        self.assertEqual(self.plan(self.queues(sv, mp)).selected_work_ids, ["sv"])
        self.assertEqual(self.plan(self.queues(sv, mp), wrr_cursor=6).selected_work_ids, ["mp"])

    def test_no_runnable_task_does_not_spend_visits_or_cycles(self):
        pending = self.item("pending", "prefill", request={"timing_required": True})
        for queues in (self.queues(), self.queues(pending)):
            plan = self.plan(queues, wrr_cursor=9, current_cycle=12)
            self.assertEqual((plan.next_wrr_cursor, plan.next_cycle, plan.slots_consumed), (9, 12, 0))
            self.assertEqual(plan.seal_reason, "no_runnable_task")

    def test_tail_budget_preserves_unspent_visit_and_no_small_backfill(self):
        mid = self.item("mv", category="mid", tokens=8)
        long = self.item("lp", "prefill", "long", tokens=100, missed_cycles=2)
        short = self.item("sv", tokens=1)
        queues = self.queues(mid, long, short)
        plan = self.plan(queues, wrr_cursor=8, max_num_seqs=4,
                         min_prefill_chunk_tokens=16, prefill_chunk_quantum=16)
        self.assertEqual(plan.selected_work_ids, ["mv"])
        self.assertEqual((plan.used_tokens, plan.slots_consumed, plan.next_wrr_cursor), (8, 1, 9))
        self.assertEqual(plan.seal_reason, "insufficient_remaining_budget")
        commit_admission_plan(plan)
        next_plan = self.plan(queues, wrr_cursor=plan.next_wrr_cursor,
                              min_prefill_chunk_tokens=16)
        self.assertEqual(next_plan.selected_work_ids, ["lp"])

    def test_bridge_candidate_that_cannot_fit_keeps_visit(self):
        a = self.item("a", tokens=3)
        b = self.item("b", tokens=4, bridge_pending=1)
        queues = self.queues(a, b)
        plan = self.plan(queues, budget=4, max_num_seqs=3)
        self.assertEqual(plan.selected_work_ids, ["a"])
        self.assertEqual(plan.next_wrr_cursor, 1)
        self.assertEqual(plan.seal_reason, "insufficient_remaining_budget")

    def test_full_budget_unschedulable_request_raises(self):
        with self.assertRaisesRegex(ValueError, "cannot fit a legal slice"):
            self.plan(self.queues(self.item("v", bridge_pending=4)), budget=4)

    def test_prefill_tail_below_minimum_must_fit_as_a_whole(self):
        verify = self.item("v", tokens=3)
        tail = self.item("p", "prefill", tokens=2)
        plan = self.plan(self.queues(verify, tail), budget=4, max_num_seqs=4)
        self.assertEqual(plan.selected_work_ids, ["v"])
        self.assertEqual(plan.used_tokens, 3)
        self.assertEqual(plan.slots_consumed, 1)

    def test_cross_batch_successes_keep_631_and_idle_keeps_cursor(self):
        items = [self.item(str(i), category=category, tokens=1)
                 for i, category in enumerate(["short"] * 6 + ["mid"] * 3 + ["long"])]
        queues, cursor, cycle, trace = self.queues(*items), 0, 0, []
        for _ in range(10):
            plan = self.plan(queues, wrr_cursor=cursor, current_cycle=cycle)
            trace.append(plan.ordered_slices[0].category)
            commit_admission_plan(plan)
            cursor, cycle = plan.next_wrr_cursor, plan.next_cycle
        self.assertEqual(trace, ["short"] * 6 + ["mid"] * 3 + ["long"])
        self.assertEqual((cursor, cycle), (0, 1))
        idle = self.plan(queues, wrr_cursor=cursor, current_cycle=cycle)
        self.assertEqual((idle.next_wrr_cursor, idle.next_cycle), (0, 1))

    def test_prefill_extension_is_one_successful_visit(self):
        plan = self.plan(self.queues(self.item("p", "prefill", tokens=100)), budget=64)
        self.assertEqual((plan.used_tokens, plan.slots_consumed, plan.next_wrr_cursor), (64, 1, 1))

    def test_shared_budget_order_and_device_reservation(self):
        p = self.item("p", "prefill", "mid", tokens=4, missed_cycles=2)
        v = self.item("v", tokens=8, bridge_pending=1)
        duplicate = self.item("duplicate")
        duplicate.proc_id = "v"
        plan = self.plan(self.queues(p, v, duplicate), budget=16,
                         wrr_cursor=6, max_num_seqs=4)
        self.assertEqual([sl.work_id for sl in plan.ordered_slices], ["p", "v"])
        self.assertEqual(plan.used_tokens, 13)
        self.assertEqual(plan.slots_consumed, 2)

    def test_kv_ready_pending_and_miss_do_not_change_formal_plan(self):
        a, b = self.item("a"), self.item("b", category="mid")
        plans = []
        for readiness in ("ready", "pending", "miss"):
            a.request["_kv_readiness"] = readiness
            b.request["_kv_readiness"] = readiness
            def cache_dependent_rank(*_args):
                raise AssertionError("formal WRR must not query a cache-dependent rank")
            plan = self.plan(self.queues(a, b), wrr_cursor=6,
                             candidate_value=cache_dependent_rank)
            plans.append(([asdict(sl) for sl in plan.ordered_slices],
                          plan.used_tokens, plan.next_wrr_cursor, plan.next_cycle))
        self.assertEqual(plans[0], plans[1])
        self.assertEqual(plans[1], plans[2])

    def test_late_arrival_not_aged_and_repeated_commit_is_idempotent(self):
        v = self.item("v", tokens=4)
        waiting = self.item("old", "prefill", "long", tokens=100)
        late = self.item("new", "prefill", tokens=100)
        plan = self.plan(self.queues(v, waiting), wrr_cursor=9)
        commit_admission_plan(plan, all_items=[v, waiting, late])
        self.assertEqual((waiting.missed_cycles, late.missed_cycles), (1, 0))
        committed = [asdict(item) for item in (v, waiting, late)]
        commit_admission_plan(plan, all_items=[v, waiting, late])
        self.assertEqual([asdict(item) for item in (v, waiting, late)], committed)

    def test_prediction_uses_next_cursor_case_a(self):
        a, b = self.item("a"), self.item("b")
        c = self.item("c", category="mid")
        queues = self.queues(a, b, c)
        current = self.plan(queues, budget=8, wrr_cursor=5)
        prediction = predict_next_plan(queues, current, token_budget=8,
                                      min_prefill_chunk_tokens=4, max_num_seqs=1)
        self.assertEqual(current.selected_work_ids, ["a"])
        self.assertEqual(prediction.selected_work_ids, ["c"])
        self.assertEqual(prediction.prediction_assumptions, {"a": "all_accept"})

    def test_prediction_residual_keeps_next_budget_case_b(self):
        a, b = self.item("a", tokens=40), self.item("b")
        queues = self.queues(a, b)
        current = self.plan(queues, budget=8)
        reserve_admission_plan(current)
        prediction = predict_next_plan(queues, current, token_budget=8,
                                      min_prefill_chunk_tokens=4, max_num_seqs=1)
        self.assertEqual(prediction.selected_work_ids, ["a"])
        self.assertEqual(prediction.verify_slices[0].offset, 8)
        safe_transfers = [pid for pid in prediction.verify_proc_ids if pid not in {"a"}]
        self.assertEqual(safe_transfers, [])

    def test_prediction_rebuckets_partial_verify_with_frozen_thresholds(self):
        classifier = TaskLengthClassifier()
        classifier.thresholds["verify"] = (4, 8)
        frozen = dict(classifier.thresholds)
        category_for = lambda item: ("short" if item.remaining_tokens <= frozen[item.task_type][0]
                                    else "mid" if item.remaining_tokens <= frozen[item.task_type][1] else "long")
        a = self.item("a", category="long", tokens=10)
        queues = self.queues(a)
        current = self.plan(queues, budget=4, wrr_cursor=5)
        prediction = predict_next_plan(queues, current, token_budget=4,
            max_num_seqs=1, min_prefill_chunk_tokens=4, category_for=category_for)
        self.assertEqual((prediction.verify_slices[0].category, prediction.verify_slices[0].offset), ("mid", 4))
        self.assertEqual(classifier.thresholds, frozen)
        self.assertEqual(list(classifier.history["verify"]), [])
        self.assertEqual((a.category, a.cursor), ("long", 0))

    def test_prediction_and_actual_commit_match_every_slice_field(self):
        a = self.item("a", category="long", tokens=30, bridge_pending=1)
        b = self.item("b", category="mid", tokens=9)
        p = self.item("p", "prefill", "long", tokens=140, missed_cycles=1)
        queues = self.queues(a, b, p)
        classifier = TaskLengthClassifier()
        thresholds = dict(classifier.thresholds)
        def category_for(item):
            r1, r2 = thresholds[item.task_type]
            return "short" if item.remaining_tokens <= r1 else "mid" if item.remaining_tokens <= r2 else "long"
        current = self.plan(queues, budget=8, wrr_cursor=9)
        original = copy.deepcopy(queues)
        options = dict(token_budget=8, max_num_seqs=2, min_prefill_chunk_tokens=4,
                       prefill_chunk_quantum=16, category_for=category_for, planning_time=50)
        for _ in range(3):
            predicted = predict_next_plan(queues, current, **options)
            self.assertEqual(queues, original)
        commit_admission_plan(current, category_for=category_for)
        actual = self.plan(self.queues(a, b, p), budget=8, max_num_seqs=2, wrr_cursor=current.next_wrr_cursor,
                           current_cycle=current.next_cycle, now=50)
        self.assertEqual([asdict(s) for s in predicted.ordered_slices],
                         [asdict(s) for s in actual.ordered_slices])
        self.assertEqual((predicted.used_tokens, predicted.next_wrr_cursor, predicted.next_cycle),
                         (actual.used_tokens, actual.next_wrr_cursor, actual.next_cycle))

    def test_known_rejection_and_cancel_have_no_projected_residual(self):
        a, b = self.item("a", tokens=20), self.item("b")
        queues = self.queues(a, b)
        current = self.plan(queues, budget=8)
        predicted = predict_next_plan(queues, current, known_phase_results={"a": 2},
            token_budget=8, max_num_seqs=1, min_prefill_chunk_tokens=4)
        self.assertEqual(predicted.selected_work_ids, ["b"])
        a.cancel_pending = True
        predicted = predict_next_plan(queues, current,
            token_budget=8, max_num_seqs=1, min_prefill_chunk_tokens=4)
        self.assertEqual(predicted.selected_work_ids, ["b"])

    def test_applied_phase_does_not_advance_again_or_age_waiters_twice(self):
        a = self.item("a", tokens=20)
        p = self.item("p", "prefill", "long", tokens=100)
        queues = self.queues(a, p)
        current = self.plan(queues, budget=8, wrr_cursor=9)
        a.cursor = a.accepted_so_far = 8
        a.bridge_pending = 0
        a.state = "running"
        before = (asdict(a), asdict(p))
        for _ in range(3):
            projected = project_after_plan(queues, current, progress={"a": 8}, applied_work_ids={"a"})
            rows = {item.work_id: item for cats in projected.values() for items in cats.values() for item in items}
            self.assertEqual(rows["a"].cursor, 8)
            self.assertEqual(rows["p"].missed_cycles, 1)
            self.assertEqual((asdict(a), asdict(p)), before)

    def test_new_arrival_in_prediction_does_not_inherit_previous_cycle(self):
        v = self.item("v", tokens=8)
        queues = self.queues(v)
        current = self.plan(queues, budget=8, wrr_cursor=9)
        late = self.item("late", "prefill", tokens=20)
        queues["prefill"]["short"].append(late)
        projected = project_after_plan(queues, current)
        self.assertEqual(projected["prefill"]["short"][0].missed_cycles, 0)


if __name__ == "__main__":
    unittest.main()
