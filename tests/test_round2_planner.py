import unittest
from dataclasses import asdict
from src.fastsd_scheduler import (WorkItem, cost_features, plan_iteration,
    project_after_plan, commit_admission_plan, length_category, transition_work_item)


class PlannerConsistencyTests(unittest.TestCase):
    def item(self, pid, kind, total, **kwargs):
        return WorkItem(pid, pid, kind, 'short', total_tokens=total,
                        request={'task_type': kind, 'proc_id': pid}, **kwargs)

    def test_project_and_commit_match_aging_rebucketing_and_partial_verify(self):
        a = self.item('a', 'verify', 16, base_prefix_len=100, bridge_pending=1)
        p = self.item('p', 'prefill', 140, missed_cycles=1)
        q = self.item('q', 'prefill', 500)
        queues = {'verify': {'short': [a]}, 'prefill': {'short': [p, q]}}
        plan = plan_iteration(queues, 8, max_num_seqs=1, min_prefill_chunk_tokens=1,
                              current_cycle=3, wrr_cursor=9)
        classify = lambda item: length_category(item.remaining_tokens, 128, 512)
        projection = project_after_plan(queues, plan, category_for=classify)
        projected = {i.work_id: i for cats in projection.values() for rows in cats.values() for i in rows}
        self.assertEqual(a.cursor, 0)
        commit_admission_plan(plan, category_for=classify)
        for item in (a, p, q):
            self.assertEqual(asdict(item), asdict(projected[item.work_id]))
        self.assertEqual(a.accepted_so_far, 7)
        self.assertEqual(a.bridge_pending, 0)
        self.assertEqual(p.missed_cycles, 2)
        self.assertEqual(p.category, 'mid')
        self.assertTrue(p.is_overdue_prefill)

    def test_candidate_selected_and_execution_share_cost_features(self):
        item = self.item('v', 'verify', 8, base_prefix_len=100, bridge_pending=1)
        seen = []
        def rank(candidate, sl, selected):
            seen.append(cost_features(candidate, sl))
            return 1
        plan = plan_iteration({'verify': {'short': [item]}}, 4,
            min_prefill_chunk_tokens=1, selection_policy='progress', candidate_value=rank)
        sl = plan.verify_slices[0]
        self.assertEqual(seen, [(99, 4)])
        req = {'_cost_features': cost_features(item, sl)}
        self.assertEqual(cost_features(req), seen[0])
        commit_admission_plan(plan)
        next_plan = plan_iteration({'verify': {'short': [item]}}, 4, min_prefill_chunk_tokens=1)
        self.assertEqual(cost_features(item, next_plan.verify_slices[0]), (103, 4))

    def test_partial_rejection_retires_round_and_never_ages_it(self):
        item = self.item('v', 'verify', 16, base_prefix_len=10)
        plan = plan_iteration({'verify': {'short': [item]}}, 4, min_prefill_chunk_tokens=1)
        changed = transition_work_item(item, plan.verify_slices[0], start_cycle=0,
                                      next_cycle=1, progress=2)
        self.assertTrue(changed.finished)
        self.assertEqual(changed.accepted_so_far, 2)
        self.assertEqual(item.cursor, 0)


if __name__ == '__main__':
    unittest.main()
