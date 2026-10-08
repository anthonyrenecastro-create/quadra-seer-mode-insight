"""Tests for reasoning compute budget allocator."""

import unittest

from atlantean_core.reasoning_compute_budget import (
    ReasoningComputeAllocator,
    ReasoningComputeBudgetPolicy,
    build_reasoning_compute_budget_policy,
)


class TestReasoningComputeBudget(unittest.TestCase):

    def _make_allocator(self, **overrides) -> ReasoningComputeAllocator:
        policy = ReasoningComputeBudgetPolicy(
            enabled=True,
            reasoning_max_ms_per_query=10.0,
            aux_max_ms_per_query=10.0,
            max_total_ms_per_window=200.0,
            window_seconds=60,
        )
        for k, v in overrides.items():
            setattr(policy, k, v)
        return ReasoningComputeAllocator(policy=policy)

    def test_allocation_allowed_under_cap(self):
        allocator = self._make_allocator()
        decision = allocator.allocate("reasoning", "simple")
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "allowed")
        self.assertEqual(decision.pass_type, "reasoning")
        self.assertEqual(decision.query_class, "simple")

    def test_allocation_denied_after_query_cap_exceeded(self):
        allocator = self._make_allocator(reasoning_max_ms_per_query=5.0)
        # Consume 5ms for simple class (cap=5.0 * 0.5 multiplier = 2.5ms)
        allocator.record_elapsed("reasoning", 3.0)
        decision = allocator.allocate("reasoning", "simple")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "query-compute-cap-exceeded")

    def test_allocation_denied_when_budget_disabled(self):
        allocator = self._make_allocator(enabled=False)
        decision = allocator.allocate("reasoning", "deep")
        # When disabled, all passes are allowed (budget-disabled reason).
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "budget-disabled")

    def test_deep_query_class_gets_higher_cap(self):
        allocator = self._make_allocator(reasoning_max_ms_per_query=10.0)
        decision_simple = allocator.allocate("reasoning", "simple")
        decision_deep = allocator.allocate("reasoning", "deep")
        # deep cap = 10.0 * 2.0 = 20.0; simple cap = 10.0 * 0.5 = 5.0
        self.assertGreater(decision_deep.effective_cap_ms, decision_simple.effective_cap_ms)

    def test_aux_pass_type_tracked_independently(self):
        allocator = self._make_allocator(reasoning_max_ms_per_query=2.0, aux_max_ms_per_query=2.0)
        allocator.record_elapsed("reasoning", 5.0)
        reasoning_decision = allocator.allocate("reasoning", "simple")
        aux_decision = allocator.allocate("aux", "simple")
        self.assertFalse(reasoning_decision.allowed)
        self.assertTrue(aux_decision.allowed)

    def test_snapshot_structure(self):
        allocator = self._make_allocator()
        allocator.record_elapsed("reasoning", 3.5)
        snap = allocator.snapshot()
        self.assertIn("enabled", snap)
        self.assertIn("passes", snap)
        self.assertIn("reasoning", snap["passes"])
        self.assertIn("aux", snap["passes"])
        self.assertAlmostEqual(snap["passes"]["reasoning"]["query_consumed_ms"], 3.5, places=2)

    def test_build_reasoning_compute_budget_policy_factory(self):
        policy = build_reasoning_compute_budget_policy(
            enabled=True,
            reasoning_max_ms_per_query=15.0,
            aux_max_ms_per_query=15.0,
            max_total_ms_per_window=300.0,
            window_seconds=120,
        )
        self.assertTrue(policy.enabled)
        self.assertEqual(policy.reasoning_max_ms_per_query, 15.0)
        self.assertEqual(policy.window_seconds, 120)


if __name__ == "__main__":
    unittest.main()
