"""Workstream D: per-model context budgeting + automatic extractive compaction."""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.agent_runtime.state import MissionState
from backend.agent_runtime.context import ContextBuilder
from backend.agent_runtime.context_budget import (
    estimate_tokens, model_window, budget_tokens, over_budget, FALLBACK_WINDOW,
)


def _events(n):
    return [SimpleNamespace(sequence=i, command=f"curl http://t/{i} " + "A" * 60,
                            result="FAIL", decision_summary="attempt")
            for i in range(n)]


def _state():
    s = MissionState()
    s.challenge_name = "T"
    s.category = "web"
    s.description = "desc"
    s.current_objective = "find flag"
    return s


class TestContextBudget(unittest.TestCase):

    def test_token_estimate_and_windows(self):
        self.assertEqual(estimate_tokens("x" * 400), 100)
        self.assertEqual(model_window("gemini-3.6-flash"), 1000000)
        self.assertEqual(model_window("z-ai/glm-5.3-flash"), 131072)
        self.assertGreaterEqual(model_window("some-unknown-codestral-x"), 131072)  # family match
        self.assertEqual(model_window(""), FALLBACK_WINDOW)

    def test_over_budget(self):
        big = "x" * (FALLBACK_WINDOW * 4 + 4000)   # ~> FALLBACK_WINDOW tokens
        self.assertTrue(over_budget("", big, "totally-unknown-model"))
        self.assertFalse(over_budget("", "short prompt", "gemini-3.6-flash"))

    def test_no_model_name_disables_budgeting(self):
        builder = ContextBuilder(memory_retriever=None, trajectory_search=None)
        _sys, prompt = builder.build(state=_state(), recent_events=[],
                                     memory_context="MEMORY_MARKER " * 200)
        self.assertIn("MEMORY_MARKER", prompt)  # untouched when no model is given

    def test_compaction_drops_memory_but_keeps_protected(self):
        state = _state()
        state.flag_candidates = ["picoCTF{keep_me}"]
        state.failed_techniques = ["sqli on /login :: 403 blocked"]
        builder = ContextBuilder(memory_retriever=None, trajectory_search=None)
        big_memory = "RETRIEVED_PLAYBOOK " * 8000   # ~38k tokens, well over a 32k window

        sys_i, prompt = builder.build(
            state=state, recent_events=_events(40),
            memory_context=big_memory, model_name="totally-unknown-model",
        )
        # Supplementary recall was compacted away...
        self.assertNotIn("RETRIEVED_PLAYBOOK", prompt)
        # ...but protected content survives verbatim.
        self.assertIn("picoCTF{keep_me}", prompt)
        self.assertIn("sqli on /login", prompt)
        # And the result actually fits the model budget now.
        self.assertFalse(over_budget(sys_i, prompt, "totally-unknown-model"))

    def test_large_window_model_keeps_memory(self):
        builder = ContextBuilder(memory_retriever=None, trajectory_search=None)
        mem = "RETRIEVED_PLAYBOOK " * 400
        _sys, prompt = builder.build(state=_state(), recent_events=_events(10),
                                     memory_context=mem, model_name="gemini-3.6-flash")
        self.assertIn("RETRIEVED_PLAYBOOK", prompt)  # 1M window → no compaction

    def test_model_switch_recomputes_budget(self):
        """Same context: compacts on a tiny-window model, not on a huge-window one (D3)."""
        builder = ContextBuilder(memory_retriever=None, trajectory_search=None)
        mem = "RECALL " * 20000   # ~35k tokens — over a 32k window, under gemini's 1M
        _s1, tiny = builder.build(state=_state(), recent_events=_events(20),
                                  memory_context=mem, model_name="totally-unknown-model")
        _s2, huge = builder.build(state=_state(), recent_events=_events(20),
                                  memory_context=mem, model_name="gemini-3.6-flash")
        self.assertNotIn("RECALL", tiny)
        self.assertIn("RECALL", huge)


if __name__ == "__main__":
    unittest.main()
