"""Workstream D (default engine): the blackboard swarm auto-compacts its prompt to the
current model's context window, dropping supplementary recall / trimming the oldest
history tail while preserving the protected head (flag candidates, decoded state)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.agents.swarm_orchestrator import swarm_orchestrator
from backend.agents.swarm_state import SwarmBlackboard
from backend.agent_runtime.context_budget import over_budget


def _board_with_bulk():
    b = SwarmBlackboard("chal-d", "run-d", "http://target.local", category="web")
    # Protected content — must survive compaction (it sits at the top of history).
    b.flag_candidates.append({"flag": "picoCTF{protected_keep}", "worker": "agent_1", "source": "regex"})
    b.deobfuscated_secrets.append("SECRET_TOKEN_KEEP")
    # Bulky, low-value supplementary recall — the first thing compaction should drop.
    b.memory_context = "RETRIEVED_PLAYBOOK " * 9000
    # Some step history (the compressible tail).
    for i in range(120):
        b.record_agent_step("agent_1", note=f"step {i} " + "Z" * 80)
    return b


class TestDefaultEngineCompaction(unittest.TestCase):

    def test_no_model_means_no_compaction(self):
        b = _board_with_bulk()
        sys_i, prompt, compacted = swarm_orchestrator._build_budgeted_prompt(b, ".", "agent_1", "")
        self.assertFalse(compacted)
        self.assertIn("RETRIEVED_PLAYBOOK", prompt)  # untouched when model is unknown

    def test_compaction_drops_recall_keeps_protected_and_fits(self):
        b = _board_with_bulk()
        sys_i, prompt, compacted = swarm_orchestrator._build_budgeted_prompt(
            b, ".", "agent_1", "totally-unknown-model")  # ~32k window
        self.assertTrue(compacted)
        self.assertNotIn("RETRIEVED_PLAYBOOK", prompt)          # supplementary recall dropped
        self.assertIn("picoCTF{protected_keep}", prompt)        # protected flag candidate kept
        self.assertIn("SECRET_TOKEN_KEEP", prompt)              # protected decoded secret kept
        self.assertFalse(over_budget(sys_i, prompt, "totally-unknown-model"))

    def test_large_window_model_keeps_everything(self):
        b = _board_with_bulk()
        sys_i, prompt, compacted = swarm_orchestrator._build_budgeted_prompt(
            b, ".", "agent_1", "gemini-3.6-flash")  # 1M window
        self.assertFalse(compacted)
        self.assertIn("RETRIEVED_PLAYBOOK", prompt)


if __name__ == "__main__":
    unittest.main()
