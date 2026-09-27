"""
END-TO-END integration test for the FORGE experience memory layer.

Unlike ``tests/test_experience_memory.py`` (which unit-tests the components with
duck-typed boards and freshly-constructed, isolated instances), this test drives
the REAL execution path through the production singletons and the real
``SwarmOrchestrator`` / ``SwarmBlackboard`` objects, proving the loop the audit
asks about:

    SOLVE  ->  SwarmOrchestrator._learn_from_run   (the flag-captured branch)
           ->  ExperienceExtractor.extract_from_board
           ->  experience_memory.store            (production singleton)
    (later run)
           ->  memory_retriever.retrieve_and_format (production singleton)
           ->  SwarmOrchestrator._build_agent_context
           ->  agent_prompt.build_user_prompt      -> memory visible to the agent
           ->  experience_memory.record_feedback   (reuse reinforces the memory)
           ->  maybe_promote                       -> Playbook Vault

The ONLY parts not exercised here are the live LLM inference and the network
target I/O of a full ``run_swarm()`` (those need provider keys + a reachable CTF
target). Every deterministic FORGE function in the learn / retrieve / inject /
feedback / promote path is the real production object, wired exactly as
``run_swarm`` wires it (see swarm_orchestrator.py lines ~915-952 for retrieval and
~1002-1023 for the flag-captured learn branch, mirrored faithfully below).

Isolation: only the Playbook Vault *sink* is swapped to a temp directory so a test
solve never writes YAML into the real, gitignored vault. The experience store is
the real singleton, pointed at the isolated ``test_forge.db`` per the project's
NO-DEMO-DATA / isolated-test-DB directive.
"""
import os
import glob
import shutil
import asyncio
import tempfile
import unittest

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

# Import AFTER DATABASE_URL is set so the module-level singletons bind to the
# isolated test database.
from backend.database.session import init_db, SessionLocal
from backend.database.models import ExperienceModel, ExperienceAttemptModel, MemoryUsageModel
from backend.agents.swarm_orchestrator import swarm_orchestrator
from backend.agents.swarm_state import SwarmBlackboard
from backend.knowledge.experience_memory import experience_memory
from backend.knowledge.memory_retriever import memory_retriever
from backend.knowledge.experience_extractor import experience_extractor
from backend.agents.agent_prompt import build_user_prompt
from backend.knowledge.playbook_vault import PlaybookVault

# Concrete, challenge-specific values that MUST NOT survive into reusable memory.
FLAG = "picoCTF{r34l_ss7i_run_2026}"
TARGET = "http://ssti.forge-range.ctf:8080"
SECRET = "R3dTeamAdminSecret!"
TARGET_IP = "10.10.14.55"

def _solved_ssti_board(challenge_id="itg-chal-1", run_id="itg-run-1", name="Reflected Greeting"):
    """Build a REAL SwarmBlackboard populated the way a real solved run would be:
    evidence recorded through the real record_agent_step(), a failed->success
    command trace, and a captured flag."""
    board = SwarmBlackboard(challenge_id=challenge_id, run_id=run_id,
                            target_scope=f"{TARGET} + {TARGET_IP}")
    board.challenge_name = name
    board.category = "WEB"
    board.difficulty = "MEDIUM"
    board.description = "A page greets you by rendering your submitted display name."
    board.discovered_endpoints.update(["/", "/greet"])
    board.extracted_headers.update({"Server": "Werkzeug/2.1 Python/3.11", "X-Powered-By": "Flask"})
    board.deobfuscated_secrets.append({"admin_pw": SECRET})
    board.candidate_usernames.add("admin")

    # A realistic trace: two dead ends, then the winning SSTI chain — recorded
    # through the REAL blackboard method that a live run uses.
    board.record_agent_step("agent_1", command=f"curl -s {TARGET}/",
                            output="HTTP/1.1 200 OK\nServer: Werkzeug/2.1 Python/3.11", note="recon")
    board.record_agent_step("agent_1",
                            command=f"curl -s '{TARGET}/greet?name=<script>alert(1)</script>'",
                            output="403 Forbidden - request blocked", note="xss probe")
    board.record_agent_step("agent_2", command=f"curl -s '{TARGET}/greet?name=.php5'",
                            output="400 Bad Request rejected", note="bogus upload-ext probe")
    board.record_agent_step("agent_2", command=f"curl -s '{TARGET}/greet?name={{{{7*7}}}}'",
                            output="Hello, 49", note="jinja render_template_string reflected")
    board.record_agent_step("agent_2", command=f"curl -s '{TARGET}/greet?name={{{{config.items()}}}}'",
                            output=f"...SECRET_KEY={SECRET}... {FLAG} ...", note="flag captured")
    board.flag_captured = FLAG
    return board

def _similar_ssti_board(challenge_id="itg-chal-2", run_id="itg-run-2", name="Name Echo"):
    """A DIFFERENT later challenge with the same SSTI-shaped fingerprint, used to
    prove retrieval pulls the earlier run's experience into this run's context."""
    board = SwarmBlackboard(challenge_id=challenge_id, run_id=run_id,
                            target_scope="http://echo.other-range.ctf:5000")
    board.challenge_name = name
    board.category = "WEB"
    board.difficulty = "MEDIUM"
    board.description = "The site echoes your input back through a server-side Jinja template."
    board.discovered_endpoints.update(["/", "/echo", "/profile"])
    board.extracted_headers.update({"Server": "Werkzeug/2.0 Python/3.10", "X-Powered-By": "Flask"})
    return board

def _run_retrieval_phase(board):
    """Faithful in-test replica of the ONE shared retrieval phase run_swarm runs
    before agents spawn (swarm_orchestrator.py ~lines 920-938). Uses the REAL
    production retriever + extractor helper + the REAL blackboard fields."""
    evidence_parts = [board.description or ""]
    evidence_parts.extend(sorted(board.discovered_endpoints)[:15])
    evidence_parts.extend(f"{k}: {v}" for k, v in list(board.extracted_headers.items())[:12])
    evidence_text = "\n".join(p for p in evidence_parts if p)
    technologies = experience_extractor._detect_technologies(evidence_text)
    mem_context, memories = memory_retriever.retrieve_and_format(
        evidence=evidence_text, category=board.category, technologies=technologies,
        query=f"{board.category} {board.challenge_name}", top_k=6,
    )
    board.memory_context = mem_context
    board.retrieved_memory_ids = [m.id for m in memories if m.kind == "experience" and m.id]
    if memories:
        experience_memory.record_retrieval(board.retrieved_memory_ids, board.run_id, board.challenge_id)
    return mem_context, memories

class MemoryIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self._clean_tables()
        # Resync the production singleton's in-memory FTS index to the now-empty DB
        # so retrieval assertions see only what THIS test stores.
        experience_memory.reload_index()
        # Swap the Playbook Vault sink to a temp dir so a test solve never pollutes
        # the real, gitignored vault (rules §6/§7/§8). Restored in tearDown.
        self.tmp_dir = tempfile.mkdtemp(prefix="forge_mem_itg_")
        self._tmp_vault = PlaybookVault(base_dir=os.path.join(self.tmp_dir, "pb"))
        self._orig_mem_vault = experience_memory.vault
        self._orig_ret_vault = memory_retriever.vault
        experience_memory.vault = self._tmp_vault
        memory_retriever.vault = self._tmp_vault  # isolate playbook retrieval too

    def tearDown(self):
        experience_memory.vault = self._orig_mem_vault
        memory_retriever.vault = self._orig_ret_vault
        try:
            self._tmp_vault.db_conn.close()
        except Exception:
            pass
        self._clean_tables()
        experience_memory.reload_index()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        # Remove any challenge logs this test created (clean-process-lifecycle rule).
        for f in glob.glob(os.path.join("backend", "logs", "challenge_itg-*.log")):
            try:
                os.remove(f)
            except OSError:
                pass

    @staticmethod
    def _clean_tables():
        db = SessionLocal()
        try:
            db.query(MemoryUsageModel).delete()
            db.query(ExperienceAttemptModel).delete()
            db.query(ExperienceModel).delete()
            db.commit()
        finally:
            db.close()

    @staticmethod
    def _learn(board):
        """Drive the REAL flag-captured learning entry point."""
        asyncio.run(swarm_orchestrator._learn_from_run(board, outcome="success"))

    # ------------------------------------------------------------------ #

    def test_01_solved_run_automatically_learns_without_operator(self):
        """A flag-captured run distils and PERSISTS an experience automatically —
        no operator 'save experience' step. (Objective; §2)"""
        db = SessionLocal()
        try:
            self.assertEqual(db.query(ExperienceModel).count(), 0, "DB should start empty")
        finally:
            db.close()

        self._learn(_solved_ssti_board())

        db = SessionLocal()
        try:
            rows = db.query(ExperienceModel).all()
            self.assertEqual(len(rows), 1, "exactly one experience should be learned from the solve")
            exp = rows[0]
            self.assertEqual(exp.outcome, "success")
            self.assertEqual(exp.source, "forge_run")
            # §9 provenance: the memory links back to the originating run/challenge.
            self.assertEqual(exp.source_run_id, "itg-run-1")
            self.assertEqual(exp.source_challenge_id, "itg-chal-1")
            self.assertIn("Template Injection", exp.technique)
        finally:
            db.close()

    def test_02_learned_experience_is_generalized(self):
        """§4: no flag / target / secret / IP survives into stored memory."""
        import json
        self._learn(_solved_ssti_board())
        exp = experience_memory.list_experiences(limit=1)[0]
        blob = json.dumps(exp)
        self.assertNotIn(FLAG, blob, "flag leaked into memory")
        self.assertNotIn(TARGET, blob, "target URL leaked into memory")
        self.assertNotIn(SECRET, blob, "secret leaked into memory")
        self.assertNotIn(TARGET_IP, blob, "target IP leaked into memory")
        self.assertIn("{TARGET_URL}", blob, "URLs should be parameterised to a sentinel")

    def test_03_failed_attempts_are_preserved_alongside_success(self):
        """§5: FORGE remembers the dead ends, not only the winning move."""
        self._learn(_solved_ssti_board())
        exp = experience_memory.get(experience_memory.list_experiences(limit=1)[0]["id"],
                                    with_children=True)
        approaches = " ".join(f.get("approach", "") for f in exp["failed_techniques"])
        self.assertIn(".php5", approaches, "the rejected upload-ext probe should be remembered")
        self.assertTrue(exp["successful_attack_chain"], "winning chain must be recorded")
        self.assertTrue(any("7*7" in c for c in exp["successful_attack_chain"]),
                        "the SSTI probe should be in the winning chain")
        failure_attempts = [a for a in exp["attempts"] if a["outcome"] == "failure"]
        self.assertGreaterEqual(len(failure_attempts), 2, "both dead ends should be recorded as failures")
        self.assertTrue(all(a["reason"] for a in failure_attempts), "each failure needs a reason")

    def test_04_later_run_retrieves_and_injects_memory_into_agent_prompt(self):
        """THE CORE PROOF: a solved run teaches a *different* later run automatically.
        Run #1's experience is retrieved for run #2 and injected into the real agent
        prompt before any action is chosen. (§5, §6, §8 of the objective)"""
        # Run #1 solves and learns.
        self._learn(_solved_ssti_board())

        # Run #2 starts: the real retrieval phase pulls run #1's experience.
        board2 = _similar_ssti_board()
        mem_context, memories = _run_retrieval_phase(board2)

        self.assertTrue(memories, "retrieval returned nothing for a clearly-relevant later run")
        exp_hits = [m for m in memories if m.kind == "experience"]
        self.assertTrue(exp_hits, "the earlier FORGE experience was not retrieved")
        self.assertTrue(any("Template Injection" in m.technique for m in exp_hits))
        self.assertTrue(board2.retrieved_memory_ids, "retrieved experience ids should be tracked for feedback")

        # The real orchestrator builds the agent context; the real prompt builder
        # renders it. Memory must be present and appear BEFORE the action section.
        ctx = swarm_orchestrator._build_agent_context(board2, self.tmp_dir, "agent_1")
        prompt = build_user_prompt(ctx)
        self.assertIn("RELEVANT FORGE MEMORY", prompt, "memory section missing from the agent prompt")
        self.assertIn("Template Injection", prompt)
        self.assertLess(prompt.index("RELEVANT FORGE MEMORY"), prompt.index("YOUR NEXT ACTION"),
                        "memory must precede the agent's action decision")
        # Injecting memory must NOT leak the source challenge's secrets.
        self.assertNotIn(FLAG, prompt)
        self.assertNotIn(TARGET, prompt)
        self.assertNotIn(SECRET, prompt)

        # §12 feedback bookkeeping: retrieval was recorded.
        src = experience_memory.get(board2.retrieved_memory_ids[0])
        self.assertGreaterEqual(src["times_retrieved"], 1)

    def test_05_reuse_in_a_solved_run_reinforces_and_promotes(self):
        """§8/§10/§12: reusing a memory in another successful run reinforces its
        statistics and, once proven (>=2 successes), promotes it to a Playbook."""
        # Run #1 learns experience E.
        self._learn(_solved_ssti_board())
        e_id = experience_memory.list_experiences(limit=1)[0]["id"]
        before = experience_memory.get(e_id)
        self.assertIsNone(before["promoted_playbook_id"], "a single solve must not auto-promote")
        self.assertEqual(before["times_successful"], 1)

        # Run #2 retrieves E, then also solves -> feedback reinforces E.
        board2 = _similar_ssti_board()
        _run_retrieval_phase(board2)
        self.assertIn(e_id, board2.retrieved_memory_ids)
        board2.flag_captured = "picoCTF{a_second_distinct_solve}"
        self._learn(board2)  # real learn branch: stores E2 + positive feedback on E

        after = experience_memory.get(e_id)
        self.assertGreaterEqual(after["times_used"], 1, "reuse should increment times_used")
        self.assertGreaterEqual(after["times_successful"], 2, "second success should be counted")
        self.assertGreater(after["confidence"], before["confidence"], "confidence should rise on success")
        # Proven across two solves -> promoted into the (temp) Playbook Vault.
        self.assertIsNotNone(after["promoted_playbook_id"], "repeated success should promote to a playbook")
        promoted = self._tmp_vault.load_playbook(after["promoted_playbook_id"])
        self.assertIsNotNone(promoted, "promoted playbook should exist in the vault")
        import json
        self.assertNotIn(FLAG, json.dumps(promoted.model_dump()), "promoted playbook must stay generalized")

if __name__ == "__main__":
    unittest.main()
