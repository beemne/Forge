"""
Tests for the Experience/Memory layer:
extraction, deterministic generalization (§4), retrieval + ranking (§6, §7),
agent-context injection (§8), the feedback loop (§12), provenance (§13),
playbook promotion (§10), and the clean empty state (§1/§2).

All state lives in the isolated unit-test database per the project directive.
"""
import os
import sys
import shutil
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

from backend.database.session import init_db, SessionLocal
from backend.database.models import ExperienceModel, ExperienceAttemptModel, MemoryUsageModel

CAPTURED_FLAG = "picoCTF{ss7i_tem9late_pwn}"
TARGET_URL = "http://blog.challenge.ctf:8080"
LEAKED_SECRET = "sup3rSecretAdminPass"
TARGET_IP = "10.10.14.99"

def _solved_board(**overrides):
    """A lightweight stand-in for a completed SwarmBlackboard (duck-typed)."""
    import types as _t
    board = _t.SimpleNamespace(
        run_id="run-unit-1",
        challenge_id="chal-unit-1",
        challenge_name="Template Terror",
        category="WEB",
        difficulty="MEDIUM",
        description="A blog renders your display name through a server-side template.",
        target_scope=f"{TARGET_URL} + {TARGET_IP}",
        discovered_endpoints={"/", "/profile", "/render"},
        extracted_headers={"Server": "Werkzeug/2.0 Python/3.9", "X-Powered-By": "Flask"},
        observed_cookies={"session": "eyJhbGciOiJub25lIn0.eyJ1c2VyIjoiZ3Vlc3QifQ."},
        deobfuscated_secrets=[{"admin_pw": LEAKED_SECRET}],
        candidate_tokens=set(),
        candidate_usernames={"admin"},
        artifact_classification=None,
        flag_captured=CAPTURED_FLAG,
        execution_history=[
            {"agent": "a1", "command": f"curl -s {TARGET_URL}/", "output": "Server: Werkzeug/2.0", "note": ""},
            {"agent": "a1", "command": f"curl -s '{TARGET_URL}/render?name=.php5'", "output": "400 Bad Request rejected", "note": ""},
            {"agent": "a1", "command": f"curl -s '{TARGET_URL}/render?name={{{{7*7}}}}'", "output": "Hello, 49", "note": "jinja render_template_string reflected"},
            {"agent": "a2", "command": f"curl -s '{TARGET_URL}/render?name={{{{config}}}}'", "output": f"...{CAPTURED_FLAG}...", "note": "flag in output"},
        ],
    )
    for k, v in overrides.items():
        setattr(board, k, v)
    return board

class MemoryTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self._clean_tables()
        self.tmp_dir = tempfile.mkdtemp(prefix="forge_mem_test_")
        from backend.knowledge.playbook_vault import PlaybookVault
        from backend.knowledge.experience_memory import ExperienceMemory
        from backend.knowledge.memory_retriever import MemoryRetriever
        from backend.knowledge.experience_extractor import ExperienceExtractor
        self.vault = PlaybookVault(base_dir=os.path.join(self.tmp_dir, "pb"))
        self.mem = ExperienceMemory(vault=self.vault)
        self.retriever = MemoryRetriever(memory=self.mem, vault=self.vault)
        self.extractor = ExperienceExtractor()

    def tearDown(self):
        try:
            self.mem.db_conn.close()
        except Exception:
            pass
        try:
            self.vault.db_conn.close()
        except Exception:
            pass
        self._clean_tables()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @staticmethod
    def _clean_tables():
        db = SessionLocal()
        try:
            # Children before parents — FK enforcement is ON.
            db.query(MemoryUsageModel).delete()
            db.query(ExperienceAttemptModel).delete()
            db.query(ExperienceModel).delete()
            db.commit()
        finally:
            db.close()

class TestExtractionAndGeneralization(MemoryTestBase):
    def test_generalization_strips_all_challenge_specific_secrets(self):
        """§4: a solved challenge must NEVER be stored with its concrete secrets."""
        import json
        rec = self.extractor.extract_from_board(_solved_board(), flag=CAPTURED_FLAG, outcome="success")
        blob = json.dumps(rec.model_dump())
        self.assertNotIn(CAPTURED_FLAG, blob, "flag leaked into experience")
        self.assertNotIn(TARGET_URL, blob, "target URL leaked into experience")
        self.assertNotIn(LEAKED_SECRET, blob, "secret leaked into experience")
        self.assertNotIn(TARGET_IP, blob, "target IP leaked into experience")
        # The reusable sentinel token must be present instead.
        self.assertIn("{TARGET_URL}", blob)

    def test_extracts_generalized_technique_and_attack_chain(self):
        rec = self.extractor.extract_from_board(_solved_board(), flag=CAPTURED_FLAG, outcome="success")
        self.assertIn("Template Injection", rec.technique)
        self.assertIn("ssti", rec.tags)
        self.assertTrue(rec.successful_attack_chain, "winning chain should not be empty")
        self.assertIn("flask", rec.technologies)

    def test_failed_approaches_are_recorded(self):
        """§5: FORGE must remember what FAILED, not only what worked."""
        rec = self.extractor.extract_from_board(_solved_board(), flag=CAPTURED_FLAG, outcome="success")
        self.assertTrue(rec.failed_techniques, "rejected attempts should be captured")
        joined = " ".join(f["approach"] for f in rec.failed_techniques)
        self.assertIn(".php5", joined)
        self.assertTrue(any(a.outcome == "failure" for a in rec.attempts))

    def test_blue_team_detection_indicators_derived(self):
        """§11: a successful attack should yield reusable defensive knowledge."""
        rec = self.extractor.extract_from_board(_solved_board(), flag=CAPTURED_FLAG, outcome="success")
        di = rec.detection_indicators
        for key in ("indicators", "logs", "investigation", "containment"):
            self.assertIn(key, di)
        self.assertTrue(di["indicators"])

    def test_failure_run_extraction(self):
        board = _solved_board(flag_captured="")
        rec = self.extractor.extract_from_board(board, flag="", outcome="failure")
        self.assertEqual(rec.outcome, "failure")
        self.assertLess(rec.confidence, 0.6)

class TestStoreRetrieveRank(MemoryTestBase):
    def _store(self, board=None, flag=CAPTURED_FLAG, outcome="success"):
        rec = self.extractor.extract_from_board(board or _solved_board(), flag=flag, outcome=outcome)
        return self.mem.store(rec)

    def test_store_and_search(self):
        exp_id = self._store()
        self.assertTrue(exp_id)
        hits = self.mem.search(query="web ssti template", category="web",
                               evidence="Werkzeug Flask jinja", top_k=5)
        self.assertTrue(any(h["id"] == exp_id for h in hits))

    def test_retrieval_returns_upload_memory_for_upload_evidence(self):
        """§21: upload evidence → upload memories returned."""
        upload_board = _solved_board(
            challenge_name="Upload Uploader", category="WEB",
            description="Profile picture upload.",
            discovered_endpoints={"/upload"},
            extracted_headers={"Server": "nginx"},
            execution_history=[
                {"agent": "a1", "command": f"curl -F 'file=@shell.phtml' {TARGET_URL}/upload -H 'Content-Type: multipart/form-data'",
                 "output": "Upload successful uploads/shell.phtml", "note": ""},
            ],
        )
        self._store(board=upload_board)
        self._store()  # an unrelated SSTI experience
        hits = self.mem.search(query="upload bypass", evidence="multipart/form-data upload filename=", top_k=5)
        self.assertTrue(hits)
        self.assertIn("Upload", hits[0]["technique"])

    def test_ranking_prefers_proven_success(self):
        """§7: repeatedly-successful / higher-confidence memory ranks higher."""
        proven = self._store()
        _weak = self._store()
        # Reinforce the "proven" one so it becomes repeated-success + higher confidence.
        self.mem.record_feedback(proven, success=True)
        hits = self.mem.search(query="web ssti template", category="web",
                               evidence="Werkzeug Flask jinja", top_k=5)
        ids = [h["id"] for h in hits]
        self.assertIn(proven, ids)
        self.assertLess(ids.index(proven), ids.index(_weak), "proven memory should outrank the weaker one")

    def test_no_secret_leak_in_prompt_context(self):
        self._store()
        ctx, mems = self.retriever.retrieve_and_format(
            evidence="Werkzeug Flask jinja render", category="web",
            technologies=["flask", "werkzeug"], query="web ssti", top_k=6)
        self.assertIn("RELEVANT FORGE MEMORY", ctx)
        self.assertNotIn(CAPTURED_FLAG, ctx)
        self.assertNotIn(TARGET_URL, ctx)
        self.assertTrue(mems)

    def test_empty_memory_is_clean(self):
        """§1/§2: no seeded/injected data — a fresh store is empty and returns nothing."""
        stats = self.mem.get_stats()
        self.assertEqual(stats["total_memories"], 0)
        self.assertEqual(self.mem.search(query="anything", evidence="anything"), [])

class TestFeedbackProvenancePromotion(MemoryTestBase):
    def _store(self, outcome="success"):
        rec = self.extractor.extract_from_board(_solved_board(), flag=CAPTURED_FLAG, outcome=outcome)
        return self.mem.store(rec)

    def test_feedback_updates_success_statistics(self):
        """§12: a memory that helped gets reinforced; one that didn't is penalised."""
        exp_id = self._store()
        before = self.mem.get(exp_id)
        self.mem.record_feedback(exp_id, success=True, note="worked")
        after = self.mem.get(exp_id)
        self.assertEqual(after["times_used"], 1)
        self.assertEqual(after["times_successful"], before["times_successful"] + 1)
        self.assertGreaterEqual(after["confidence"], before["confidence"])
        # A failure decrements confidence and bumps the failure counter.
        self.mem.record_feedback(exp_id, success=False, note="did not apply")
        after2 = self.mem.get(exp_id)
        self.assertEqual(after2["times_failed"], 1)
        self.assertLess(after2["confidence"], after["confidence"])

    def test_retrieval_records_usage_event(self):
        exp_id = self._store()
        self.mem.record_retrieval([exp_id], run_id="run-x", challenge_id="chal-x")
        detail = self.mem.get(exp_id, with_children=True)
        self.assertEqual(detail["times_retrieved"], 1)
        self.assertTrue(any(u["event"] == "retrieved" for u in detail["usage_log"]))

    def test_provenance_links_back_to_run(self):
        """§13: memory retains a traceable link to its originating run/challenge."""
        exp_id = self._store()
        detail = self.mem.get(exp_id)
        self.assertEqual(detail["source"], "forge_run")
        self.assertEqual(detail["source_run_id"], "run-unit-1")
        self.assertEqual(detail["source_challenge_id"], "chal-unit-1")

    def test_promotion_to_playbook(self):
        """§10: a repeatedly-successful experience is promoted into the Playbook Vault."""
        exp_id = self._store()
        # First feedback pushes times_successful to 2 → eligible for promotion.
        self.mem.record_feedback(exp_id, success=True)
        detail = self.mem.get(exp_id)
        self.assertIsNotNone(detail["promoted_playbook_id"])
        promoted = self.vault.load_playbook(detail["promoted_playbook_id"])
        self.assertIsNotNone(promoted, "promoted playbook should exist in the vault")
        # The promoted playbook must also be generalized (no leaked flag/target).
        import json
        self.assertNotIn(CAPTURED_FLAG, json.dumps(promoted.model_dump()))

class TestAgentContextInjection(MemoryTestBase):
    def test_memory_appears_in_agent_prompt(self):
        """§8: retrieved memory is injected into the agent prompt as reference."""
        from backend.agents.agent_prompt import make_context_from_env, build_agent_prompt
        rec = self.extractor.extract_from_board(_solved_board(), flag=CAPTURED_FLAG, outcome="success")
        self.mem.store(rec)
        memory_context, mems = self.retriever.retrieve_and_format(
            evidence="Werkzeug Flask jinja", category="web", technologies=["flask"], query="web ssti", top_k=6)
        self.assertTrue(mems)
        ctx = make_context_from_env(
            env_info={}, challenge_name="X", platform="", category="WEB", difficulty="EASY",
            description="d", target_url="http://t", working_directory=".", max_iterations=10,
            max_minutes=10, flag_pattern="FLAG{...}", memory_context=memory_context)
        system, user = build_agent_prompt(ctx)
        self.assertIn("RELEVANT FORGE MEMORY", user)
        self.assertIn("Template Injection", user)
        # Injecting memory must not leak the source challenge's flag/target.
        self.assertNotIn(CAPTURED_FLAG, user)

class TestBug5Classification(MemoryTestBase):
    def test_rev_challenge_classified_as_reverse_engineering(self):
        board = _solved_board(
            category="REV",
            description="Reverse engineer this binary using ghidra or gdb",
            execution_history=[
                {"agent": "a1", "command": "ghidra --headless ./chall", "output": "decompiled main", "note": ""},
                {"agent": "a1", "command": "gdb ./chall -ex 'disassemble main'", "output": "checksec: partial relro, gdb output", "note": ""},
            ]
        )
        rec = self.extractor.extract_from_board(board, flag=CAPTURED_FLAG, outcome="success")
        self.assertEqual(rec.technique, "Reverse Engineering / Static Analysis")
        self.assertIn("rev", rec.tags)

    def test_xxe_requires_cooccurrence(self):
        # Ordinary HTML containing <!DOCTYPE should NOT be classified as XXE
        board = _solved_board(
            category="WEB",
            description="Web app with html login",
            execution_history=[
                {"agent": "a1", "command": "curl -s http://target/", "output": "<!DOCTYPE html><html><body>Login</body></html>", "note": ""}
            ]
        )
        rec = self.extractor.extract_from_board(board, flag=CAPTURED_FLAG, outcome="success")
        self.assertNotEqual(rec.technique, "XML External Entity (XXE)")

        # Actual XXE payload containing <!ENTITY or system "file should classify as XXE
        board_xxe = _solved_board(
            category="WEB",
            description="XML parser endpoint",
            execution_history=[
                {"agent": "a1", "command": "curl -X POST -d '<!DOCTYPE foo [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]>' http://target/xml", "output": "root:x:0:0", "note": ""}
            ]
        )
        rec_xxe = self.extractor.extract_from_board(board_xxe, flag=CAPTURED_FLAG, outcome="success")
        self.assertEqual(rec_xxe.technique, "XML External Entity (XXE)")

    def test_hashgate_repro_credential_bruteforce_not_xss(self):
        """Regression test for Hashgate repro:
        A web challenge solved via credential brute-forcing must be classified as
        'Credential Brute-Force / Weak Auth', not 'Cross-Site Scripting (XSS)', even when
        incidental <script> substrings appear in recon grep or HTML output.
        """
        board = _solved_board(
            challenge_name="Hashgate",
            category="WEB",
            description="A secure portal requiring authentication credentials.",
            discovered_endpoints={"/", "/login", "/dashboard"},
            extracted_headers={"Server": "nginx/1.18.0"},
            execution_history=[
                # Incidental <script> fragment appearing only in recon grep/curl
                {"agent": "a1", "command": "curl -s http://target/ | grep -v '<script>'",
                 "output": "<html><head><script src='main.js'></script></head><body><form action='/login'>...</form></body></html>",
                 "note": "recon"},
                # Decisive credential brute-force commands
                {"agent": "a2", "command": "hydra -l admin -P /usr/share/wordlists/rockyou.txt http://target/login http-post-form",
                 "output": "1 valid password found: admin:hunter2", "note": "hydra brute-force"},
                {"agent": "a2", "command": "python3 -c 'import requests; print(\"brute-force credential check\")'",
                 "output": "Valid credentials found: admin:hunter2", "note": "credential verification"},
                {"agent": "a2", "command": "curl -s -X POST http://target/login -d 'user=admin&pass=hunter2'",
                 "output": f"200 OK Welcome admin! Dashboard: {CAPTURED_FLAG}", "note": "flag retrieved with credentials"},
            ]
        )
        rec = self.extractor.extract_from_board(board, flag=CAPTURED_FLAG, outcome="success")
        self.assertEqual(rec.technique, "Credential Brute-Force / Weak Auth")
        self.assertIn("auth", rec.tags)
        self.assertIn("bruteforce", rec.tags)
        self.assertNotIn("xss", rec.tags)

    def test_classify_technique_prefers_commands_over_passive_comments(self):
        """When evidence text has noise/comments for one technique but commands for another,
        classify_technique must prefer the rule whose needles matched in commands/winning chain.
        """
        from backend.knowledge.memory_models import classify_technique
        evidence = (
            "<!-- Potential vectors: <script>alert(1)</script> document.cookie onerror= -->\n"
            "hydra -l admin -P /tmp/wordlist.txt http://target/login\n"
            "curl -d 'user=admin&pass=pass' http://target/login\n"
        )
        clf = classify_technique(
            evidence,
            category="web",
            commands=[
                "hydra -l admin -P /tmp/wordlist.txt http://target/login",
                "curl -d 'user=admin&pass=pass' http://target/login",
            ],
            winning_chain=[
                "hydra -l admin -P /tmp/wordlist.txt http://target/login",
                "curl -d 'user=admin&pass=pass' http://target/login",
            ]
        )
        self.assertEqual(clf["technique"], "Credential Brute-Force / Weak Auth")

    # ── Concern 1: stdout/command key isolation ──────────────────────────────

    def test_commands_list_never_contains_stdout(self):
        """Concern 1: raw_commands must be sourced ONLY from step['command'], never
        step['output'].  Even if stdout echoes a <script> tag (e.g. a curl
        response or a script print), it must not appear in the commands list
        passed to classify_technique — so the XSS rule can never be boosted by
        its own output.

        We verify this by confirming the 'commands' arg to classify_technique
        contains none of the output strings from exec_history.
        """
        # Patch via the module where the name is looked up at call time
        from unittest.mock import patch
        import backend.knowledge.experience_extractor as ee_module
        import backend.knowledge.memory_models as mm

        poisoned_output = "<script>document.cookie</script> onerror=alert(1)"
        exec_history = [
            {"agent": "a1",
             "command": "hydra -l admin -P /tmp/rock.txt http://target/login",
             "output": poisoned_output,  # stdout contains XSS keywords
             "note": "hydra run"},
            {"agent": "a1",
             "command": "curl -X POST http://target/login -d 'user=admin&pass=abc123'",
             "output": f"200 OK {CAPTURED_FLAG}",
             "note": ""},
            # Heredoc write — body contains <script> as a string literal, not an attack
            {"agent": "a1",
             "command": (
                 "cat > /tmp/check.py << 'EOF'\n"
                 "print('Page has <script> tags:', True)\n"
                 "EOF"
             ),
             "output": "",
             "note": "wrote probe script"},
        ]

        captured_commands: dict = {}

        original_clf = mm.classify_technique
        def _spy(evidence_text="", category="", *, commands=None, winning_chain=None):
            captured_commands["commands"] = list(commands or [])
            captured_commands["winning_chain"] = list(winning_chain or [])
            return original_clf(evidence_text, category, commands=commands, winning_chain=winning_chain)

        import types as _t
        board = _t.SimpleNamespace(
            run_id="r1", challenge_id="c1", challenge_name="Hashgate",
            category="WEB", difficulty="EASY",
            description="Login portal",
            target_scope="http://target",
            discovered_endpoints={"/", "/login"},
            extracted_headers={},
            observed_cookies={},
            deobfuscated_secrets=[],
            candidate_tokens=set(),
            candidate_usernames=set(),
            artifact_classification=None,
            execution_history=exec_history,
        )

        # patch.object on ee_module intercepts the call because classify_technique
        # is looked up as ee_module.classify_technique at runtime.
        with patch.object(ee_module, "classify_technique", side_effect=_spy):
            rec = ee_module.ExperienceExtractor().extract_from_board(board, flag=CAPTURED_FLAG, outcome="success")

        # (a) Structural: stdout strings must NEVER appear in the commands list
        for cmd in captured_commands.get("commands", []):
            self.assertNotIn("<script>document.cookie", cmd,
                "stdout containing <script>document.cookie must NOT appear in the commands list")
            self.assertNotIn("onerror=", cmd,
                "stdout containing onerror= must NOT appear in the commands list")

        # (b) Heredoc first-line stripping: the heredoc command's first line is
        # 'cat > /tmp/check.py << ...' — the body lines with '<script>' must be absent
        heredoc_entry = [c for c in captured_commands.get("commands", []) if c.startswith("cat >")]
        self.assertEqual(len(heredoc_entry), 1, "heredoc command should appear once (first line only)")
        self.assertNotIn("<script>", heredoc_entry[0],
            "heredoc body containing <script> must not appear in the first-line command token")

        # (c) Genuine shell commands must be present
        cmds = captured_commands.get("commands", [])
        self.assertTrue(any("hydra" in c for c in cmds), "hydra command should be in commands list")
        self.assertTrue(any("curl" in c for c in cmds), "curl command should be in commands list")

        # (d) End-to-end: only stdout contains XSS + heredoc body: must NOT classify as XSS
        self.assertNotEqual(rec.technique, "Cross-Site Scripting (XSS)",
            "XSS keywords only in stdout/heredoc body must not produce an XSS classification")

    # ── Concern 2: Messy real-world Hashgate replay ──────────────────────────

    def test_hashgate_messy_realworld_replay(self):
        """Concern 2: messy Hashgate-style log with --- HIDDEN CLUES --- sections,
        Python tracebacks, agent-generated script source that prints <script> tags,
        and incidental XSS keywords in comments/output.  The decisive exploit chain
        is purely credential brute-force; no XSS payload was ever attempted.
        """
        flag = CAPTURED_FLAG
        board_exec = [
            # Step 0: initial recon — stdout contains <script> from HTML
            {"agent": "a1",
             "command": "curl -s http://target.ctf/ -L",
             "output": (
                 "HTTP/1.1 200 OK\nServer: nginx\n"
                 "<html><head><script src='/static/app.js'></script></head>"
                 "<body><h1>Hashgate Portal</h1>"
                 "<form action='/login' method='POST'>...</form></body></html>"
             ),
             "note": "recon"},
            # Step 1: agent writes a helper script — script SOURCE contains <script> in a print
            {"agent": "a1",
             "command": (
                 "cat > /tmp/probe.py << 'EOF'\n"
                 "import requests\n"
                 "# probe for XSS potential (note: not the attack vector)\n"
                 "r = requests.get('http://target.ctf/')\n"
                 "print('Page has <script> tags:', '<script>' in r.text)\n"
                 "EOF"
             ),
             "output": "",
             "note": "wrote probe script"},
            # Step 2: run probe — stdout says xss not applicable
            {"agent": "a1",
             "command": "python3 /tmp/probe.py",
             "output": "Page has <script> tags: True\n--- HIDDEN CLUES ---\nLogin form requires valid credentials\n",
             "note": "probe output"},
            # Step 3: agent hits a traceback trying an XSS probe (not a real attack)
            {"agent": "a1",
             "command": "curl -s 'http://target.ctf/login?name=<script>alert(document.cookie)</script>'",
             "output": (
                 "Traceback (most recent call last):\n"
                 "  File 'probe.py', line 10, in <module>\n"
                 "    raise ValueError('XSS probe returned 403')\n"
                 "ValueError: XSS probe returned 403 Forbidden\n"
             ),
             "note": "xss probe — rejected"},
            # Steps 4-7: the actual exploit chain — brute-force
            {"agent": "a2",
             "command": "hydra -l admin -P /usr/share/wordlists/rockyou.txt http-post-form://target.ctf/login:user=^USER^&pass=^PASS^:Invalid",
             "output": "[80][http-post-form] host: target.ctf   login: admin   password: hashcat1\n1 valid password found",
             "note": "hydra brute-force — password found"},
            {"agent": "a2",
             "command": "curl -s -c /tmp/cookies.txt -X POST http://target.ctf/login -d 'user=admin&pass=hashcat1'",
             "output": "302 Found\nSet-Cookie: session=eyJhbGciOiJub25lIn0.abc.def\nLocation: /dashboard",
             "note": "credential — login"},
            {"agent": "a2",
             "command": "curl -s -b /tmp/cookies.txt http://target.ctf/dashboard",
             "output": f"200 OK\nWelcome, admin!\nYour flag: {flag}",
             "note": "Valid credentials found — flag retrieved"},
        ]

        import types as _t
        board = _t.SimpleNamespace(
            run_id="r-hashgate", challenge_id="c-hashgate",
            challenge_name="Hashgate", category="WEB", difficulty="MEDIUM",
            description="Secure portal. Hash-based auth. Investigate credential policies.",
            target_scope="http://target.ctf",
            discovered_endpoints={"/", "/login", "/dashboard"},
            extracted_headers={"Server": "nginx", "X-Powered-By": "Express"},
            observed_cookies={"session": "eyJhbGciOiJub25lIn0.abc.def"},
            deobfuscated_secrets=[],
            candidate_tokens=set(),
            candidate_usernames={"admin"},
            artifact_classification=None,
            execution_history=board_exec,
        )
        from backend.knowledge.experience_extractor import ExperienceExtractor
        rec = ExperienceExtractor().extract_from_board(board, flag=flag, outcome="success")

        self.assertEqual(rec.technique, "Credential Brute-Force / Weak Auth",
            f"Messy Hashgate replay classified as {rec.technique!r} — expected brute-force")
        self.assertIn("auth", rec.tags)
        self.assertIn("bruteforce", rec.tags)
        self.assertNotIn("xss", rec.tags,
            "XSS tag must not appear — the XSS probe was a failed/rejected step, not the exploit")

    # ── Concern 3: Existing correct classifications still land right ──────────

    def test_ssti_classification_unaffected_by_new_scoring(self):
        """Concern 3: The new multi-tier scoring must not regress a clean SSTI run.
        The standard _solved_board fixture has Jinja2 SSTI as the actual exploit chain
        ({{7*7}} in commands, flag in render output) — verify technique is still SSTI.
        """
        from backend.knowledge.memory_models import classify_technique

        # Construct evidence + structured args exactly as the extractor would for the SSTI fixture
        ssti_commands = [
            f"curl -s '{TARGET_URL}/render?name={{{{7*7}}}}'",
            f"curl -s '{TARGET_URL}/render?name={{{{config}}}}'",
        ]
        ssti_winning_chain = [
            f"curl -s '{TARGET_URL}/render?name={{{{7*7}}}}'",
            f"curl -s '{TARGET_URL}/render?name={{{{config}}}}'",
        ]
        evidence = "\n".join(ssti_commands * 3 + [
            "Server: Werkzeug/2.0 Python/3.9",
            "Hello, 49",
            "jinja render_template_string reflected",
        ])

        clf = classify_technique(
            evidence,
            category="web",
            commands=ssti_commands,
            winning_chain=ssti_winning_chain,
        )
        self.assertEqual(clf["technique"], "Server-Side Template Injection (SSTI)",
            f"SSTI run classified as {clf['technique']!r} — regression in new scoring")
        self.assertIn("ssti", clf["tags"])

    def test_sqli_classification_unaffected_by_new_scoring(self):
        """Concern 3 continued: SQL injection runs must still land as SQLi."""
        from backend.knowledge.memory_models import classify_technique

        sqli_commands = [
            "sqlmap -u 'http://target/search?q=1' --dbs",
            "curl -s \"http://target/search?q=1' UNION SELECT null,flag FROM secrets--\"",
        ]
        evidence = "\n".join(sqli_commands * 3 + [
            "SQL syntax error near UNION SELECT",
            "information_schema.tables",
        ])
        clf = classify_technique(
            evidence,
            category="web",
            commands=sqli_commands,
            winning_chain=sqli_commands,
        )
        self.assertEqual(clf["technique"], "SQL Injection",
            f"SQLi run classified as {clf['technique']!r} — regression in new scoring")
        self.assertIn("sqli", clf["tags"])

if __name__ == "__main__":
    unittest.main()
