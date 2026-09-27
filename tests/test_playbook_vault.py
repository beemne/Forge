"""
Tests for Phase 2: Playbook Vault (File-based YAML storage, FTS5 search, synthesis flywheel)
"""
import os
import sys
import unittest
import tempfile
import shutil
import time

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

class TestPlaybookVault(unittest.TestCase):
    """Test PlaybookVault YAML storage, FTS5 search index, and synthesis flywheel."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="forge_playbooks_test_")
        from backend.knowledge.playbook_vault import PlaybookVault, PlaybookSchema
        self.vault = PlaybookVault(base_dir=self.temp_dir)
        self.PlaybookSchema = PlaybookSchema

    def tearDown(self):
        try:
            self.vault.db_conn.close()
        except Exception:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_playbook(self):
        """Save a playbook as YAML and reload it."""
        pb = self.PlaybookSchema(
            id="test-web-sqli-union",
            category="web",
            tags=["sqli", "union", "web"],
            trigger_signatures=["SQL syntax", "UNION SELECT"],
            exploit_template="' UNION SELECT 1,2,flag FROM flags--",
            notes="Basic union-based SQL injection.",
            source="curated",
            confidence_score=1.0
        )
        path = self.vault.save_playbook(pb)
        self.assertTrue(os.path.exists(path), "YAML file should exist on disk")

        loaded = self.vault.load_playbook("test-web-sqli-union")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.id, "test-web-sqli-union")
        self.assertIn("sqli", loaded.tags)

    def test_fts5_search_finds_matching_playbook(self):
        """FTS5 search should find playbooks matching trigger_signatures keywords."""
        pb = self.PlaybookSchema(
            id="test-web-jinja2-ssti",
            category="web",
            tags=["ssti", "jinja2", "flask"],
            trigger_signatures=["Werkzeug", "Flask", "render_template_string"],
            exploit_template="{{lipsum.__globals__}}",
            source="curated",
            confidence_score=1.0
        )
        self.vault.save_playbook(pb)

        results = self.vault.search_playbooks("Werkzeug Flask SSTI")
        self.assertTrue(len(results) > 0, "Should find at least one matching playbook")
        self.assertEqual(results[0].id, "test-web-jinja2-ssti")

    def test_fts5_search_latency_under_50ms(self):
        """Search should complete in under 50ms for a modest playbook set."""
        # Insert 20 playbooks
        for i in range(20):
            pb = self.PlaybookSchema(
                id=f"perf-test-{i}",
                category="web",
                tags=[f"tag{i}", "web", "performance"],
                trigger_signatures=[f"sig{i}", "http", "test"],
                exploit_template=f"echo 'exploit {i}'",
                source="curated",
                confidence_score=1.0
            )
            self.vault.save_playbook(pb)

        start = time.time()
        results = self.vault.search_playbooks("http web performance")
        elapsed_ms = (time.time() - start) * 1000

        self.assertLess(elapsed_ms, 50, f"FTS5 search took {elapsed_ms:.1f}ms, should be <50ms")

    def test_auto_generated_low_confidence_not_in_default_search(self):
        """Auto-generated playbooks at low confidence should NOT appear in default search results."""
        pb = self.PlaybookSchema(
            id="gen-web-test-lowconf",
            category="web",
            tags=["auto_learned", "web"],
            trigger_signatures=["unique_lowconf_signature"],
            exploit_template="echo low confidence",
            source="generated",
            confidence_score=0.3,
            times_used=0,
            is_promoted=False
        )
        self.vault.save_playbook(pb)

        # Default search should NOT find it
        results = self.vault.search_playbooks("unique_lowconf_signature", include_unpromoted=False)
        self.assertEqual(len(results), 0, "Low-confidence auto-generated playbook should NOT be in default search")

        # Explicitly including unpromoted SHOULD find it
        results_all = self.vault.search_playbooks("unique_lowconf_signature", include_unpromoted=True)
        self.assertTrue(len(results_all) > 0, "Should find when include_unpromoted=True")

    def test_auto_promote_after_reuse_threshold(self):
        """Auto-generated playbook should be promoted after reaching reuse threshold."""
        pb = self.PlaybookSchema(
            id="gen-web-promo-test",
            category="web",
            tags=["auto_learned"],
            trigger_signatures=["promo_test_sig"],
            exploit_template="echo promoted",
            source="generated",
            confidence_score=0.3,
            times_used=0,
            is_promoted=False
        )
        self.vault.save_playbook(pb)

        # Record 3 successful uses (threshold)
        for _ in range(3):
            self.vault.record_playbook_use("gen-web-promo-test", success=True)

        # Reload and verify promotion
        promoted = self.vault.load_playbook("gen-web-promo-test")
        self.assertTrue(promoted.is_promoted, "Playbook should be auto-promoted after 3 successful uses")
        self.assertGreaterEqual(promoted.confidence_score, 0.8)

    def test_synthesis_parameterization(self):
        """Synthesis flywheel should parameterize target-specific values."""
        pb = self.vault.synthesize_from_run(
            challenge_id="test-ch-001",
            challenge_title="Web Login Bypass",
            category="web",
            target_endpoint="http://10.10.14.5:8080",
            winning_payload="curl http://10.10.14.5:8080/admin?user=admin' OR 1=1--",
            winning_commands=["curl http://10.10.14.5:8080/admin?user=admin' OR 1=1--"],
            flag="picoCTF{sql_1nj3ct10n_ez}"
        )
        self.assertIsNotNone(pb, "Synthesis should produce a playbook")
        self.assertEqual(pb.source, "generated")
        self.assertEqual(pb.confidence_score, 0.3)
        self.assertIn("{TARGET_URL}", pb.exploit_template, "Should parameterize target URL")
        self.assertNotIn("10.10.14.5", pb.exploit_template, "Should NOT contain raw IP")

    def test_empty_query_returns_empty(self):
        """Empty search query should return empty results."""
        results = self.vault.search_playbooks("")
        self.assertEqual(len(results), 0)

    def test_reload_index_picks_up_new_files(self):
        """Reloading index should discover newly added YAML files."""
        import yaml
        # Write a playbook directly to disk
        cat_dir = os.path.join(self.temp_dir, "web")
        os.makedirs(cat_dir, exist_ok=True)
        data = {
            "id": "manual-disk-write",
            "category": "web",
            "tags": ["manual"],
            "trigger_signatures": ["disk_test_sig"],
            "exploit_template": "echo manual",
            "source": "curated",
            "confidence_score": 1.0,
            "times_used": 0,
            "success_rate": 1.0,
            "is_promoted": True,
            "notes": "Manual test",
            "created_at": "2026-01-01T00:00:00"
        }
        with open(os.path.join(cat_dir, "manual-disk-write.yaml"), "w") as f:
            yaml.safe_dump(data, f)

        self.vault.reload_index()
        results = self.vault.search_playbooks("disk_test_sig")
        self.assertTrue(len(results) > 0, "Reload should pick up new YAML files")

if __name__ == "__main__":
    unittest.main()
