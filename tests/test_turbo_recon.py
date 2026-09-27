"""
Tests for Phase 3: Turbo Recon (Pre-warmed parallel probing)
"""
import os
import unittest
import asyncio

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

class TestTurboRecon(unittest.TestCase):
    """Test TurboReconManager cache and probe logic."""

    def setUp(self):
        from backend.recon.turbo_recon import TurboReconManager
        self.recon = TurboReconManager()

    def test_cache_empty_initially(self):
        """Cache should return None for unknown challenge IDs."""
        result = self.recon.get_cached_recon("nonexistent-challenge")
        self.assertIsNone(result)

    def test_web_recon_caches_results(self):
        """Running turbo recon on a web target should cache results for the challenge."""
        loop = asyncio.new_event_loop()
        try:
            # Use a non-routable IP to force quick timeout (no real network needed)
            result = loop.run_until_complete(
                self.recon.start_turbo_recon("test-ch-web", "http://192.0.2.1:9999", "web")
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["challenge_id"], "test-ch-web")
            self.assertEqual(result["target"], "http://192.0.2.1:9999")
            self.assertIn("raw_summary", result)

            # Verify cache hit
            cached = self.recon.get_cached_recon("test-ch-web")
            self.assertIsNotNone(cached)
            self.assertEqual(cached["challenge_id"], "test-ch-web")
        finally:
            loop.close()

    def test_binary_recon_on_local_file(self):
        """Turbo recon on a local file path should extract binary info."""
        import tempfile
        loop = asyncio.new_event_loop()
        try:
            # Create a fake ELF file
            tmp = tempfile.NamedTemporaryFile(suffix=".elf", delete=False)
            tmp.write(b"\x7fELF" + b"\x00" * 100 + b"FLAG{test_binary_strings}" + b"\x00" * 50)
            tmp.close()

            result = loop.run_until_complete(
                self.recon.start_turbo_recon("test-ch-bin", tmp.name, "pwn")
            )
            self.assertIsNotNone(result)
            self.assertIn("binary_info", result)

            os.unlink(tmp.name)
        finally:
            loop.close()

    def test_multi_target_splitting(self):
        """Multi-target addresses joined with '+' should all be probed."""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.recon.start_turbo_recon("test-ch-multi", "http://192.0.2.1:80 + http://192.0.2.2:80", "web")
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["challenge_id"], "test-ch-multi")
        finally:
            loop.close()

if __name__ == "__main__":
    unittest.main()
