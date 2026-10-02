"""Workstream C: Gemini is reserved for vision + report authoring ONLY.

These tests fail if Gemini is (re)introduced into any solving/fallback chain, and
verify the runtime guard in route_request skips Gemini for a solving capability even
if a stray chain entry puts it there.
"""
import os
import sys
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.providers.router import ModelRouter, model_router

SOLVING_CAPABILITIES = [
    "recon", "directory_enumeration", "web_analysis", "web_testing",
    "code_analysis", "reverse_engineering", "fast_reasoning",
    "general_reasoning", "verification",
]
ALLOWED = {"report_generation", "vision_read"}


class TestGeminiRestriction(unittest.TestCase):

    def test_gemini_present_in_exactly_report_and_vision(self):
        chains_with_gemini = {
            cap for cap, chain in ModelRouter.DEFAULT_ROUTING_MAP.items() if "gemini" in chain
        }
        self.assertEqual(chains_with_gemini, ALLOWED)

    def test_gemini_absent_from_every_solving_chain(self):
        for cap in SOLVING_CAPABILITIES:
            self.assertNotIn(
                "gemini", ModelRouter.DEFAULT_ROUTING_MAP.get(cap, []),
                f"Gemini must not appear in solving chain '{cap}'",
            )

    def test_allowed_capabilities_constant_matches(self):
        self.assertEqual(set(ModelRouter.GEMINI_ALLOWED_CAPABILITIES), ALLOWED)

    def test_runtime_guard_skips_gemini_for_solving_even_if_in_chain(self):
        """Force-inject gemini into a solving chain; the guard must still skip it."""
        fake_gemini = MagicMock()
        fake_gemini.name = "gemini"
        fake_gemini.is_paid = True
        fake_gemini.is_available = AsyncMock(return_value=True)
        fake_gemini.generate_response = AsyncMock()

        original_providers = model_router.providers
        original_recon = list(ModelRouter.DEFAULT_ROUTING_MAP["recon"])
        try:
            # Only gemini is "registered"; inject it into the recon chain.
            model_router.providers = {"gemini": fake_gemini}
            if "gemini" not in ModelRouter.DEFAULT_ROUTING_MAP["recon"]:
                ModelRouter.DEFAULT_ROUTING_MAP["recon"].insert(0, "gemini")

            resp = asyncio.run(model_router.route_request(prompt="enumerate the target", capability="recon"))

            # Guard skipped gemini; no other provider is registered -> clean refusal,
            # and gemini.generate_response was NEVER invoked for a solving task.
            self.assertTrue(resp.is_refusal)
            fake_gemini.generate_response.assert_not_called()
        finally:
            model_router.providers = original_providers
            ModelRouter.DEFAULT_ROUTING_MAP["recon"] = original_recon

    def test_runtime_guard_allows_gemini_for_vision_read(self):
        """The same guard must permit gemini on an allowed capability (vision_read)."""
        fake_gemini = MagicMock()
        fake_gemini.name = "gemini"
        fake_gemini.is_paid = True
        fake_gemini.is_available = AsyncMock(return_value=True)
        fake_gemini.generate_response = AsyncMock(
            return_value=MagicMock(is_refusal=False, content="flag{seen}",
                                   estimated_cost_usd=0.0, model_name="gemini-3.6-flash",
                                   provider_name="gemini")
        )
        original_providers = model_router.providers
        try:
            model_router.providers = {"gemini": fake_gemini}
            resp = asyncio.run(model_router.route_request(prompt="read this image", capability="vision_read"))
            self.assertFalse(resp.is_refusal)
            fake_gemini.generate_response.assert_called()
        finally:
            model_router.providers = original_providers


if __name__ == "__main__":
    unittest.main()
