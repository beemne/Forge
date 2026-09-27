import os
import sys
import base64
import tempfile
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.providers.real_providers import GeminiProvider, _detect_mime_type
from backend.providers.router import ModelRouter, model_router
from backend.tools.manager import tool_manager
from backend.tools.registry import tool_registry
from backend.agents.swarm_orchestrator import SwarmOrchestrator
from backend.agents.swarm_state import SwarmBlackboard

class TestVisionCapability(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    # ── Test (a): Gemini payload with image (inline_data, base64, mime type) ──

    async def test_gemini_payload_with_image_png(self):
        """Image path produces a payload containing inline_data with base64 bytes and correct MIME type."""
        provider = GeminiProvider(api_key="test-key-123")

        # Create a dummy PNG file (with valid PNG magic bytes)
        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
        img_path = os.path.join(self.workdir, "test_artifact.png")
        with open(img_path, "wb") as f:
            f.write(png_bytes)

        captured_payload = None

        async def fake_post_json(url, headers, payload, timeout=30.0):
            nonlocal captured_payload
            captured_payload = payload
            return {"candidates": [{"content": {"parts": [{"text": "Extracted: flag{sample_flag}"}]}}]}, {}

        with patch.object(provider, "_post_json", side_effect=fake_post_json):
            resp = await provider.generate_response(
                prompt="Extract any readable text",
                image_path=img_path
            )

        self.assertFalse(resp.is_refusal)
        self.assertEqual(resp.content, "Extracted: flag{sample_flag}")
        self.assertIsNotNone(captured_payload)

        contents = captured_payload.get("contents", [])
        self.assertEqual(len(contents), 1)
        parts = contents[0].get("parts", [])
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], {"text": "Extract any readable text"})
        self.assertEqual(parts[1]["inline_data"]["mime_type"], "image/png")
        self.assertEqual(parts[1]["inline_data"]["data"], base64.b64encode(png_bytes).decode("utf-8"))

    async def test_gemini_payload_with_image_jpeg(self):
        """JPEG magic bytes and extension correctly detect image/jpeg."""
        provider = GeminiProvider(api_key="test-key-123")

        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"
        img_path = os.path.join(self.workdir, "test_artifact.jpg")
        with open(img_path, "wb") as f:
            f.write(jpeg_bytes)

        captured_payload = None

        async def fake_post_json(url, headers, payload, timeout=30.0):
            nonlocal captured_payload
            captured_payload = payload
            return {"candidates": [{"content": {"parts": [{"text": "Extracted text"}]}}]}, {}

        with patch.object(provider, "_post_json", side_effect=fake_post_json):
            await provider.generate_response(
                prompt="Extract text",
                image_path=img_path
            )

        self.assertIsNotNone(captured_payload)
        parts = captured_payload["contents"][0]["parts"]
        self.assertEqual(parts[1]["inline_data"]["mime_type"], "image/jpeg")
        self.assertEqual(parts[1]["inline_data"]["data"], base64.b64encode(jpeg_bytes).decode("utf-8"))

    # ── Test (b): Text-only backwards compatibility ──────────────────────────

    async def test_gemini_text_only_payload_unchanged(self):
        """A text-only call (no image) produces byte-for-byte identical payload structure."""
        provider = GeminiProvider(api_key="test-key-123")
        captured_payload = None

        async def fake_post_json(url, headers, payload, timeout=30.0):
            nonlocal captured_payload
            captured_payload = payload
            return {"candidates": [{"content": {"parts": [{"text": "Response text"}]}}]}, {}

        with patch.object(provider, "_post_json", side_effect=fake_post_json):
            await provider.generate_response(prompt="Hello Gemini")

        self.assertIsNotNone(captured_payload)
        expected_payload = {
            "contents": [{"parts": [{"text": "Hello Gemini"}]}],
            "safetySettings": GeminiProvider.SAFETY_SETTINGS
        }
        self.assertEqual(captured_payload, expected_payload)

    # ── Test (c): End-to-end reachability from analyze_derived_artifact ──────

    async def test_vision_read_end_to_end_from_derived_artifact_escalation(self):
        """vision_read capability is reachable from analyze_derived_artifact end-to-end,

        records the candidate in flag_candidates (NOT auto-promoted), and marks the artifact analyzed.
        """
        board = SwarmBlackboard("chal-vision-1", "run-vision-1", "http://target.local")
        orchestrator = SwarmOrchestrator()

        # Create a dummy image artifact
        img_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
        img_path = os.path.join(self.workdir, "derived_001.png")
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        # Register derived artifact on board as unanalyzed
        board.derived_artifacts.append({
            "derived_path": img_path,
            "artifact_type": "png",
            "byte_count": len(img_bytes),
            "scheme": "ascii_binary",
            "analyzed": False,
            "recommended_tools": ["file", "strings"]
        })

        mock_vision_response = MagicMock()
        mock_vision_response.is_refusal = False
        mock_vision_response.content = "Here is the exact text found in image:\nflag{vision_extracted_secret_flag_123}"
        mock_vision_response.model_name = "gemini-3.6-flash"

        async def fake_route_request(*args, **kwargs):
            self.assertEqual(kwargs.get("capability"), "vision_read")
            self.assertEqual(kwargs.get("image_path"), img_path)
            return mock_vision_response

        with patch("backend.providers.router.model_router.route_request", side_effect=fake_route_request):
            # Run analyze_derived_artifact
            result = await orchestrator.analyze_derived_artifact(board, img_path, "worker_1")

        self.assertIsNotNone(result)
        self.assertIn("flag{vision_extracted_secret_flag_123}", result)

        # Verify candidate flag was recorded and verified by AnswerResolver
        candidates = [c["flag"] for c in board.flag_candidates]
        self.assertIn("flag{vision_extracted_secret_flag_123}", candidates)
        self.assertEqual(board.flag_candidates[0]["source"], "vision_read")

        # Verify it was resolved/verified and promoted to captured flag
        self.assertEqual(board.flag_captured, "flag{vision_extracted_secret_flag_123}")
        self.assertTrue(board.flag_event.is_set())

        # Verify artifact marked as analyzed
        analyzed_flags = [d["analyzed"] for d in board.derived_artifacts if d.get("derived_path") == img_path]
        self.assertEqual(analyzed_flags, [True])

    # ── Test (d): Paid model gating & refusal surfacing ──────────────────────

    async def test_vision_read_blocked_when_paid_models_disabled(self):
        """When PAID_MODEL_ALLOWED is False, vision_read returns a clear refusal message."""
        router = ModelRouter()
        router.set_paid_allowed(False)

        # Register a mock gemini provider
        mock_gemini = AsyncMock()
        mock_gemini.name = "gemini"
        mock_gemini.is_paid = True
        mock_gemini.is_available = AsyncMock(return_value=True)
        router.register_provider("gemini", mock_gemini)

        resp = await router.route_request(
            prompt="Extract text",
            capability="vision_read",
            image_path="some_file.png"
        )

        self.assertTrue(resp.is_refusal)
        self.assertIn("PAID_MODEL_ALLOWED=False", resp.refusal_reason)
        self.assertIn("vision_read", resp.refusal_reason)

    # ── Test (e): Tool manager capability execution ──────────────────────────

    async def test_tool_manager_execute_capability_vision_read(self):
        """tool_manager.execute_capability('vision_read', target=path) invokes route_request."""
        img_path = os.path.join(self.workdir, "test_cap.png")
        with open(img_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")

        mock_resp = MagicMock()
        mock_resp.is_refusal = False
        mock_resp.content = "Text: picoCTF{vision_read_worked}"

        with patch("backend.providers.router.model_router.route_request", return_value=mock_resp) as mock_route:
            res = await tool_manager.execute_capability("vision_read", target=img_path)

            mock_route.assert_called_once()
            self.assertEqual(res.status, "SUCCESS")
            self.assertIn("picoCTF{vision_read_worked}", res.stdout)
            self.assertEqual(res.tool_name, "vision_read")
            self.assertEqual(res.capability, "vision_read")

if __name__ == "__main__":
    unittest.main()
