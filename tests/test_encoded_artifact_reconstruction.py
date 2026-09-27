"""Tests for encoded-artifact reconstruction & escalation (Task #3).

Covers the deterministic reconstruction module (backend/agents/artifact_reconstruction.py)
and its wiring into the LIVE swarm engine (backend/agents/swarm_orchestrator.py):

  * ASCII binary  -> JPEG / PNG reconstruction (byte-identical)
  * invalid binary rejected; whitespace-separated bits handled
  * normal text is NOT misclassified as encoded
  * '#'/'.' ASCII bitmap detection
  * a reconstructed artifact enters the pipeline (persisted + registered + attached)
  * a "no flag" step does NOT terminate the run while an un-analyzed derived
    artifact exists (escalation surfaces in build_history_context)
  * the Binary Digits regression (#14): after rebuilding the JPEG, FORGE CONTINUES
    the workflow instead of concluding "no flag" — and fabricates nothing.

No demo/mock data, no hard-coded challenge or flag: every artifact used here is
constructed in-test from real magic bytes, never a stored fixture flag.
"""

import hashlib
import os
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

from backend.agents.artifact_reconstruction import (
    ReconState,
    detect_encoded_stream,
    reconstruct_from_text,
    persist_derived_artifact,
    is_image_type,
    detect_ascii_bitmap,
    _bits_to_bytes,
)
from backend.agents.swarm_orchestrator import SwarmOrchestrator
from backend.agents.swarm_state import SwarmBlackboard

# --------------------------------------------------------------------------- #
# Helpers — build REAL byte payloads from magic signatures (no stored fixtures)
# --------------------------------------------------------------------------- #

def _bytes_to_bits(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)

def _make_jpeg(payload_len: int = 600) -> bytes:
    """A byte blob that begins with the real JPEG/JFIF magic and ends with EOI.

    Not necessarily a decodable image — the reconstruction + magic-byte gate only
    needs the FF D8 FF signature to classify it as 'jpeg'. PIL degrades gracefully.
    """
    head = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    body = bytes((i * 7 + 3) & 0xFF for i in range(max(0, payload_len - len(head) - 2)))
    return head + body + b"\xff\xd9"

def _make_png(payload_len: int = 300) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    body = bytes((i * 5 + 1) & 0xFF for i in range(max(0, payload_len - len(sig))))
    return sig + body

# --------------------------------------------------------------------------- #
# Detection + reconstruction (pure, deterministic)
# --------------------------------------------------------------------------- #

class TestReconstructionCore(unittest.TestCase):

    def test_13a_ascii_binary_to_jpeg_byte_identical(self):
        jpeg = _make_jpeg()
        text = "Decoded stream follows:\n" + _bytes_to_bits(jpeg)
        outcome = reconstruct_from_text(text, origin_label="unit:jpeg")
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.scheme, "ascii_binary")
        self.assertEqual(outcome.artifact_type, "jpeg")
        self.assertTrue(outcome.is_file)
        self.assertTrue(outcome.should_persist)
        self.assertEqual(outcome.state, ReconState.ARTIFACT_TYPE_IDENTIFIED)
        # Byte-for-byte exact — the core guarantee (no text-mode corruption).
        self.assertEqual(outcome.raw_bytes, jpeg)
        self.assertEqual(outcome.output_sha256, hashlib.sha256(jpeg).hexdigest())
        self.assertIn("exiftool", outcome.recommended_tools)

    def test_13b_ascii_binary_to_png_byte_identical(self):
        png = _make_png()
        text = _bytes_to_bits(png)
        outcome = reconstruct_from_text(text)
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.artifact_type, "png")
        self.assertTrue(outcome.is_file)
        self.assertEqual(outcome.raw_bytes, png)

    def test_13c_invalid_binary_rejected(self):
        # Not a multiple of 8 bits -> no reconstruction.
        self.assertIsNone(_bits_to_bytes("0101"))
        self.assertIsNone(_bits_to_bytes("01010101010"))       # 11 bits
        # Non-{0,1} characters -> rejected.
        self.assertIsNone(_bits_to_bytes("0101010X01010101"))
        # A long non-multiple-of-8 wall yields no usable detection.
        wall = "01" * 35 + "0"       # 71 chars, %8 != 0
        self.assertIsNone(reconstruct_from_text(wall))

    def test_13d_whitespace_separated_bits_handled(self):
        self.assertEqual(_bits_to_bytes("0100100001101001"), b"Hi")
        jpeg = _make_jpeg(200)
        bits = _bytes_to_bits(jpeg)
        # Insert whitespace (spaces + newlines) — must be stripped, bytes preserved.
        spaced = "\n".join(bits[i:i + 40] for i in range(0, len(bits), 40))
        outcome = reconstruct_from_text("dump:\n" + spaced)
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.artifact_type, "jpeg")
        self.assertEqual(outcome.raw_bytes, jpeg)

    def test_13e_normal_text_not_misclassified(self):
        prose = ("Nmap scan report for host. 22 open ssh. 80 open http. "
                 "No flag here yet, keep going team. Enumerate the web root next.")
        self.assertEqual(detect_encoded_stream(prose), [])
        self.assertIsNone(reconstruct_from_text(prose))

    def test_13e2_hash_like_hex_not_persisted_as_file(self):
        # A 64-char hex string (looks like a sha256) must NOT become a derived file.
        h = hashlib.sha256(b"whatever").hexdigest()
        outcome = reconstruct_from_text(f"digest = {h}")
        # Either no outcome, or an outcome that is explicitly not a persisted file.
        if outcome is not None:
            self.assertFalse(outcome.should_persist)
            self.assertFalse(outcome.is_file)

    def test_13f_ascii_bitmap_detected(self):
        # A small monochrome 'A' drawn in '#' and '.' — >=4 near-rectangular rows.
        art = "\n".join([
            ".###.",
            "#...#",
            "#####",
            "#...#",
            "#...#",
        ])
        info = detect_ascii_bitmap(art)
        self.assertTrue(info["is_bitmap"])
        self.assertEqual(info["rows"], 5)
        self.assertEqual(info["cols"], 5)
        # A normal paragraph is not a bitmap.
        self.assertFalse(detect_ascii_bitmap("just a normal line of prose text here").get("is_bitmap"))

    def test_ascii_binary_opaque_bytes_preserved_as_unknown(self):
        # A deliberate wall of bits that rebuilds to opaque (non-magic) bytes is still
        # a real artifact -> preserved + escalated as UNKNOWN/RECONSTRUCTED. Use clearly
        # non-printable bytes (no magic hit) so classification is 'generic_binary'.
        raw = bytes([0x00, 0x01, 0x02, 0x03] * 24)   # 96 opaque bytes, no known magic
        outcome = reconstruct_from_text(_bytes_to_bits(raw))
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.scheme, "ascii_binary")
        self.assertEqual(outcome.artifact_type, "generic_binary")
        self.assertEqual(outcome.state, ReconState.ARTIFACT_RECONSTRUCTED)
        self.assertTrue(outcome.should_persist)
        self.assertEqual(outcome.raw_bytes, raw)

    def test_base64_readable_text_is_scan_only_not_a_file(self):
        import base64
        token = base64.b64encode(b"the secret note is not a flag here").decode()
        outcome = reconstruct_from_text(f"payload: {token}")
        # Readable text -> returned for flag-scanning, but never persisted as a file.
        if outcome is not None:
            self.assertFalse(outcome.should_persist)
            self.assertTrue(outcome.decoded_text)

    def test_is_image_type(self):
        self.assertTrue(is_image_type("jpeg"))
        self.assertTrue(is_image_type("png"))
        self.assertFalse(is_image_type("elf"))
        self.assertFalse(is_image_type("generic_binary"))

# --------------------------------------------------------------------------- #
# Persistence with provenance (requirement #5)
# --------------------------------------------------------------------------- #

class TestPersistence(unittest.TestCase):

    def test_13g_persist_writes_file_and_provenance_inside_workspace(self):
        jpeg = _make_jpeg()
        outcome = reconstruct_from_text(_bytes_to_bits(jpeg))
        self.assertIsNotNone(outcome)
        with tempfile.TemporaryDirectory() as wd:
            path = persist_derived_artifact(outcome, wd)
            self.assertIsNotNone(path)
            # Inside the FORGE workspace, under forge_derived/ (never /tmp-only).
            self.assertTrue(os.path.isfile(path))
            self.assertIn("forge_derived", path.replace("\\", "/"))
            self.assertTrue(os.path.commonpath([os.path.abspath(path), os.path.abspath(wd)]) == os.path.abspath(wd))
            # Byte-identical on disk.
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), jpeg)
            # Provenance sidecar with the security note (analyze, do not execute).
            prov = path + ".provenance.json"
            self.assertTrue(os.path.isfile(prov))
            import json
            with open(prov) as fh:
                meta = json.load(fh)
            self.assertEqual(meta["artifact_type"], "jpeg")
            self.assertIn("DO NOT EXECUTE", meta["security_note"])
            self.assertEqual(meta["output_sha256"], hashlib.sha256(jpeg).hexdigest())

# --------------------------------------------------------------------------- #
# Live-engine escalation (requirements #6/#11/#12/#14)
# --------------------------------------------------------------------------- #

class TestSwarmEscalation(unittest.IsolatedAsyncioTestCase):

    def _fresh_board(self) -> SwarmBlackboard:
        return SwarmBlackboard("chal-recon-test", "run-recon-test", "local-artifact")

    async def test_13g_reconstructed_artifact_enters_pipeline(self):
        orch = SwarmOrchestrator()
        board = self._fresh_board()
        board.agent_ids.append("agent-0")
        jpeg = _make_jpeg()
        text = "Agent decoded the stream:\n" + _bytes_to_bits(jpeg)
        with tempfile.TemporaryDirectory() as wd:
            await orch._reconstruct_and_escalate(text, board, "agent-0", wd, origin_label="unit")
            # Registered as evidence.
            self.assertEqual(len(board.derived_artifacts), 1)
            rec = board.derived_artifacts[0]
            self.assertEqual(rec["artifact_type"], "jpeg")
            self.assertFalse(rec["analyzed"])
            # Became a first-class analyzable input.
            dpath = rec["derived_path"]
            self.assertIn(dpath, board.attached_file_paths)
            self.assertTrue(os.path.isfile(dpath))
            # Still needs analysis.
            self.assertTrue(board.has_unanalyzed_derived())

    async def test_dedup_same_input_is_noop(self):
        orch = SwarmOrchestrator()
        board = self._fresh_board()
        jpeg = _make_jpeg()
        text = _bytes_to_bits(jpeg)
        with tempfile.TemporaryDirectory() as wd:
            await orch._reconstruct_and_escalate(text, board, "agent-0", wd)
            await orch._reconstruct_and_escalate(text, board, "agent-0", wd)
            self.assertEqual(len(board.derived_artifacts), 1)

    async def test_13h_no_flag_does_not_terminate_with_unanalyzed_artifact(self):
        orch = SwarmOrchestrator()
        board = self._fresh_board()
        board.agent_ids.append("agent-0")
        jpeg = _make_jpeg()
        with tempfile.TemporaryDirectory() as wd:
            await orch._reconstruct_and_escalate(_bytes_to_bits(jpeg), board, "agent-0", wd)
            # The shared channel EVERY agent sees must carry the escalation directive.
            ctx = board.build_history_context("agent-0")
            self.assertIn("RECONSTRUCTED DERIVED ARTIFACTS", ctx)
            self.assertIn("ANALYSIS REQUIRED", ctx)
            self.assertIn("WHETHER", ctx)          # WHAT/HOW/WHETHER framing present
            self.assertIn("never execute", ctx.lower())
            # The finish-without-flag guard keys off exactly this predicate.
            self.assertTrue(board.has_unanalyzed_derived())
            # Once analyzed, the run is free to conclude.
            board.mark_derived_analyzed(board.derived_artifacts[0]["derived_path"])
            self.assertFalse(board.has_unanalyzed_derived())

    async def test_14_binary_digits_regression_continues_workflow(self):
        """#14 — the exact Binary Digits failure mode: a wall of ASCII bits that is
        really a JPEG. FORGE must reconstruct it AND CONTINUE (not return 'no flag'),
        and must not fabricate a flag from opaque image bytes."""
        orch = SwarmOrchestrator()
        board = self._fresh_board()
        board.agent_ids.append("agent-0")
        jpeg = _make_jpeg(8875 % 4096 + 512)   # sizeable, JPEG-magic blob
        # Simulate the real tool output: a "no flag" line followed by the bit wall.
        tool_output = "Flag candidate: NONE\nRaw file contents:\n" + _bytes_to_bits(jpeg)
        with tempfile.TemporaryDirectory() as wd:
            await orch._reconstruct_and_escalate(tool_output, board, "agent-0", wd,
                                                 origin_label="tool_output:cat binary.txt")
            # 1) The JPEG was reconstructed and preserved as evidence.
            self.assertEqual(len(board.derived_artifacts), 1)
            self.assertEqual(board.derived_artifacts[0]["artifact_type"], "jpeg")
            with open(board.derived_artifacts[0]["derived_path"], "rb") as fh:
                self.assertEqual(fh.read(), jpeg)
            # 2) The workflow CONTINUES: an un-analyzed artifact exists and the
            #    escalation is visible to every agent -> the run cannot stop at "no flag".
            self.assertTrue(board.has_unanalyzed_derived())
            self.assertIn("ANALYSIS REQUIRED", board.build_history_context("agent-0"))
            # 3) FORGE fabricated nothing: opaque JPEG bytes are not printable, so no
            #    flag candidate was invented (flags come only from real readable output).
            self.assertEqual(board.flag_candidates, [])

if __name__ == "__main__":
    unittest.main()
