"""Encoded-artifact reconstruction & escalation (deterministic, zero-LLM).

Recognizes when a challenge artifact or tool output actually contains an *encoded
representation of another file* (the classic "a wall of ASCII 0/1 that is really a
JPEG" trick), reconstructs the real bytes, identifies the file type by magic bytes,
and hands back a structured outcome so the swarm can PRESERVE it as evidence and
ESCALATE it into the normal analysis pipeline instead of concluding "no flag".

Design constraints (FORGE rules):
- Deterministic and dependency-light: reuses ``artifact_classifier`` for magic-byte
  typing and byte-safe writes. PIL / pytesseract are OPTIONAL — every path degrades
  gracefully when they are absent (no hard tesseract/sudo dependency).
- No demo/mock data, no hard-coded challenge or flag: this is a general capability.
- Never executes a reconstructed artifact. It only reads/writes bytes and (optionally)
  reads image metadata. Analysis of the derived file happens through the existing
  capability-gated pipeline, never by running it here.

The module is intentionally free of any dependency on the orchestrator so it stays
unit-testable in isolation; flag extraction and task escalation are performed by the
caller (which owns the single FLAG_REGEX and the shared blackboard).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from backend.agents.artifact_classifier import (
    _classify_magic, save_artifact_binary,
)

logger = logging.getLogger("forge.artifact_reconstruction")


# --------------------------------------------------------------------------- #
# States (requirement #12) — a small, explicit vocabulary the swarm can surface
# --------------------------------------------------------------------------- #

class ReconState:
    UNKNOWN_ARTIFACT = "UNKNOWN_ARTIFACT"
    ENCODED_DATA_DETECTED = "ENCODED_DATA_DETECTED"
    ARTIFACT_RECONSTRUCTED = "ARTIFACT_RECONSTRUCTED"
    ARTIFACT_TYPE_IDENTIFIED = "ARTIFACT_TYPE_IDENTIFIED"
    DERIVED_ANALYSIS_PENDING = "DERIVED_ANALYSIS_PENDING"
    DERIVED_ANALYSIS_COMPLETE = "DERIVED_ANALYSIS_COMPLETE"
    FLAG_CANDIDATE_FOUND = "FLAG_CANDIDATE_FOUND"


# Map recognized magic labels to a sensible on-disk extension for the derived file.
_EXT_FOR_TYPE: Dict[str, str] = {
    "jpeg": ".jpg", "png": ".png", "gif": ".gif", "pdf": ".pdf",
    "zip": ".zip", "gzip": ".gz", "bzip2": ".bz2", "xz": ".xz",
    "rar": ".rar", "7zip": ".7z", "elf": ".elf", "pe": ".exe",
    "macho_32le": ".macho", "macho_64le": ".macho", "java_class": ".class",
    "riff": ".riff", "ms_ole": ".ole", "sqlite": ".sqlite", "wasm": ".wasm",
    "pcap": ".pcap", "pcap_le": ".pcap", "pcap_be": ".pcap", "pcapng": ".pcapng",
    "generic_binary": ".bin",
}

# Labels that mean "this really is another file" (a genuine magic-byte hit).
_TEXTUAL_LABELS = {"text", "unknown"}

# Whitespace that is SAFE to strip from an encoded stream (requirement #3).
_SAFE_WHITESPACE = " \t\r\n\f\v"

# Minimum useful reconstruction size, in bytes.
_DEFAULT_MIN_BYTES = 8


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

@dataclass
class EncodedDetection:
    """A candidate encoded region found in text (before reconstruction)."""
    scheme: str                 # "ascii_binary" | "hex" | "base64"
    confidence: float           # 0..1 heuristic (pre-reconstruction)
    token: str                  # the cleaned stream (whitespace removed), truncated for storage
    full_token: str = ""        # cleaned stream, full length (not stored in evidence)
    est_bytes: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"scheme": self.scheme, "confidence": round(self.confidence, 3),
                "est_bytes": self.est_bytes, "reason": self.reason,
                "sample": self.token[:80]}


_RE_BINARY_REGION = re.compile(r"[01][01 \t\r\n\f\v]{62,}[01]")
_RE_HEX_REGION = re.compile(r"[0-9a-fA-F][0-9a-fA-F \t\r\n\f\v]{30,}[0-9a-fA-F]")
_RE_B64_TOKEN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")


def _strip_ws(s: str) -> str:
    return s.translate({ord(c): None for c in _SAFE_WHITESPACE})


def detect_encoded_stream(text: str, min_bytes: int = _DEFAULT_MIN_BYTES) -> List[EncodedDetection]:
    """Detect candidate encoded representations in ``text``.

    Confidence-ordered (best first). Detection is deliberately permissive because the
    reconstruction + magic-byte gate downstream is what actually decides whether to
    escalate — but each scheme still applies real structural checks so normal prose is
    not misclassified (requirement #2). Order matters: ASCII-binary is the most specific
    charset ({0,1}) and is checked first, so a pure bit-stream is never mistaken for hex
    or base64 (whose charsets are supersets of it).
    """
    out: List[EncodedDetection] = []
    if not text or len(text) < min_bytes * 8:
        # Even the densest encoding (binary, 8 chars/byte) needs this many chars.
        if not text:
            return out

    seen_tokens: set = set()

    # --- ASCII binary (CRITICAL path) --------------------------------------- #
    best_bin: Optional[str] = None
    for m in _RE_BINARY_REGION.finditer(text):
        cleaned = _strip_ws(m.group(0))
        if not cleaned or any(c not in "01" for c in cleaned):
            continue
        if len(cleaned) % 8 != 0:
            continue
        if len(cleaned) < min_bytes * 8:
            continue
        if best_bin is None or len(cleaned) > len(best_bin):
            best_bin = cleaned
    if best_bin:
        seen_tokens.add(best_bin)
        out.append(EncodedDetection(
            scheme="ascii_binary", confidence=0.96, token=best_bin[:400],
            full_token=best_bin, est_bytes=len(best_bin) // 8,
            reason=f"{len(best_bin)} bits ({len(best_bin)//8} bytes), charset {{0,1}}, length % 8 == 0"))

    # --- hex --------------------------------------------------------------- #
    best_hex: Optional[str] = None
    for m in _RE_HEX_REGION.finditer(text):
        cleaned = _strip_ws(m.group(0))
        if not cleaned or len(cleaned) % 2 != 0:
            continue
        if any(c not in "0123456789abcdefABCDEF" for c in cleaned):
            continue
        if len(cleaned) < min_bytes * 2:
            continue
        # Skip a region that is ALSO pure binary — that's the ascii_binary case.
        if all(c in "01" for c in cleaned):
            continue
        if cleaned in seen_tokens:
            continue
        if best_hex is None or len(cleaned) > len(best_hex):
            best_hex = cleaned
    if best_hex:
        seen_tokens.add(best_hex)
        # Lower confidence: long hex runs are often hashes/keys, not encoded files.
        conf = 0.55 if len(best_hex) < 128 else 0.65
        out.append(EncodedDetection(
            scheme="hex", confidence=conf, token=best_hex[:400], full_token=best_hex,
            est_bytes=len(best_hex) // 2,
            reason=f"{len(best_hex)} nybbles ({len(best_hex)//2} bytes), hex charset, even length"))

    # --- base64 ------------------------------------------------------------ #
    best_b64: Optional[str] = None
    for token in _RE_B64_TOKEN.findall(text):
        if len(token) % 4 != 0:
            continue
        if token in seen_tokens:
            continue
        # Skip pure-binary / pure-hex tokens already covered above.
        if all(c in "01" for c in token):
            continue
        try:
            dec = base64.b64decode(token, validate=True)
        except Exception:
            continue
        if len(dec) < min_bytes:
            continue
        if best_b64 is None or len(token) > len(best_b64):
            best_b64 = token
    if best_b64:
        try:
            est = len(base64.b64decode(best_b64, validate=True))
        except Exception:
            est = 0
        out.append(EncodedDetection(
            scheme="base64", confidence=0.5, token=best_b64[:400], full_token=best_b64,
            est_bytes=est, reason=f"valid base64, {est} decoded bytes"))

    out.sort(key=lambda d: (d.confidence, d.est_bytes), reverse=True)
    return out


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #

@dataclass
class ReconstructionOutcome:
    state: str
    scheme: str
    artifact_type: str                    # magic label (jpeg/png/.../generic_binary/text)
    recommended_tools: List[str] = field(default_factory=list)
    raw_bytes: bytes = b""
    byte_count: int = 0
    is_file: bool = False                 # genuine magic-byte hit for a non-text file type
    should_persist: bool = False          # persist + escalate as a derived FILE artifact
    decoded_text: str = ""                # printable rendering (for flag scanning), if any
    detection: Optional[EncodedDetection] = None
    origin_label: str = ""
    input_sha256: str = ""                # sha256 of the SOURCE text (caller-level dedup)
    output_sha256: str = ""               # sha256 of raw_bytes (artifact-level dedup)
    note: str = ""

    def to_evidence_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state, "scheme": self.scheme, "artifact_type": self.artifact_type,
            "byte_count": self.byte_count, "is_file": self.is_file,
            "recommended_tools": list(self.recommended_tools),
            "origin_label": self.origin_label, "output_sha256": self.output_sha256,
            "input_sha256": self.input_sha256, "note": self.note,
            "detection": self.detection.to_dict() if self.detection else None,
        }


def _bits_to_bytes(bits: str) -> Optional[bytes]:
    """Reconstruct bytes from an ASCII bit-string using 8-bit groups.

    NOT an integer conversion of the whole string — each consecutive group of 8
    characters is one byte (requirement #3). Rejects invalid input.
    """
    if not bits or len(bits) % 8 != 0:
        return None
    if any(c not in "01" for c in bits):
        return None
    out = bytearray(len(bits) // 8)
    for idx in range(0, len(bits), 8):
        out[idx // 8] = int(bits[idx:idx + 8], 2)
    return bytes(out)


def _reconstruct_bytes(det: EncodedDetection) -> Optional[bytes]:
    token = det.full_token or det.token
    try:
        if det.scheme == "ascii_binary":
            return _bits_to_bytes(token)
        if det.scheme == "hex":
            return bytes.fromhex(token)
        if det.scheme == "base64":
            return base64.b64decode(token, validate=True)
    except Exception:
        return None
    return None


def _printable_ratio(data: bytes) -> float:
    if not data:
        return 0.0
    printable = sum(1 for b in data if 0x20 <= b <= 0x7E or b in (0x09, 0x0A, 0x0D))
    return printable / len(data)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def reconstruct_from_text(text: str, origin_label: str = "",
                          min_bytes: int = _DEFAULT_MIN_BYTES) -> Optional[ReconstructionOutcome]:
    """Detect → reconstruct → magic-type the best encoded artifact in ``text``.

    Returns the single best :class:`ReconstructionOutcome`, or ``None`` when there is
    no useful encoded artifact. "Best" prefers, in order:

      3. a genuine reconstructed FILE (magic-byte hit)                → escalate
      2. an ASCII-binary stream that rebuilt to opaque binary bytes   → preserve (UNKNOWN)
      1. an encoded stream that rebuilt to readable TEXT              → flag-scan only
      0. anything that rebuilt to hex/base64 noise without a magic hit → dropped

    Only categories 3 and 2 set ``should_persist`` (a derived file is written and its
    analysis escalated). Category 1 is returned so the caller can scan ``decoded_text``
    for a flag, but no derived file is created (the existing text-decode path covers it).
    """
    if not text:
        return None
    detections = detect_encoded_stream(text, min_bytes=min_bytes)
    if not detections:
        return None

    input_hash = _sha256_text(text)
    best: Optional[ReconstructionOutcome] = None
    best_score = -1

    for det in detections:
        raw = _reconstruct_bytes(det)
        if raw is None or len(raw) < min_bytes:
            continue
        label, tools = _classify_magic(raw)
        is_file = label not in _TEXTUAL_LABELS and label != "generic_binary"
        ratio = _printable_ratio(raw)

        if is_file:
            score = 3
            state = ReconState.ARTIFACT_TYPE_IDENTIFIED
            should_persist = True
        elif det.scheme == "ascii_binary" and label == "generic_binary":
            # A deliberate wall of bits that rebuilt to opaque bytes is a real artifact
            # even when its type is unknown — preserve + escalate as UNKNOWN_ARTIFACT.
            score = 2
            state = ReconState.ARTIFACT_RECONSTRUCTED
            should_persist = True
        elif ratio > 0.85:
            # Rebuilt to readable text — useful for flag scanning, not a derived file.
            score = 1
            state = ReconState.ARTIFACT_RECONSTRUCTED
            should_persist = False
        else:
            # hex/base64 noise with no magic hit — almost certainly a hash/key/token.
            continue

        if score > best_score:
            decoded_text = ""
            if ratio > 0.6:
                decoded_text = raw.decode("utf-8", "replace")
            best = ReconstructionOutcome(
                state=state, scheme=det.scheme, artifact_type=label,
                recommended_tools=list(tools), raw_bytes=raw, byte_count=len(raw),
                is_file=is_file, should_persist=should_persist, decoded_text=decoded_text,
                detection=det, origin_label=origin_label, input_sha256=input_hash,
                output_sha256=hashlib.sha256(raw).hexdigest(),
                note=f"{det.scheme} → {label} ({len(raw)} bytes)")
            best_score = score
            if best_score == 3:
                break  # can't do better than a confirmed file type

    return best


# --------------------------------------------------------------------------- #
# Persistence with provenance (requirement #5)
# --------------------------------------------------------------------------- #

def persist_derived_artifact(outcome: ReconstructionOutcome, working_directory: str,
                             subdir: str = "forge_derived") -> Optional[str]:
    """Write the reconstructed bytes into the FORGE workspace with a provenance sidecar.

    Returns the absolute path of the derived file, or ``None`` if there is nothing to
    persist. Bytes are written byte-for-byte (never through a text layer). Stored under
    ``<working_directory>/<subdir>/`` — inside the challenge workspace, NOT /tmp.
    """
    if not outcome or not outcome.raw_bytes:
        return None
    if not working_directory:
        return None
    derived_dir = os.path.join(working_directory, subdir)
    ext = _EXT_FOR_TYPE.get(outcome.artifact_type, ".bin")
    short = (outcome.output_sha256 or hashlib.sha256(outcome.raw_bytes).hexdigest())[:12]
    name = f"derived_{short}{ext}"
    try:
        path = save_artifact_binary(outcome.raw_bytes, derived_dir, name)
    except Exception as exc:
        logger.warning("[artifact_reconstruction] persist failed: %s", exc)
        return None

    provenance = {
        "derived_file": path,
        "created_by": "forge.artifact_reconstruction",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scheme": outcome.scheme,
        "artifact_type": outcome.artifact_type,
        "byte_count": outcome.byte_count,
        "is_file": outcome.is_file,
        "recommended_tools": list(outcome.recommended_tools),
        "origin": outcome.origin_label,
        "input_sha256": outcome.input_sha256,
        "output_sha256": outcome.output_sha256,
        "detection": outcome.detection.to_dict() if outcome.detection else None,
        "security_note": "Reconstructed from untrusted challenge data — ANALYZE, DO NOT EXECUTE.",
    }
    try:
        with open(path + ".provenance.json", "w", encoding="utf-8") as fh:
            json.dump(provenance, fh, indent=2)
    except Exception as exc:
        logger.debug("[artifact_reconstruction] provenance sidecar failed: %s", exc)
    logger.info("[artifact_reconstruction] derived artifact persisted: %s (%s, %d bytes)",
                path, outcome.artifact_type, outcome.byte_count)
    return path


# --------------------------------------------------------------------------- #
# Visual artifact handling (requirement #7) — tesseract-free metadata + optional OCR
# --------------------------------------------------------------------------- #

_IMAGE_TYPES = {"jpeg", "png", "gif", "riff"}


def is_image_type(artifact_type: str) -> bool:
    return artifact_type in _IMAGE_TYPES


def inspect_image(path: str) -> Dict[str, Any]:
    """Return image metadata via PIL if available; degrade gracefully otherwise.

    Never requires tesseract. Verifies the image is well-formed and reports
    format/mode/dimensions so a vision-capable agent can decide how to read it.
    """
    try:
        from PIL import Image  # optional dependency
    except Exception:
        return {"available": False, "reason": "PIL (Pillow) not installed"}
    if not path or not os.path.exists(path):
        return {"available": False, "reason": "file not found"}
    try:
        with Image.open(path) as im:
            im.verify()  # structural integrity, does not decode pixels
        with Image.open(path) as im2:
            width, height = im2.size
            info = {"available": True, "valid": True, "format": im2.format,
                    "mode": im2.mode, "width": width, "height": height}
        return info
    except Exception as exc:
        return {"available": True, "valid": False, "reason": f"{type(exc).__name__}: {exc}"}


def attempt_ocr(path: str) -> Dict[str, Any]:
    """Best-effort OCR. Returns text when possible, otherwise a graceful 'unavailable'.

    OCR is NOT mandatory (requirement #7): a missing pytesseract binding or missing
    tesseract binary yields ``{"available": False, ...}`` rather than an error, so the
    flag rendered in an image is simply left for a vision-capable agent to read.
    """
    try:
        import pytesseract  # optional
        from PIL import Image  # optional
    except Exception as exc:
        return {"available": False, "reason": f"OCR unavailable ({type(exc).__name__})"}
    if not path or not os.path.exists(path):
        return {"available": False, "reason": "file not found"}
    try:
        text = pytesseract.image_to_string(Image.open(path))
        return {"available": True, "text": text or ""}
    except Exception as exc:
        # TesseractNotFoundError et al. — the binary isn't installed. Not fatal.
        return {"available": False, "reason": f"OCR unavailable ({type(exc).__name__})"}


# --------------------------------------------------------------------------- #
# Terminal ASCII-art / bitmap handling (requirement #8)
# --------------------------------------------------------------------------- #

_BITMAP_CHARS = set("#.*@ 01xX ")


def detect_ascii_bitmap(text: str) -> Dict[str, Any]:
    """Detect a monochrome ASCII-art / bitmap grid (a flag drawn in '#' and '.').

    Heuristic: >= 4 lines drawn from a tiny glyph alphabet, of roughly equal width,
    with at least two distinct glyph characters actually used (so a plain paragraph or
    a block of identical padding is not misread as art).
    """
    if not text:
        return {"is_bitmap": False}
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    grid = [ln for ln in lines if ln and all(c in _BITMAP_CHARS for c in ln)]
    if len(grid) < 4:
        return {"is_bitmap": False}
    widths = [len(ln) for ln in grid]
    min_w, max_w = min(widths), max(widths)
    if max_w < 4 or min_w == 0:
        return {"is_bitmap": False}
    if (max_w - min_w) > max(4, int(0.25 * max_w)):
        return {"is_bitmap": False}  # not rectangular enough
    used = set("".join(grid)) - {" "}
    if len(used) < 2 and " " not in "".join(grid):
        return {"is_bitmap": False}
    if len(used) < 1:
        return {"is_bitmap": False}
    return {"is_bitmap": True, "rows": len(grid), "cols": max_w,
            "charset": "".join(sorted(used)), "grid": grid}

