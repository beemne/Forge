"""
Deterministic binary-artifact pre-classifier for FORGE.

Runs BEFORE any LLM call or task-pool selection.  Checks HTTP response metadata
and/or a local file to decide whether the target is a downloadable binary artifact
that must be handled by the binary-safe workflow rather than the general web flow.

Design principles
-----------------
- Zero LLM calls.  Every decision is a cheap string / byte check.
- Defense-in-depth: category tagging alone is unreliable (a REV challenge may
  still be served from a web URL; a WEB challenge may include a downloadable
  binary that unlocks further steps).  The classifier fires independently of
  category metadata so that even mislabelled or multi-step challenges are handled
  correctly.
- Raw bytes are NEVER decoded as text.  The classifier only reads the first few
  bytes (magic bytes), then immediately releases the handle.
- If the result is BINARY, callers must save the artifact via
  save_artifact_binary() before any agent reasoning begins.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("forge.artifact_classifier")

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    is_binary: bool
    reason: str                             # human-readable, logged to challenge log
    artifact_type: str = "unknown"          # "elf", "zip", "pe", "pdf", "generic_binary", "text"
    recommended_tools: list = field(default_factory=list)
    safe_file_path: Optional[str] = None    # set after save_artifact_binary() is called


# ---------------------------------------------------------------------------
# Static-storage / CDN server signatures
# ---------------------------------------------------------------------------

_STATIC_SERVER_SIGNATURES = frozenset([
    "amazons3", "s3.amazonaws", "cloudfront",
    "googleusercontent", "storage.googleapis",
    "azureblob", "blob.core.windows",
    "fastly", "cdn", "static", "assets", "files",
    "r2.cloudflarestorage",
])

# Content-Type prefixes that indicate a non-text, non-HTML artifact
_BINARY_CONTENT_TYPES = (
    "application/octet-stream",
    "application/zip",
    "application/x-zip",
    "application/x-tar",
    "application/gzip",
    "application/x-gzip",
    "application/x-bzip2",
    "application/x-xz",
    "application/x-elf",
    "application/x-executable",
    "application/x-sharedlib",
    "application/x-dosexec",          # PE / EXE
    "application/x-msdownload",
    "application/pdf",
    "application/java-archive",
    "application/x-java-class",
    "application/vnd.android.package-archive",  # APK
    "application/x-7z-compressed",
    "application/x-rar",
    "image/",                          # prefix match — images are binary
    "audio/",
    "video/",
    "font/",
)

# Hostname / URL path fragments that indicate artifact hosting
_ARTIFACT_HOST_PATTERNS = (
    "files.", "/files/",
    "cdn.", "/cdn/",
    "static.", "/static/",
    "assets.", "/assets/",
    "download", "/download",
    "storage.",
    ".s3.", ".s3-",
    "artifact", "release",
    "challenge-files",
)

# Known binary file extensions
_BINARY_EXTENSIONS = frozenset([
    ".bin", ".elf", ".exe", ".dll", ".so", ".o", ".ko",
    ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z", ".rar",
    ".apk", ".jar", ".class",
    ".pdf", ".iso", ".img",
    ".pyc", ".pyo",
    ".pcap", ".pcapng",
    ".db", ".sqlite", ".sqlite3",
    ".dump", ".dmp",
    ".rom", ".fw", ".hex",
    ".img", ".raw",
])

# Magic byte signatures: (offset, bytes, label, recommended_tools)
_MAGIC_SIGNATURES: list[tuple[int, bytes, str, list[str]]] = [
    (0, b"\x7fELF",              "elf",            ["file", "checksec", "strings", "objdump", "radare2", "ghidra"]),
    (0, b"MZ",                   "pe",             ["file", "strings", "pe-sieve", "radare2"]),
    (0, b"\x50\x4b\x03\x04",    "zip",            ["unzip", "file", "binwalk"]),
    (0, b"\x1f\x8b",            "gzip",           ["file", "gunzip", "binwalk"]),
    (0, b"BZh",                  "bzip2",          ["file", "bunzip2", "binwalk"]),
    (0, b"\xfd7zXZ\x00",        "xz",             ["file", "unxz", "binwalk"]),
    (0, b"Rar!",                 "rar",            ["file", "unrar", "binwalk"]),
    (0, b"7z\xbc\xaf\x27\x1c",  "7zip",           ["file", "7z", "binwalk"]),
    (0, b"\xca\xfe\xba\xbe",    "java_class",     ["file", "javap", "cfr"]),
    (0, b"\xce\xfa\xed\xfe",    "macho_32le",     ["file", "otool", "radare2"]),
    (0, b"\xcf\xfa\xed\xfe",    "macho_64le",     ["file", "otool", "radare2"]),
    (0, b"%PDF",                 "pdf",            ["file", "pdfparser", "pdfextract"]),
    (0, b"\x89PNG\r\n\x1a\n",   "png",            ["file", "binwalk", "zsteg"]),
    (0, b"\xff\xd8\xff",        "jpeg",           ["file", "exiftool", "binwalk"]),
    (0, b"GIF8",                 "gif",            ["file", "exiftool", "binwalk"]),
    (0, b"RIFF",                 "riff",           ["file", "strings"]),
    (0, b"\xd0\xcf\x11\xe0",    "ms_ole",         ["file", "oletools", "strings"]),
    (0, b"SQLite format 3\x00", "sqlite",          ["file", "sqlite3"]),
    (0, b"\x00asm",              "wasm",           ["file", "wasm-decompile"]),
    (0, b"PCAP",                 "pcap",           ["tshark", "wireshark", "tcpdump"]),
    (0, b"\xd4\xc3\xb2\xa1",    "pcap_le",        ["tshark", "wireshark"]),
    (0, b"\xa1\xb2\xc3\xd4",    "pcap_be",        ["tshark", "wireshark"]),
    (0, b"\x0a\x0d\x0d\x0a",    "pcapng",         ["tshark", "wireshark"]),
]


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------

def classify_http_response(
    url: str,
    content_type: str,
    server_header: str,
    content_length: Optional[int],
    response_body_prefix: bytes,   # first 512 bytes of body, may be empty
) -> ClassificationResult:
    """Classify an HTTP response as binary artifact or web content.

    Parameters
    ----------
    url:
        The full URL that was fetched.
    content_type:
        Value of the Content-Type response header (lowercased by caller).
    server_header:
        Value of the Server response header (lowercased by caller).
    content_length:
        Value of Content-Length header, or None if absent.
    response_body_prefix:
        First up to 512 bytes of the response body, as raw bytes.
        Pass b"" if the body was not pre-read.
    """
    url_lower = url.lower()

    # ── Check 1: Content-Type ───────────────────────────────────────────────
    ct = (content_type or "").lower().split(";")[0].strip()
    for binary_ct in _BINARY_CONTENT_TYPES:
        if ct.startswith(binary_ct):
            label, tools = _classify_magic(response_body_prefix)
            if label == "text":
                return ClassificationResult(
                    is_binary=False,
                    reason=f"Content-Type '{ct}' claimed binary but response body is plain text",
                    artifact_type="text",
                )
            return ClassificationResult(
                is_binary=True,
                reason=f"Content-Type '{ct}' indicates binary artifact",
                artifact_type=label,
                recommended_tools=tools,
            )

    # ── Check 2: Server header / static-storage signature ──────────────────
    srv = (server_header or "").lower()
    for sig in _STATIC_SERVER_SIGNATURES:
        if sig in srv:
            label, tools = _classify_magic(response_body_prefix)
            return ClassificationResult(
                is_binary=True,
                reason=f"Server header '{server_header}' matches static-storage signature '{sig}'",
                artifact_type=label,
                recommended_tools=tools,
            )

    # ── Check 3: Hostname / URL path artifact patterns ──────────────────────
    for pattern in _ARTIFACT_HOST_PATTERNS:
        if pattern in url_lower:
            label, tools = _classify_magic(response_body_prefix)
            return ClassificationResult(
                is_binary=True,
                reason=f"URL '{url}' matches artifact-hosting pattern '{pattern}'",
                artifact_type=label,
                recommended_tools=tools,
            )

    # ── Check 4: URL file extension ─────────────────────────────────────────
    path_part = url_lower.split("?")[0].split("#")[0]
    _, ext = os.path.splitext(path_part)
    if ext in _BINARY_EXTENSIONS:
        label, tools = _classify_magic(response_body_prefix)
        return ClassificationResult(
            is_binary=True,
            reason=f"URL file extension '{ext}' indicates binary artifact",
            artifact_type=label,
            recommended_tools=tools,
        )

    # ── Check 5: Magic bytes of response body prefix ────────────────────────
    if response_body_prefix:
        label, tools = _classify_magic(response_body_prefix)
        if label != "text":
            return ClassificationResult(
                is_binary=True,
                reason=f"Response body magic bytes indicate '{label}' binary format",
                artifact_type=label,
                recommended_tools=tools,
            )

    # ── Check 6: Very small non-HTML body (likely raw binary blob) ──────────
    if content_length is not None and content_length > 0:
        body_text = response_body_prefix.decode("utf-8", errors="replace") if response_body_prefix else ""
        if (
            content_length < 10_000
            and ct not in ("text/html", "text/plain", "application/json",
                           "application/javascript", "text/css")
            and not body_text.lstrip().startswith(("<", "{", "["))
        ):
            return ClassificationResult(
                is_binary=True,
                reason=(
                    f"Small payload ({content_length} bytes) with non-text Content-Type "
                    f"'{ct}' — treating as binary blob"
                ),
                artifact_type="generic_binary",
                recommended_tools=["file", "xxd", "strings", "binwalk"],
            )

    return ClassificationResult(
        is_binary=False,
        reason="No binary indicators found — treating as web/text target",
        artifact_type="text",
    )


def classify_local_file(file_path: str) -> ClassificationResult:
    """Classify an already-present local file (uploaded or pre-downloaded).

    Reads only the first 32 bytes for magic detection; never decodes as text.
    """
    if not file_path or not os.path.isfile(file_path):
        return ClassificationResult(
            is_binary=False,
            reason=f"File path '{file_path}' does not exist or is not a file",
            artifact_type="unknown",
        )

    # Extension check first (cheap)
    _, ext = os.path.splitext(file_path.lower())
    if ext in _BINARY_EXTENSIONS:
        try:
            with open(file_path, "rb") as fh:
                header = fh.read(32)
        except OSError as exc:
            return ClassificationResult(
                is_binary=False,
                reason=f"Could not read file for magic classification: {exc}",
                artifact_type="unknown",
            )
        label, tools = _classify_magic(header)
        if label == "text":
            return ClassificationResult(
                is_binary=False,
                reason=f"File extension '{ext}' claimed binary but file content is plain text",
                artifact_type="text",
                safe_file_path=file_path,
            )
        return ClassificationResult(
            is_binary=True,
            reason=f"File extension '{ext}' indicates binary artifact",
            artifact_type=label,
            recommended_tools=tools,
            safe_file_path=file_path,
        )

    # Magic bytes
    try:
        with open(file_path, "rb") as fh:
            header = fh.read(32)
    except OSError as exc:
        return ClassificationResult(
            is_binary=False,
            reason=f"Could not read file for magic classification: {exc}",
            artifact_type="unknown",
        )

    label, tools = _classify_magic(header)
    if label != "text":
        return ClassificationResult(
            is_binary=True,
            reason=f"File magic bytes indicate '{label}' binary format",
            artifact_type=label,
            recommended_tools=tools,
            safe_file_path=file_path,
        )

    # Heuristic: high proportion of non-printable bytes → binary
    non_print = sum(1 for b in header if b < 0x09 or (0x0E <= b <= 0x1F) or b == 0x7F or b > 0x7E)
    if header and (non_print / len(header)) > 0.20:
        return ClassificationResult(
            is_binary=True,
            reason=f"File header has {non_print}/{len(header)} non-printable bytes — treating as binary",
            artifact_type="generic_binary",
            recommended_tools=["file", "xxd", "strings", "binwalk"],
            safe_file_path=file_path,
        )

    return ClassificationResult(
        is_binary=False,
        reason="File appears to be text/source",
        artifact_type="text",
        safe_file_path=file_path,
    )


# ---------------------------------------------------------------------------
# Artifact save helper
# ---------------------------------------------------------------------------

def save_artifact_binary(
    raw_bytes: bytes,
    working_directory: str,
    suggested_name: str = "artifact.bin",
) -> str:
    """Write raw bytes to disk byte-for-byte without passing through any
    text-decoding layer.

    This is the ONLY safe way to persist a binary artifact in FORGE.
    Never use open(..., "w") or any UTF-8 decoding step on raw artifact bytes.

    Returns the absolute path of the saved file.
    """
    os.makedirs(working_directory, exist_ok=True)
    dest_path = os.path.join(working_directory, suggested_name)

    # If a file by this name already exists, append a counter to avoid clobbering.
    if os.path.exists(dest_path):
        base, ext = os.path.splitext(suggested_name)
        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(working_directory, f"{base}_{counter}{ext}")
            counter += 1

    with open(dest_path, "wb") as fh:
        fh.write(raw_bytes)

    logger.info(
        f"[ArtifactClassifier] Binary artifact saved byte-for-byte: "
        f"{dest_path} ({len(raw_bytes):,} bytes)"
    )
    return dest_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_magic(data: bytes) -> tuple[str, list[str]]:
    """Return (artifact_type_label, recommended_tools) from magic bytes.

    Falls back to ("generic_binary", [...]) if no signature matches but data
    looks non-textual, or ("text", []) for plain-text content.
    """
    if not data:
        return "unknown", ["file"]

    for offset, magic, label, tools in _MAGIC_SIGNATURES:
        end = offset + len(magic)
        if len(data) >= end and data[offset:end] == magic:
            return label, tools

    # Heuristic: if > 20 % of the first 32 bytes are non-printable, call it binary
    sample = data[:32]
    non_print = sum(1 for b in sample if b < 0x09 or (0x0E <= b <= 0x1F) or b == 0x7F or b > 0x7E)
    if sample and (non_print / len(sample)) > 0.20:
        return "generic_binary", ["file", "xxd", "strings", "binwalk"]

    return "text", []


# ---------------------------------------------------------------------------
# Module-level singleton (import-friendly)
# ---------------------------------------------------------------------------

artifact_classifier = type(
    "ArtifactClassifier", (), {
        "classify_http_response": staticmethod(classify_http_response),
        "classify_local_file":    staticmethod(classify_local_file),
        "save_artifact_binary":   staticmethod(save_artifact_binary),
    }
)()
