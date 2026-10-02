"""
Target specification, detection, and mismatch analysis (Phase 4.x §9–§11).

Before Phase 4.x, FORGE treated every ``target`` as an opaque string and let agents
discover the hard way that a supplied artifact was not the live service the
challenge actually required — burning dozens of iterations.  This module gives the
intelligence layer a *structured, conservative* model of what kind of thing a target
is, so a mismatch (e.g. the challenge needs ``nc host port`` but only a static
``source.py`` was provided) is caught as structured evidence instead of by trial and
error.

Detection is deliberately conservative: when the type cannot be determined with
reasonable confidence it is reported as :attr:`TargetType.UNKNOWN` rather than
guessed, so the swarm never invents a target that isn't there.

Nothing here executes anything — it only classifies strings/paths.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from urllib.parse import urlparse


class TargetType(str, Enum):
    STATIC_FILE = "STATIC_FILE"             # a downloadable/source file (source.py, .txt)
    LOCAL_ARTIFACT = "LOCAL_ARTIFACT"       # a local evidence file (.pcap, .zip, image)
    LOCAL_EXECUTABLE = "LOCAL_EXECUTABLE"   # a local binary to run/analyze
    LOCAL_SCRIPT = "LOCAL_SCRIPT"           # a local script (.py/.sh) to run/analyze
    LIVE_HTTP = "LIVE_HTTP"                 # an http(s) service
    LIVE_TCP = "LIVE_TCP"                   # a raw tcp service (nc host port)
    INTERACTIVE_PROCESS = "INTERACTIVE_PROCESS"  # a process that speaks a stdin/stdout dialogue
    REMOTE_SERVICE = "REMOTE_SERVICE"       # a networked host, protocol not yet known
    UNKNOWN = "UNKNOWN"


# Family groupings used for mismatch reasoning.
_LIVE_FAMILY = {TargetType.LIVE_HTTP, TargetType.LIVE_TCP, TargetType.REMOTE_SERVICE}
_STATIC_FAMILY = {TargetType.STATIC_FILE, TargetType.LOCAL_ARTIFACT}
_LOCAL_RUNNABLE = {TargetType.LOCAL_EXECUTABLE, TargetType.LOCAL_SCRIPT, TargetType.INTERACTIVE_PROCESS}

# Extension → type maps (lower-case, no dot).
_SCRIPT_EXT = {"py", "sh", "pl", "rb", "js", "ps1", "php"}
_SOURCE_TEXT_EXT = {"txt", "md", "c", "h", "cpp", "cc", "hpp", "java", "go", "rs",
                    "json", "xml", "yaml", "yml", "csv", "log", "asm", "s"}
_ARTIFACT_EXT = {"pcap", "pcapng", "cap", "zip", "tar", "gz", "tgz", "7z", "rar",
                 "jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff", "tif", "svg",
                 "pdf", "wav", "mp3", "mp4", "avi", "img", "dd", "raw", "vmem",
                 "docx", "xlsx", "pptx", "sqlite", "db", "hex"}
_EXECUTABLE_EXT = {"bin", "elf", "exe", "out", "so", "dll", "o", "ko", "axf"}
# Extensions that, over HTTP, are dynamic endpoints rather than static downloads.
_DYNAMIC_WEB_EXT = {"php", "asp", "aspx", "jsp", "cgi", "do", "action", "html", "htm"}

_HOSTPORT_RE = re.compile(r"^(?P<host>[A-Za-z0-9_.\-]+):(?P<port>\d{1,5})$")
_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_NC_RE = re.compile(r"^\s*(?:nc|ncat|netcat)\b(?P<rest>.*)$", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"^[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$")


@dataclass
class Target:
    """A structured, conservatively-typed view of one target string."""
    raw: str
    type: TargetType = TargetType.UNKNOWN
    host: str = ""
    port: Optional[int] = None
    path: str = ""
    url: str = ""
    scheme: str = ""
    confidence: float = 0.0
    reason: str = ""

    @property
    def is_live(self) -> bool:
        return self.type in _LIVE_FAMILY or self.type is TargetType.INTERACTIVE_PROCESS

    @property
    def is_local(self) -> bool:
        return self.type in (_STATIC_FAMILY | _LOCAL_RUNNABLE) and self.type is not TargetType.INTERACTIVE_PROCESS

    def to_dict(self) -> dict:
        return {
            "raw": self.raw, "type": self.type.value, "host": self.host, "port": self.port,
            "path": self.path, "url": self.url, "scheme": self.scheme,
            "confidence": round(self.confidence, 3), "reason": self.reason,
            "is_live": self.is_live, "is_local": self.is_local,
        }


@dataclass
class TargetMismatch:
    """Structured evidence that the provided target is the wrong KIND of thing (§11)."""
    expected: TargetType
    observed: TargetType
    target: str
    reason: str = ""
    recommended_action: str = "replan"     # replan | request_live_target | acquire_artifact

    def to_dict(self) -> dict:
        return {
            "kind": "TARGET_MISMATCH", "expected": self.expected.value,
            "observed": self.observed.value, "target": self.target,
            "reason": self.reason, "recommended_action": self.recommended_action,
        }

    def summary(self) -> str:
        return (f"TARGET_MISMATCH expected={self.expected.value} "
                f"observed={self.observed.value} action={self.recommended_action}")


def _ext_of(path: str) -> str:
    base = os.path.basename(path.split("?")[0].split("#")[0])
    if "." not in base:
        return ""
    return base.rsplit(".", 1)[1].lower()


def _classify_by_extension(ext: str, *, local: bool) -> Optional[TargetType]:
    if not ext:
        return None
    if ext in _EXECUTABLE_EXT:
        return TargetType.LOCAL_EXECUTABLE
    if ext in _SCRIPT_EXT:
        return TargetType.LOCAL_SCRIPT
    if ext in _ARTIFACT_EXT:
        return TargetType.LOCAL_ARTIFACT
    if ext in _SOURCE_TEXT_EXT:
        return TargetType.STATIC_FILE
    return None


class TargetDetector:
    """Conservative target-type inference. Never guesses when genuinely ambiguous."""

    def detect(self, raw: str) -> Target:
        s = (raw or "").strip()
        if not s:
            return Target(raw=raw or "", type=TargetType.UNKNOWN, confidence=0.0,
                          reason="empty target")

        # 1) nc/ncat command → raw TCP dialogue (interactive service).
        m = _NC_RE.match(s)
        if m:
            tokens = [t for t in m.group("rest").split() if not t.startswith("-")]
            host = tokens[0] if tokens else ""
            port = None
            if len(tokens) >= 2 and tokens[1].isdigit():
                port = int(tokens[1])
            return Target(raw=s, type=TargetType.LIVE_TCP, host=host, port=port,
                          confidence=0.95, reason="netcat command implies a raw TCP service")

        # 2) Explicit URL scheme.
        if s.lower().startswith(("http://", "https://")):
            u = urlparse(s)
            ext = _ext_of(u.path)
            static = _classify_by_extension(ext, local=False)
            host_hint = any(h in (u.hostname or "") for h in
                            ("challenge-files", "artifacts", "files.", "-files", "cdn", "download"))
            if static in _STATIC_FAMILY or (static and ext not in _DYNAMIC_WEB_EXT):
                return Target(raw=s, type=TargetType.STATIC_FILE, host=u.hostname or "",
                              port=u.port, path=u.path, url=s, scheme=u.scheme,
                              confidence=0.85 if host_hint else 0.75,
                              reason=f"HTTP URL to a downloadable .{ext} file")
            return Target(raw=s, type=TargetType.LIVE_HTTP, host=u.hostname or "",
                          port=u.port, path=u.path, url=s, scheme=u.scheme,
                          confidence=0.9, reason="http(s) service URL")

        # 3) host:port with no scheme → raw TCP service.
        hp = _HOSTPORT_RE.match(s)
        if hp:
            port = int(hp.group("port"))
            if 0 < port <= 65535:
                return Target(raw=s, type=TargetType.LIVE_TCP, host=hp.group("host"),
                              port=port, confidence=0.8,
                              reason="host:port with no scheme implies a raw TCP service")

        # 4) A run-a-local-script command (e.g. "python challenge.py").
        parts = s.split()
        if parts and parts[0] in ("python", "python3", "py", "./", "sh", "bash") or s.startswith("./"):
            script = next((p for p in parts if _ext_of(p) in _SCRIPT_EXT), "")
            if script:
                return Target(raw=s, type=TargetType.LOCAL_SCRIPT, path=script, confidence=0.7,
                              reason="command runs a local script")

        # 5) Filesystem path — existing or path-shaped.
        looks_like_path = ("/" in s or "\\" in s or s.startswith(".") or _ext_of(s) != ""
                           or (len(s) > 2 and s[1] == ":"))
        if looks_like_path:
            ext = _ext_of(s)
            if os.path.exists(s):
                if os.path.isdir(s):
                    return Target(raw=s, type=TargetType.LOCAL_ARTIFACT, path=s, confidence=0.7,
                                  reason="existing local directory")
                by_ext = _classify_by_extension(ext, local=True)
                if by_ext:
                    return Target(raw=s, type=by_ext, path=s, confidence=0.9,
                                  reason=f"existing local .{ext} file")
                # Executable bit (POSIX) with no informative extension.
                try:
                    if os.access(s, os.X_OK) and not ext:
                        return Target(raw=s, type=TargetType.LOCAL_EXECUTABLE, path=s,
                                      confidence=0.75, reason="existing executable file")
                except Exception:
                    pass
                return Target(raw=s, type=TargetType.LOCAL_ARTIFACT, path=s, confidence=0.7,
                              reason="existing local file")
            # Path-shaped but not present: infer by extension at lower confidence.
            by_ext = _classify_by_extension(ext, local=True)
            if by_ext:
                return Target(raw=s, type=by_ext, path=s, confidence=0.55,
                              reason=f"path-shaped .{ext} (not present locally)")
            # A bare path we can't type.
            if "/" in s or "\\" in s:
                return Target(raw=s, type=TargetType.UNKNOWN, path=s, confidence=0.3,
                              reason="path-shaped but type undetermined")

        # 6) Bare IP or domain with no port/scheme → networked host, protocol unknown.
        if _IP_RE.match(s) or _DOMAIN_RE.match(s):
            return Target(raw=s, type=TargetType.REMOTE_SERVICE, host=s, confidence=0.55,
                          reason="bare host/IP — networked target, protocol not specified")

        return Target(raw=s, type=TargetType.UNKNOWN, confidence=0.2,
                      reason="could not confidently determine target type")

    def detect_multi(self, raw: str) -> List[Target]:
        """Split a multi-target spec on the FORGE ``+`` delimiter and type each part."""
        parts = [p.strip() for p in (raw or "").split("+") if p.strip()]
        return [self.detect(p) for p in parts] or [self.detect(raw)]

    # ------------------------------------------------------------------ #

    def classify_mismatch(self, required: TargetType, provided: Target) -> Optional[TargetMismatch]:
        """Return a :class:`TargetMismatch` when *provided* is the wrong kind for *required*.

        Conservative: only cross-family mismatches (live↔static) are flagged. When
        either side is UNKNOWN, or both are live, no mismatch is asserted.
        """
        if required is TargetType.UNKNOWN or provided.type is TargetType.UNKNOWN:
            return None
        req_live = required in _LIVE_FAMILY or required is TargetType.INTERACTIVE_PROCESS
        prov_static = provided.type in _STATIC_FAMILY

        if req_live and prov_static:
            return TargetMismatch(
                expected=required, observed=provided.type, target=provided.raw,
                reason=(f"Challenge requires a live target ({required.value}) but a "
                        f"static file ({provided.type.value}) was supplied."),
                recommended_action="request_live_target")

        req_static = required in (_STATIC_FAMILY | _LOCAL_RUNNABLE)
        prov_live = provided.type in _LIVE_FAMILY
        if req_static and prov_live:
            return TargetMismatch(
                expected=required, observed=provided.type, target=provided.raw,
                reason=(f"Task expects a local artifact/executable ({required.value}) but a "
                        f"live service ({provided.type.value}) was supplied."),
                recommended_action="acquire_artifact")
        return None


# Module-level singleton + convenience wrappers.
target_detector = TargetDetector()


def detect_target(raw: str) -> Target:
    return target_detector.detect(raw)


def detect_targets(raw: str) -> List[Target]:
    return target_detector.detect_multi(raw)
