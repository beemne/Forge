"""FORGE Response Profiler & Empirical Anomaly Engine.

Tracks baseline characteristics (status codes, body length distributions, structural
token hashes) of responses per endpoint/command target without hardcoded challenge strings.
Detects statistical and structural anomalies (ANOMALOUS_RESPONSE), extracts candidate
artifacts (paths, URLs, tokens) via generic structural patterns, and provides priority
preemption and post-exploitation probe generators.
"""

from __future__ import annotations

import hashlib
import math
import re
import urllib.parse
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple


# ── Generic structural extraction patterns (strictly pattern-based, no challenge keywords) ──
_URL_PATTERN = re.compile(r"https?://[^\s\"'<>)\]}]+", re.IGNORECASE)

# Standard file extensions commonly encountered in CTF / web / system interactions
_COMMON_EXTENSIONS = (
    r"php\d*|phtml|phar|inc|"
    r"py|pyc|sh|bash|pl|cgi|rb|js|ts|json|xml|yaml|yml|"
    r"png|jpg|jpeg|gif|bmp|webp|ico|svg|pdf|zip|tar|gz|7z|rar|bz2|"
    r"txt|log|html|htm|css|sql|db|sqlite|bak|old|tmp|conf|ini|env|swp|dist"
)

# Relative / absolute path pattern: e.g., uploads/abc.php, /var/www/uploads/test.png, ./files/out.txt, images/avatar.jpg
_PATH_PATTERN = re.compile(
    rf"""
    (?:\b|(?<=['"\s:=<>]))
    (
        (?:[a-zA-Z0-9_\-\.]+/)*
        [a-zA-Z0-9_\-\.]+
        \.(?:{_COMMON_EXTENSIONS})
    )
    (?:\b|(?=['"\s:=<>,;?#]|$))
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Root-relative or directory-structured path: e.g. /app/data, /api/v1/user/123, uploads/2026/
_DIR_PATH_PATTERN = re.compile(
    r"""
    (?:\b|(?<=['"\s:=<>]))
    (
        /(?:[a-zA-Z0-9_\-\.]+/)*[a-zA-Z0-9_\-\.]+/?
    )
    (?:\b|(?=['"\s:=<>,;?]|$))
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Generic token / identifier pattern: e.g. 16+ hex chars, 20+ base64/url-safe token chars, UUIDs
_UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_TOKEN_PATTERN = re.compile(
    r"\b(?=[a-zA-Z0-9_\-]{20,80}\b)(?:[0-9a-fA-F]{24,64}|[a-zA-Z0-9_\-]{20,80})\b"
)


def _tokenize_structure(text: str) -> str:
    """Compute a normalized structural signature of text.
    Replaces digits, alphanumeric words, and whitespace runs to retain tag/delimiter skeleton.
    """
    if not text:
        return ""
    # Strip numbers
    skeleton = re.sub(r"\d+", "0", text)
    # Strip long words to general tokens
    skeleton = re.sub(r"[a-zA-Z]{4,}", "word", skeleton)
    # Normalize whitespace
    skeleton = re.sub(r"\s+", " ", skeleton).strip()
    return hashlib.md5(skeleton.encode("utf-8", "replace")).hexdigest()[:12]


@dataclass
class CandidateArtifact:
    """A structurally extracted artifact candidate (URL, file path, or token) from a response."""
    raw_value: str
    artifact_type: str                  # "URL", "PATH", "TOKEN", "ENDPOINT"
    normalized_target: str = ""         # fully qualified URL or clean path
    source_command: str = ""
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyResult:
    """Evaluation result comparing a response against an endpoint baseline."""
    is_anomalous: bool
    score: float                        # 0.0 to 1.0 divergence score
    reasons: List[str] = field(default_factory=list)
    candidate_artifacts: List[CandidateArtifact] = field(default_factory=list)
    baseline_samples: int = 0
    response_length: int = 0
    baseline_mean_length: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["candidate_artifacts"] = [c.to_dict() for c in self.candidate_artifacts]
        return d


@dataclass
class EndpointBaseline:
    """Statistical & structural profile of responses observed for an endpoint or command target."""
    target_key: str
    sample_count: int = 0
    lengths: List[int] = field(default_factory=list)
    status_codes: Dict[int, int] = field(default_factory=dict)
    structural_hashes: Dict[str, int] = field(default_factory=dict)
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0

    @property
    def mean_length(self) -> float:
        if not self.lengths:
            return 0.0
        return sum(self.lengths) / len(self.lengths)

    @property
    def stddev_length(self) -> float:
        if len(self.lengths) < 2:
            return 0.0
        mean = self.mean_length
        variance = sum((x - mean) ** 2 for x in self.lengths) / (len(self.lengths) - 1)
        return math.sqrt(variance)

    @property
    def dominant_status(self) -> Optional[int]:
        if not self.status_codes:
            return None
        return max(self.status_codes.items(), key=lambda kv: kv[1])[0]

    @property
    def dominant_structural_hash(self) -> Optional[str]:
        if not self.structural_hashes:
            return None
        return max(self.structural_hashes.items(), key=lambda kv: kv[1])[0]

    def record(self, length: int, status_code: Optional[int] = None, struct_hash: str = ""):
        self.sample_count += 1
        # Keep last 50 lengths to be memory and calculation efficient
        self.lengths.append(length)
        if len(self.lengths) > 50:
            self.lengths.pop(0)

        if status_code is not None:
            self.status_codes[status_code] = self.status_codes.get(status_code, 0) + 1

        if struct_hash:
            self.structural_hashes[struct_hash] = self.structural_hashes.get(struct_hash, 0) + 1

    def evaluate_divergence(
        self,
        length: int,
        status_code: Optional[int] = None,
        struct_hash: str = "",
        min_samples: int = 2,
    ) -> Tuple[bool, float, List[str]]:
        """Evaluate how much a new response diverges from this baseline.
        Returns (is_anomalous, divergence_score, reasons).
        """
        if self.sample_count < min_samples:
            # Not enough samples to establish a confident baseline yet
            return False, 0.0, []

        reasons = []
        score = 0.0

        # 1. Status code divergence
        if status_code is not None and self.status_codes:
            dominant = self.dominant_status
            if dominant is not None and status_code != dominant:
                # Different status code from dominant baseline
                score += 0.4
                reasons.append(f"Status code changed from baseline {dominant} to {status_code}")

        # 2. Length divergence (relative delta & standard deviations)
        mean_len = self.mean_length
        std_len = self.stddev_length
        abs_diff = abs(length - mean_len)

        if mean_len > 0:
            relative_diff = abs_diff / max(mean_len, 1.0)
            if relative_diff > 0.25 and abs_diff >= 8:
                # More than 25% difference in length and at least 8 bytes delta
                score += min(0.45, max(0.3, relative_diff * 0.4))
                reasons.append(
                    f"Response length {length} bytes diverges from baseline mean {mean_len:.1f} bytes "
                    f"(delta: {length - mean_len:+.1f} bytes, {relative_diff * 100:.1f}%)"
                )
            elif std_len > 0.5 and abs_diff > (2.5 * std_len) and abs_diff >= 8:
                # Statistical outlier (> 2.5 std devs)
                score += 0.35
                reasons.append(
                    f"Response length {length} is {abs_diff / std_len:.1f} standard deviations from baseline mean"
                )
        elif length > 0 and abs_diff >= 8:
            score += 0.4
            reasons.append(f"Response length {length} bytes received on previously empty baseline")

        # 3. Structural skeleton divergence
        if struct_hash and self.structural_hashes:
            dominant_hash = self.dominant_structural_hash
            if dominant_hash and struct_hash != dominant_hash:
                score += 0.35
                reasons.append(f"Structural skeleton hash {struct_hash} differs from baseline {dominant_hash}")

        # Normalize score to max 1.0
        score = min(1.0, score)
        is_anomalous = score >= 0.35
        return is_anomalous, score, reasons

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EndpointBaseline":
        return cls(
            target_key=d.get("target_key", ""),
            sample_count=int(d.get("sample_count", 0)),
            lengths=list(d.get("lengths", [])),
            status_codes={int(k): int(v) for k, v in (d.get("status_codes") or {}).items()},
            structural_hashes=dict(d.get("structural_hashes") or {}),
            first_seen_ts=float(d.get("first_seen_ts", 0.0)),
            last_seen_ts=float(d.get("last_seen_ts", 0.0)),
        )


def extract_generic_artifacts(text: str, base_url: str = "") -> List[CandidateArtifact]:
    """Lightweight, purely structural extraction of URLs, relative file paths,
    and unique identifier tokens from response text without any hardcoded keywords.
    """
    if not text or not isinstance(text, str):
        return []

    candidates: List[CandidateArtifact] = []
    seen_values: Set[str] = set()

    # Helper to clean extracted paths
    def _clean_path(raw: str) -> str:
        p = raw.strip().strip("'\"`<>(),;:[]{}")
        # remove trailing punctuation if attached
        p = re.sub(r"[.,;:)]+$", "", p)
        return p

    # 1. Full URLs
    for m in _URL_PATTERN.findall(text):
        clean_url = _clean_path(m)
        if clean_url and clean_url not in seen_values and len(clean_url) > 7:
            seen_values.add(clean_url)
            seen_values.add(clean_url.rstrip("/"))
            candidates.append(CandidateArtifact(
                raw_value=clean_url,
                artifact_type="URL",
                normalized_target=clean_url,
                confidence=0.9,
                metadata={"scheme": urllib.parse.urlparse(clean_url).scheme}
            ))

    # 2. File paths with extensions (e.g. uploads/shell.php, /images/pic.png, file.php)
    for m in _PATH_PATTERN.findall(text):
        clean = _clean_path(m)
        if not clean or clean in seen_values or clean.strip("/") in seen_values or len(clean) < 3:
            continue
        # Filter obvious source-code syntax or programming artifacts
        if clean.endswith(".len") or clean.endswith(".get") or clean.endswith(".post"):
            continue
        if clean.startswith("http://") or clean.startswith("https://"):
            continue

        seen_values.add(clean)
        seen_values.add(clean.strip("/"))
        # Normalize target with base_url if available
        norm_target = clean
        if base_url:
            norm_target = urllib.parse.urljoin(base_url, clean.lstrip("/"))

        candidates.append(CandidateArtifact(
            raw_value=clean,
            artifact_type="PATH",
            normalized_target=norm_target,
            confidence=0.85,
            metadata={"extension": clean.rsplit(".", 1)[-1].lower()}
        ))

    # 3. Directory paths (e.g. /uploads/, /api/v1/item)
    for m in _DIR_PATH_PATTERN.findall(text):
        clean = _clean_path(m)
        if not clean or clean in seen_values or clean.strip("/") in seen_values or len(clean) < 2:
            continue
        if clean in ("/", "//", "/.", "/..", "/html", "/body", "/head"):
            continue
        seen_values.add(clean)
        seen_values.add(clean.strip("/"))
        norm_target = clean
        if base_url:
            norm_target = urllib.parse.urljoin(base_url, clean.lstrip("/"))

        candidates.append(CandidateArtifact(
            raw_value=clean,
            artifact_type="ENDPOINT",
            normalized_target=norm_target,
            confidence=0.75,
            metadata={"is_dir": clean.endswith("/")}
        ))

    # 4. UUIDs / Tokens
    for m in _UUID_PATTERN.findall(text):
        if m not in seen_values:
            seen_values.add(m)
            candidates.append(CandidateArtifact(
                raw_value=m,
                artifact_type="TOKEN",
                normalized_target=m,
                confidence=0.8,
                metadata={"subtype": "uuid"}
            ))

    for m in _TOKEN_PATTERN.findall(text):
        if m not in seen_values and len(m) >= 20:
            # Ignore standard HTML / CSS tokens if any
            if m.lower() in ("content-type", "authorization", "x-forwarded-for"):
                continue
            seen_values.add(m)
            candidates.append(CandidateArtifact(
                raw_value=m,
                artifact_type="TOKEN",
                normalized_target=m,
                confidence=0.7,
                metadata={"subtype": "hex_or_alphanumeric"}
            ))

    return candidates


def generate_post_exploitation_probes(target_url_or_path: str, base_url: str = "") -> List[str]:
    """Generates generic, pluggable post-exploitation verification and interaction commands
    for an extracted candidate artifact (e.g. newly discovered path / uploaded file).
    Extensible and completely free of hardcoded target strings.
    """
    if not target_url_or_path:
        return []

    target = target_url_or_path
    if base_url and not target.startswith("http://") and not target.startswith("https://"):
        target = urllib.parse.urljoin(base_url, target.lstrip("/"))

    probes = []
    # 1. Direct retrieval / status probe
    probes.append(f'curl -s -i "{target}"')

    # 2. If it is an executable script convention (e.g. PHP, CGI, PY, PL), test standard parameter conventions
    lower_t = target.lower()
    if any(lower_t.endswith(ext) or (ext + "?") in lower_t for ext in (
        ".php", ".phtml", ".phar", ".cgi", ".pl", ".py", ".sh", ".jsp", ".asp", ".aspx"
    )):
        # Probe common parameter passing conventions for command execution / verification
        for param in ("cmd", "c", "exec", "command", "e", "code", "q"):
            sep = "&" if "?" in target else "?"
            probes.append(f'curl -s "{target}{sep}{param}=id"')
            probes.append(f'curl -s "{target}{sep}{param}=ls+-la"')
            probes.append(f'curl -s "{target}{sep}{param}=cat+/flag*+flag.txt"')

    return probes


class ResponseProfiler:
    """Manages empirical response baselines across endpoints and evaluates anomalies."""

    def __init__(self):
        self.baselines: Dict[str, EndpointBaseline] = {}
        self.anomalous_history: List[Dict[str, Any]] = []

    def _normalize_key(self, command_or_target: str) -> str:
        """Extract a canonical endpoint/target key for baseline tracking."""
        if not command_or_target:
            return "default"
        # Try extracting URL
        m = _URL_PATTERN.search(command_or_target)
        if m:
            parsed = urllib.parse.urlparse(m.group(0))
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        # Fallback to normalized command structure
        tokens = command_or_target.strip().split()
        if tokens:
            return tokens[0].lower() + (f":{tokens[1]}" if len(tokens) > 1 and not tokens[1].startswith("-") else "")
        return "default"

    def profile_and_evaluate(
        self,
        command_or_target: str,
        output: str,
        status_code: Optional[int] = None,
        base_url: str = "",
    ) -> AnomalyResult:
        """Profiles a tool/command response, updates the empirical baseline,
        and returns an AnomalyResult if the response diverges.
        """
        key = self._normalize_key(command_or_target)
        length = len(output or "")
        struct_hash = _tokenize_structure(output or "")

        baseline = self.baselines.get(key)
        if baseline is None:
            baseline = EndpointBaseline(target_key=key)
            self.baselines[key] = baseline

        # Evaluate against current baseline before updating it
        is_anom, score, reasons = baseline.evaluate_divergence(
            length=length,
            status_code=status_code,
            struct_hash=struct_hash,
            min_samples=2,
        )

        # Update the baseline with this observation
        baseline.record(length=length, status_code=status_code, struct_hash=struct_hash)

        # Extract candidates if anomalous or if output contains plausible file/URL artifacts
        candidates: List[CandidateArtifact] = []
        if is_anom or length > 0:
            extracted = extract_generic_artifacts(output, base_url=base_url)
            for c in extracted:
                c.source_command = command_or_target
                candidates.append(c)

        result = AnomalyResult(
            is_anomalous=is_anom,
            score=score,
            reasons=reasons,
            candidate_artifacts=candidates,
            baseline_samples=baseline.sample_count,
            response_length=length,
            baseline_mean_length=baseline.mean_length,
        )

        if is_anom:
            self.anomalous_history.append({
                "target_key": key,
                "command": command_or_target,
                "length": length,
                "score": score,
                "reasons": reasons,
                "candidates": [c.to_dict() for c in candidates],
            })

        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baselines": {k: b.to_dict() for k, b in self.baselines.items()},
            "anomalous_history": list(self.anomalous_history[-50:]),
        }

    def load_dict(self, data: Optional[Dict[str, Any]]):
        if not data or not isinstance(data, dict):
            return
        for k, b_data in (data.get("baselines") or {}).items():
            if isinstance(b_data, dict):
                self.baselines[k] = EndpointBaseline.from_dict(b_data)
        self.anomalous_history = list(data.get("anomalous_history", []))
