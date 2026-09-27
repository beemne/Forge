"""
FORGE Agent Runtime — the observation engine.

Turns raw tool output (an :class:`~backend.agent_runtime.action.ExecResult`) into a
structured :class:`Observation`. Two hard rules (Step 2):

1. Observations are derived **only** from actual tool output / evidence. Nothing is
   inferred from what the model *said* — only from bytes a command actually produced.
2. Extraction is fully deterministic (regex + fixed signature tables). No LLM call.

`novelty` is computed against the current :class:`MissionState` so the runtime can
tell a genuinely new discovery from output it has effectively seen before — that is
the primary no-progress signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Canonical flag patterns live in the verifier (single source of truth for the runtime).
from backend.agent_runtime.verifier import (
    FLAG_REGEX, FALSE_FLAG_PATTERNS, AnswerResolver, AnswerCandidate, AnswerSource,
)

# ── Deterministic extraction patterns ────────────────────────────────────────
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]}]+", re.IGNORECASE)
_LOCATION_RE = re.compile(r"^\s*Location:\s*(\S+)", re.IGNORECASE | re.MULTILINE)
# dirb/gobuster/ffuf-style hit lines: a path with an HTTP status nearby.
_DIRBUST_RE = re.compile(r"(/[A-Za-z0-9._/\-]+)\s*(?:\(Status:\s*\d{3}\)|\[Status:\s*\d{3}\]|\s\d{3}\b)")
# nmap-style open port/service.
_NMAP_RE = re.compile(r"^(\d{1,5})/(tcp|udp)\s+open\s+(\S+)", re.IGNORECASE | re.MULTILINE)
# explicit file-creation phrasing only (avoids hallucinating files from arbitrary text).
_FILE_SAVED_RE = re.compile(
    r"(?:saved to|written to|downloaded to|output written to|saving to|-o)\s+['\"]?([A-Za-z0-9._/\\\-]+)",
    re.IGNORECASE,
)
_FILE_UPLOADED_RE = re.compile(
    r"(?:uploaded successfully to|successfully uploaded to|uploaded to|upload succeeded|"
    r"upload successful|upload successfully|stored at|saved at|uploaded file|"
    r"(?:path|destination)\s*:)\s*(?:is|to|at|:)?\s+['\"]?([A-Za-z0-9._/\\\-]+)",
    re.IGNORECASE,
)
_SOURCE_FILE_RE = re.compile(
    r'(?:open|fopen|file_get_contents|read_file|include|require|include_once|require_once)\s*\(\s*[\'"]([A-Za-z0-9._/\\\-]+)[\'"]',
    re.IGNORECASE,
)
_SESSION_RE = re.compile(r'\[SESSION:\s*([A-Za-z0-9_\-]+)\]')
_CRED_RE = re.compile(r"(?:password|passwd|pwd|secret|api[_-]?key)\s*[:=]\s*['\"]?([^\s'\"]{3,80})", re.IGNORECASE)
_BASIC_AUTH_RE = re.compile(r"\b([A-Za-z0-9_.\-]{2,40}):([^\s:@/]{3,40})@")

# Vulnerability signatures — (name, compiled pattern). Conservative + high-signal.
_VULN_SIGNATURES = [
    ("SQL injection", re.compile(r"SQL syntax|mysql_fetch|ORA-\d{5}|SQLSTATE|unclosed quotation mark", re.I)),
    ("Local File Inclusion", re.compile(r"root:.*:0:0:|\[boot loader\]|/etc/passwd", re.I)),
    ("Server-Side Template Injection", re.compile(r"\b(?:49|1337)\b.*(?:template|render)|TemplateSyntaxError|jinja2", re.I)),
    ("Stack trace / debug leak", re.compile(r"Traceback \(most recent call last\)|Werkzeug Debugger|Whoops", re.I)),
    ("Directory listing", re.compile(r"Index of /|<title>Directory listing for", re.I)),
    ("Exposed .git", re.compile(r"\[core\]\s*repositoryformatversion|ref:\s*refs/heads/", re.I)),
]

# Technology fingerprints (fallback when the experience extractor is unavailable).
_TECH_SIGNATURES = [
    ("Flask", re.compile(r"\bflask\b|werkzeug", re.I)),
    ("Django", re.compile(r"\bdjango\b|csrftoken", re.I)),
    ("nginx", re.compile(r"\bnginx\b", re.I)),
    ("Apache", re.compile(r"\bapache\b|mod_", re.I)),
    ("PHP", re.compile(r"\bphp/?\d|X-Powered-By:\s*PHP", re.I)),
    ("Express", re.compile(r"\bexpress\b|X-Powered-By:\s*Express", re.I)),
    ("WordPress", re.compile(r"wp-content|wp-includes|wordpress", re.I)),
    ("Node.js", re.compile(r"\bnode\.js\b|nodejs", re.I)),
    ("OpenSSH", re.compile(r"OpenSSH", re.I)),
]

# Errors we should surface for recovery classification.
_ERROR_HINTS = re.compile(
    r"command not found|not recognized|No such file or directory|Permission denied|"
    r"Connection refused|Could not resolve host|timed out|SyntaxError|ModuleNotFoundError|"
    r"ImportError|Traceback \(most recent call last\)",
    re.IGNORECASE,
)

_MAX_IMPORTANT = 1500


def _detect_technologies(text: str) -> List[str]:
    """Reuse the experience extractor's tech detector when present; else fall back."""
    try:
        from backend.knowledge.experience_extractor import experience_extractor
        techs = experience_extractor._detect_technologies(text)  # noqa: SLF001 (intentional reuse)
        if techs:
            return list(techs)
    except Exception:
        pass
    found: List[str] = []
    for name, pat in _TECH_SIGNATURES:
        if pat.search(text) and name not in found:
            found.append(name)
    return found


from backend.agents.response_profiler import extract_generic_artifacts, ResponseProfiler, AnomalyResult

@dataclass
class Observation:
    """Structured, evidence-only view of a single command's output."""
    novelty: bool = False
    new_endpoints: List[str] = field(default_factory=list)
    new_services: List[str] = field(default_factory=list)
    new_technologies: List[str] = field(default_factory=list)
    new_files: List[str] = field(default_factory=list)
    file_provenance: Dict[str, str] = field(default_factory=dict)
    interactive_sessions: List[str] = field(default_factory=list)
    new_credentials: List[str] = field(default_factory=list)
    new_vulnerabilities: List[str] = field(default_factory=list)
    new_headers: Dict[str, str] = field(default_factory=dict)
    new_cookies: Dict[str, str] = field(default_factory=dict)
    flag_candidates: List[str] = field(default_factory=list)
    answer_candidates: List[Any] = field(default_factory=list)
    candidate_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    anomalous_response: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    important_output: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if "answer_candidates" in d:
            d["answer_candidates"] = [
                c.to_dict() if hasattr(c, "to_dict") else str(c) for c in (self.answer_candidates or [])
            ]
        return d


class ObservationEngine:
    """Deterministically converts an ExecResult into an Observation (no model calls)."""

    def __init__(self, resolver: Optional[AnswerResolver] = None, profiler: Optional[ResponseProfiler] = None):
        self.resolver = resolver or AnswerResolver()
        self.profiler = profiler or ResponseProfiler()

    def observe(self, result: Any, state: Optional[Any] = None) -> Observation:
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        cmd_or_target = getattr(result, "command", "") or ""
        combined = f"{stdout}\n{stderr}"
        obs = Observation()

        # ── Endpoints ──
        for m in _URL_RE.findall(stdout):
            obs.new_endpoints.append(m.rstrip(".,);"))
        for m in _LOCATION_RE.findall(combined):
            obs.new_endpoints.append(m.strip())
        for m in _DIRBUST_RE.findall(stdout):
            obs.new_endpoints.append(m if isinstance(m, str) else m[0])

        # ── Services (nmap) ──
        for port, proto, svc in _NMAP_RE.findall(combined):
            obs.new_services.append(f"{port}/{proto} {svc}")

        # ── Technologies ──
        obs.new_technologies = _detect_technologies(combined)

        # ── Files (explicit creation phrasing only -> LOCAL_FILE) ──
        for m in _FILE_SAVED_RE.findall(combined):
            obs.new_files.append(m)
            obs.file_provenance[m] = "LOCAL_FILE"

        # ── Uploaded / discovered remote files -> REMOTE_FILE ──
        for m in _FILE_UPLOADED_RE.findall(combined):
            clean_m = m.strip().rstrip(".,);")
            if clean_m and len(clean_m) > 1:
                obs.new_files.append(clean_m)
                if clean_m not in obs.file_provenance:
                    obs.file_provenance[clean_m] = "REMOTE_FILE"
                if clean_m.startswith("/") or clean_m.startswith("uploads/") or "/" in clean_m:
                    obs.new_endpoints.append(clean_m)

        # ── Source code file references -> SOURCE_CODE_REFERENCE ──
        # Runs BEFORE the generic extractor below, which stamps every `word.ext` token
        # it recognises as a PATH with REMOTE_FILE. `file_provenance` is
        # first-writer-wins, so in the previous order the generic pass always won and
        # SOURCE_CODE_REFERENCE was unreachable for any file carrying a common
        # extension — e.g. the `config/db_secret.php` named in a PHP require_once
        # warning, which is a *reference to* a source file, not an uploaded artifact.
        # The explicit upload phrasing above still outranks this: "File uploaded to X"
        # is a stronger claim about X than a mere mention of it.
        for m in _SOURCE_FILE_RE.findall(combined):
            obs.new_files.append(m)
            if m not in obs.file_provenance:
                obs.file_provenance[m] = "SOURCE_CODE_REFERENCE"

        # ── Generic Structural Artifact Extraction (Pattern-based, no keyword reliance) ──
        target_base = getattr(state, "target", "") or getattr(state, "target_url", "") or ""
        generic_artifacts = extract_generic_artifacts(combined, base_url=target_base)
        for cand in generic_artifacts:
            obs.candidate_artifacts.append(cand.to_dict())
            if cand.artifact_type in ("PATH", "ENDPOINT", "URL"):
                val = cand.normalized_target or cand.raw_value
                if val and val not in obs.new_endpoints:
                    obs.new_endpoints.append(val)
                if cand.artifact_type == "PATH":
                    raw_val = cand.raw_value
                    if raw_val not in obs.new_files:
                        obs.new_files.append(raw_val)
                    if raw_val not in obs.file_provenance:
                        obs.file_provenance[raw_val] = "REMOTE_FILE"

        # ── Empirical Baseline Profiling & Anomaly Evaluation ──
        status_code = getattr(result, "status_code", None)
        if self.profiler and (stdout or stderr):
            anom_result = self.profiler.profile_and_evaluate(
                command_or_target=cmd_or_target,
                output=stdout or stderr,
                status_code=status_code,
                base_url=target_base,
            )
            if anom_result.is_anomalous:
                obs.anomalous_response = anom_result.to_dict()

        # ── Interactive Sessions ──
        for sess in _SESSION_RE.findall(combined):
            obs.interactive_sessions.append(sess)

        # ── Credentials ──
        for m in _CRED_RE.findall(combined):
            obs.new_credentials.append(m)
        for user, pwd in _BASIC_AUTH_RE.findall(combined):
            obs.new_credentials.append(f"{user}:{pwd}")

        # ── Headers / cookies (from HTTP-ish output) ──
        obs.new_headers = self._extract_headers(combined)
        obs.new_cookies = self._extract_cookies(combined)

        # ── Vulnerabilities ──
        for name, pat in _VULN_SIGNATURES:
            if pat.search(combined):
                obs.new_vulnerabilities.append(name)

        # ── Generic and Flag Candidate Extraction ──
        task_context = {}
        if state is not None:
            task_context = {
                "description": getattr(state, "description", "") or getattr(state, "current_objective", ""),
                "category": getattr(state, "category", ""),
                "flag_pattern": getattr(state, "flag_format", ""),
                "target_scope": getattr(state, "target", ""),
            }

        extracted_cands = self.resolver.extract_candidates(
            combined, task_context=task_context, source=AnswerSource.TOOL_OUTPUT
        )
        obs.answer_candidates = extracted_cands
        for cand in extracted_cands:
            obs.flag_candidates.append(cand.value)

        # Explicit fallback flag regex scan on stdout
        for m in FLAG_REGEX.findall(stdout):
            cand_str = m if isinstance(m, str) else (m[0] if m else "")
            cand_str = cand_str.strip()
            if cand_str and not FALSE_FLAG_PATTERNS.search(cand_str):
                obs.flag_candidates.append(cand_str)

        # ── Errors ──
        for hint in _ERROR_HINTS.findall(combined):
            obs.errors.append(hint)
        if getattr(result, "exit_code", 0) not in (0, None):
            obs.errors.append(f"exit_code={getattr(result, 'exit_code')}")
        fc = getattr(result, "failure_category", None)
        if fc:
            obs.errors.append(f"failure:{fc}")

        # De-dupe while preserving order.
        for name in ("new_endpoints", "new_services", "new_technologies", "new_files",
                     "new_credentials", "new_vulnerabilities", "flag_candidates", "errors"):
            setattr(obs, name, list(dict.fromkeys(getattr(obs, name))))

        obs.important_output = self._important(stdout, stderr)
        obs.novelty = self._compute_novelty(obs, state)
        obs.summary = self._summarize(obs, result)
        return obs


    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_headers(text: str) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        for line in text.splitlines():
            m = re.match(r"^([A-Za-z][A-Za-z0-9\-]{1,40}):\s?(.+)$", line.strip())
            if not m:
                continue
            name, value = m.group(1), m.group(2).strip()
            # Only well-known response headers — avoids scraping arbitrary "Key: value" prose.
            if name.lower() in {
                "server", "x-powered-by", "content-type", "location", "www-authenticate",
                "x-forwarded-for", "set-cookie", "x-flag", "x-backend-server",
            } and 0 < len(value) <= 256:
                headers[name] = value
        return headers

    @staticmethod
    def _extract_cookies(text: str) -> Dict[str, str]:
        cookies: Dict[str, str] = {}
        for m in re.findall(r"Set-Cookie:\s*([^=;\s]+)=([^;\s]+)", text, re.IGNORECASE):
            cookies[m[0]] = m[1]
        return cookies

    @staticmethod
    def _important(stdout: str, stderr: str) -> str:
        body = stdout.strip() or stderr.strip()
        if len(body) <= _MAX_IMPORTANT:
            return body
        # Keep head + tail — the informative parts of most tool output.
        head = body[: _MAX_IMPORTANT // 2]
        tail = body[-_MAX_IMPORTANT // 2:]
        return f"{head}\n...[{len(body) - _MAX_IMPORTANT} chars elided]...\n{tail}"

    @staticmethod
    def _compute_novelty(obs: Observation, state: Optional[Any]) -> bool:
        if obs.anomalous_response is not None:
            return True
        buckets = [
            obs.new_endpoints, obs.new_services, obs.new_technologies, obs.new_files,
            obs.new_credentials, obs.new_vulnerabilities, obs.flag_candidates,
        ]
        if state is None:
            return any(buckets) or bool(obs.new_headers or obs.new_cookies or obs.candidate_artifacts)
        # Novel only if at least one extracted item is NOT already known.
        known_maps = {
            "endpoints": set(getattr(state, "known_endpoints", []) or []),
            "services": set(getattr(state, "known_services", []) or []),
            "tech": set(getattr(state, "technologies", []) or []),
            "files": set(getattr(state, "known_files", []) or []),
            "creds": set(getattr(state, "credentials", []) or []),
            "vulns": set(getattr(state, "vulnerabilities", []) or []),
            "flags": set(getattr(state, "flag_candidates", []) or []),
        }
        checks = [
            (obs.new_endpoints, known_maps["endpoints"]),
            (obs.new_services, known_maps["services"]),
            (obs.new_technologies, known_maps["tech"]),
            (obs.new_files, known_maps["files"]),
            (obs.new_credentials, known_maps["creds"]),
            (obs.new_vulnerabilities, known_maps["vulns"]),
            (obs.flag_candidates, known_maps["flags"]),
        ]
        for found, known in checks:
            if any(item not in known for item in found):
                return True
        # New header/cookie keys also count as novelty.
        if any(k not in (getattr(state, "headers", {}) or {}) for k in obs.new_headers):
            return True
        if any(k not in (getattr(state, "cookies", {}) or {}) for k in obs.new_cookies):
            return True
        return False

    @staticmethod
    def _summarize(obs: Observation, result: Any) -> str:
        parts: List[str] = []
        if obs.anomalous_response:
            parts.append("ANOMALOUS_RESPONSE detected")
        if obs.flag_candidates:
            parts.append(f"{len(obs.flag_candidates)} flag candidate(s)")
        if obs.new_vulnerabilities:
            parts.append("vuln: " + ", ".join(obs.new_vulnerabilities[:3]))
        if obs.new_endpoints:
            parts.append(f"{len(obs.new_endpoints)} endpoint(s)")
        if obs.new_services:
            parts.append(f"{len(obs.new_services)} service(s)")
        if obs.new_technologies:
            parts.append("tech: " + ", ".join(obs.new_technologies[:4]))
        if obs.new_credentials:
            parts.append(f"{len(obs.new_credentials)} credential(s)")
        if obs.errors and not parts:
            parts.append("error: " + "; ".join(obs.errors[:2]))
        if not parts:
            status = getattr(result, "status", "SUCCESS")
            parts.append("no new signal" if status == "SUCCESS" else f"{status.lower()}, no new signal")
        return "; ".join(parts)
