"""
FORGE Experience/Memory data models + deterministic generalization.

This module is the CONTRACT layer for the experience memory system:

* ``AttemptRecord`` / ``ExperienceRecord`` — Pydantic DTOs describing a distilled,
  reusable FORGE experience (the shape stored in ``ExperienceModel``).
* ``RetrievedMemory`` — the unified retrieval result returned to the swarm,
  covering BOTH FORGE experiences and Playbook Vault entries.
* ``Generalizer`` — the deterministic sanitizer that enforces rule §4: a solved
  challenge must NEVER be stored as challenge-specific knowledge. Concrete
  targets, flags, credentials, tokens and secrets are stripped/parameterized so
  what remains is a *technique*, not a *solution*.

Everything here is deterministic (no LLM calls) so extraction stays fast enough
for a live competition (rule §17).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Source provenance weighting (§7). FORGE-verified experience outranks unproven
# external knowledge; repeated FORGE success outranks everything.
# ---------------------------------------------------------------------------

SOURCE_WEIGHTS: Dict[str, float] = {
    "forge_repeated_success": 1.6,   # VERY HIGH — proven across multiple runs
    "forge_success": 1.3,            # HIGH — FORGE solved it once, verified
    "external_writeup": 1.0,         # MEDIUM — imported but unverified here
    "forge_attempt": 0.7,            # LOW/MEDIUM — attempted, not yet verified
    "forge_failure": 0.55,           # still useful ("what NOT to repeat")
}

# Flags in any of the known CTF prefixes — used to scrub captured flags from
# stored experience. Mirrors swarm_orchestrator.FLAG_REGEX intent but local so
# this module has no import cycle with the swarm.
_FLAG_RE = re.compile(
    r"\b(?:picoCTF|FLAG|flag|HTB|CTF|THM|pico|CSAW|ctf)\{[^}\n]{0,200}\}",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")
_LONG_HEX_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_B64_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")


# ---------------------------------------------------------------------------
# Deterministic generalizer (§4)
# ---------------------------------------------------------------------------

class Generalizer:
    """Strips challenge-specific secrets from text, leaving a reusable technique.

    Order matters: caller-supplied concrete secrets (target tokens, the flag,
    discovered credentials/tokens) are replaced FIRST with named sentinel tokens,
    then generic structural patterns (URLs, IPs, JWTs, hex/base64 blobs) are
    parameterized. Known exploit keywords (UNION SELECT, {{7*7}}, etc.) are left
    intact so the technique stays legible.
    """

    @staticmethod
    def _sub_literal(text: str, needle: str, sentinel: str) -> str:
        needle = (needle or "").strip()
        if not needle or len(needle) < 3:
            return text
        return re.sub(re.escape(needle), sentinel, text, flags=re.IGNORECASE)

    @classmethod
    def generalize(
        cls,
        text: str,
        *,
        target_tokens: Optional[List[str]] = None,
        flag: str = "",
        secrets: Optional[List[str]] = None,
        usernames: Optional[List[str]] = None,
    ) -> str:
        """Return a generalized copy of *text* with concrete artifacts removed."""
        if not text:
            return ""
        out = str(text)

        # 1) Named, challenge-specific values first (most specific).
        for tok in (target_tokens or []):
            tok = (tok or "").strip()
            if not tok:
                continue
            # A bare host:port / path / IP / URL target token.
            if tok.startswith("http"):
                out = cls._sub_literal(out, tok, "{TARGET_URL}")
                # The bare host also appears WITHOUT its scheme — as a path segment,
                # a Host: header, a "Resolving <host>..." log line. Substituting only
                # the full URL left `{TARGET_URL}/a-upload.ctf/upload.php` and any
                # scheme-less mention intact, so the concrete target still leaked into
                # stored experience (rule §4). Scrubbed AFTER the URL so the full-URL
                # literal is still matchable. {HOST} is the same sentinel _IPV4_RE
                # already uses, and _sub_literal ignores needles under 3 chars, so a
                # degenerate host cannot match everything.
                _host = urlparse(tok).hostname or ""
                if _host:
                    out = cls._sub_literal(out, _host, "{HOST}")
            elif "/" in tok or "\\" in tok:
                out = cls._sub_literal(out, tok, "{ARTIFACT_PATH}")
            else:
                out = cls._sub_literal(out, tok, "{TARGET}")

        if flag:
            out = cls._sub_literal(out, flag, "{FLAG}")
        for s in (secrets or []):
            out = cls._sub_literal(out, s, "{SECRET}")
        for u in (usernames or []):
            out = cls._sub_literal(out, u, "{USER}")

        # 2) Generic structural patterns (order: most-specific token classes first).
        out = _FLAG_RE.sub("{FLAG}", out)
        out = _JWT_RE.sub("{JWT}", out)
        out = _URL_RE.sub("{TARGET_URL}", out)
        out = _IPV4_RE.sub("{HOST}", out)
        # host:port left over from a scrubbed URL (":{PORT}") — normalize ports.
        out = re.sub(r":(\d{2,5})\b", ":{PORT}", out)
        out = _LONG_HEX_RE.sub("{HEX}", out)
        out = _B64_RE.sub("{B64}", out)
        return out


# ---------------------------------------------------------------------------
# Technique classification + blue-team indicators (deterministic heuristics)
# ---------------------------------------------------------------------------

# (matched_substrings, technique_label, tags, signatures, success_indicators)
_TECHNIQUE_RULES = [
    (("ssti", "jinja", "{{7*7}}", "{{7*7", "render_template_string", "twig", "freemarker"),
     "Server-Side Template Injection (SSTI)",
     ["ssti", "template_injection"],
     ["Jinja2", "Werkzeug", "render_template_string", "Twig"],
     [r"uid=\d+", r"49$", r"root:x:0:0"]),
    (("union select", "' or 1=1", "or 1=1--", "sqlmap", "information_schema", "' or '1'='1"),
     "SQL Injection",
     ["sqli", "sql_injection"],
     ["SQL syntax", "UNION SELECT", "information_schema", "sqlite3.OperationalError"],
     [r"(?:SQL syntax|mysql_fetch|sqlite3\.OperationalError|PG::)", r"\|\s*flag"]),
    (("../", "..%2f", "/etc/passwd", "?file=", "&file=", "?page=", "lfi", "php://filter", "php://"),
     "Local File Inclusion / Path Traversal",
     ["lfi", "path_traversal"],
     ["root:x:0:0", "php://filter", "../../"],
     [r"root:x:0:0", r"daemon:x:"]),
    (("multipart/form-data", "upload", ".php5", ".phtml", "content-type: image", "filename="),
     "File Upload Validation Bypass",
     ["upload", "file_upload_bypass"],
     ["multipart/form-data", "Content-Disposition", "uploads/"],
     [r"(?:uploaded|Upload successful)", r"\.ph(?:p|tml)"]),
    (("jwt", "eyj", "alg\":\"none", "rs256", "hs256", "bearer "),
     "JWT Forgery / Weak Signature",
     ["jwt", "token_forgery"],
     ["eyJ", "alg", "none algorithm", "HS256"],
     [r"(?:HTTP/1\.[01] 200|admin|role)"]),
    # Reverse Engineering — MUST appear before Binary Exploitation so that a pure
    # reversing session (disassemble, decompile, strings, ltrace) is not pulled into
    # "Binary Exploitation / Memory Corruption" just because gdb/checksec appeared.
    (("ghidra", "ida ", "radare2", "r2 ", "objdump -d", "disassemble", "decompile",
      "ltrace", "strace", "angr", "z3.", "strings ", "readelf", "file "),
     "Reverse Engineering / Static Analysis",
     ["rev", "reverse_engineering"],
     ["ELF", "PE", "Mach-O", "stripped", "not stripped"],
     [r"(?:flag|correct|success)"]),
    (("pwntools", "p32(", "p64(", "rop", "ret2", "pattern_offset", "cyclic(",
      "shellcraft", "flat(", "process(", "remote("),
     "Binary Exploitation / Memory Corruption",
     ["pwn", "buffer_overflow", "rop"],
     ["ELF", "checksec", "ROP", "GOT"],
     [r"(?:\[\+\] Opening connection|Switching to interactive)"]),
    (("<script>", "onerror=", "xss", "document.cookie", "alert("),
     "Cross-Site Scripting (XSS)",
     ["xss"],
     ["<script>", "reflected", "innerHTML"],
     [r"(?:document\.cookie|alert\()"]),
    (("hydra", "brute", "wordlist", "rockyou", "password spray", "credential"),
     "Credential Brute-Force / Weak Auth",
     ["auth", "bruteforce"],
     ["401 Unauthorized", "Login", "Invalid password"],
     [r"(?:200 OK|Welcome|Dashboard)"]),
    # XXE: require co-occurrence — `<!doctype` alone appears in normal HTML pages.
    # A standalone `xxe` mention is enough; `<!doctype` only counts when paired with
    # `<!entity` or `system "file` (the actual XXE payload shape).
    (("xxe", "<!entity", "system \"file"),
     "XML External Entity (XXE)",
     ["xxe"],
     ["<!ENTITY", "SYSTEM", "DOCTYPE"],
     [r"root:x:0:0"]),
    (("nc ", "reverse shell", "bash -i", "/dev/tcp/", "msfvenom", "rev shell"),
     "Command Injection / Reverse Shell",
     ["rce", "command_injection"],
     ["; id", "$(", "`id`", "/dev/tcp"],
     [r"uid=\d+\(", r"gid=\d+"]),
    (("xor", "aes", "rsa", "frequency analysis", "vigenere", "base64 -d", "openssl enc", "rot13"),
     "Cryptographic Weakness Exploitation",
     ["crypto"],
     ["ciphertext", "key", "IV", "mode"],
     [r"\{FLAG\}"]),
]

# Maps the challenge's declared category to the technique-rule tags it corresponds to.
# Used to boost category-agreeing rules during scored classification.
_CATEGORY_TAG_MAP: Dict[str, List[str]] = {
    "web": ["ssti", "template_injection", "sqli", "sql_injection", "lfi", "path_traversal",
            "upload", "file_upload_bypass", "jwt", "token_forgery", "xss", "auth", "bruteforce",
            "xxe", "rce", "command_injection"],
    "pwn": ["pwn", "buffer_overflow", "rop"],
    "rev": ["rev", "reverse_engineering"],
    "reverse": ["rev", "reverse_engineering"],
    "reverse_engineering": ["rev", "reverse_engineering"],
    "crypto": ["crypto"],
    "forensics": ["rev", "reverse_engineering", "crypto"],
    "misc": [],
}


def classify_technique(
    evidence_text: str = "",
    category: str = "",
    *,
    commands: Optional[List[str]] = None,
    winning_chain: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Deterministically map observed evidence to a generalized technique + tags.

    Uses scored ranking:
    1. Winning exploit chain (decisive commands that produced the flag) receives highest priority.
    2. Agent commands (active actions) receive 3x weighting over passive output/text.
    3. Distinct needle hits across evidence text provide base score.
    4. Boost by category agreement (+5), penalize category disagreement (-2).
    5. Tie-breaking prioritizes winning chain presence > command presence > total needle hits.

    Falls back to a category-scoped generic label when nothing specific matches.
    """
    low = (evidence_text or "").lower()
    cat = (category or "").lower().strip()
    cat_tags = set(_CATEGORY_TAG_MAP.get(cat, []))

    # For command-tier and winning-chain scoring, take only the first line of each
    # command entry (stripping multi-line heredocs/inline script bodies so that file
    # contents/comments with keywords do not masquerade as active command verbs).
    cmds_low = [
        (c or "").lower().splitlines()[0].strip()
        for c in (commands or [])
        if c and (c or "").strip()
    ]
    cmds_low = [c for c in cmds_low if c]
    win_low = [
        (c or "").lower().splitlines()[0].strip()
        for c in (winning_chain or [])
        if c and (c or "").strip()
    ]
    win_low = [w for w in win_low if w]

    best_score_tuple = (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
    best_rule = None

    for needles, label, tags, sigs, indicators in _TECHNIQUE_RULES:
        # Check needle matches in different tiers
        text_hits = sum(1 for n in needles if n in low)
        cmd_needle_hits = sum(1 for n in needles if any(n in c for c in cmds_low))
        win_needle_hits = sum(1 for n in needles if any(n in w for w in win_low))

        # Total distinct needle hits across all sources
        total_needle_hits = sum(
            1 for n in needles
            if (n in low or any(n in c for c in cmds_low) or any(n in w for w in win_low))
        )
        if total_needle_hits == 0:
            continue

        # Count how many individual commands matched any needle of this rule
        win_cmd_matches = sum(1 for w in win_low if any(n in w for n in needles))
        cmd_matches = sum(1 for c in cmds_low if any(n in c for n in needles))

        # 1. Winning chain score: decisive commands that actually solved the challenge
        winning_score = (win_needle_hits * 10.0) + (min(win_cmd_matches, 6) * 3.0)

        # 2. Command score: agent-issued actions weighted 3x over passive output
        command_score = (cmd_needle_hits * 3.0) + (min(cmd_matches, 10) * 1.0)

        # 3. Base evidence score: passive text, output, descriptions
        base_score = float(total_needle_hits)

        raw_score = winning_score + command_score + base_score

        # 4. Category-agreement boost / disagreement penalty
        category_modifier = 0.0
        if cat_tags and any(t in cat_tags for t in tags):
            category_modifier = 5.0
        elif cat_tags:
            category_modifier = -2.0

        total_score = raw_score + category_modifier

        # Score tuple for ranking and decisive tie-breaking:
        # (total_score, winning_score, command_score, total_needle_hits)
        score_tuple = (total_score, winning_score, command_score, total_needle_hits)

        if score_tuple > best_score_tuple:
            best_score_tuple = score_tuple
            best_rule = (label, tags, sigs, indicators)

    if best_rule is not None:
        label, tags, sigs, indicators = best_rule
        return {
            "technique": label,
            "tags": list(dict.fromkeys(tags + ([cat] if cat else []))),
            "signatures": sigs,
            "success_indicators": indicators,
        }
    cat = cat or "general"
    return {
        "technique": f"{cat.upper()} solve methodology",
        "tags": [cat, "custom_exploit"],
        "signatures": [],
        "success_indicators": [],
    }


# Blue-team knowledge derived from a red-team technique (§11). Deterministic —
# no SIEM, just reusable defensive pointers keyed by tag.
_DETECTION_BY_TAG: Dict[str, Dict[str, List[str]]] = {
    "ssti": {
        "indicators": ["Template engine errors in application logs", "Unusual arithmetic/format tokens in request params ({{ }}, ${ })"],
        "logs": ["web application logs", "WAF logs"],
        "investigation": ["Grep request params for template delimiters", "Review template rendering call sites for user input"],
        "containment": ["Sandbox the template engine", "Reject template metacharacters in input"],
    },
    "sqli": {
        "indicators": ["SQL syntax errors returned to client", "Spikes in UNION/OR-based query strings", "Abnormal DB error rates"],
        "logs": ["web access logs", "database query logs"],
        "investigation": ["Correlate 500s with query params", "Search access logs for UNION SELECT / information_schema"],
        "containment": ["Parameterize queries", "Deploy WAF rule for SQLi signatures"],
    },
    "lfi": {
        "indicators": ["Requests containing ../ or /etc/passwd", "Access to unexpected local file paths"],
        "logs": ["web access logs"],
        "investigation": ["Search access logs for traversal sequences", "Audit file-inclusion endpoints"],
        "containment": ["Whitelist include paths", "Disable url_include / restrict open_basedir"],
    },
    "upload": {
        "indicators": ["Executable/script files in upload directories", "Requests to uploaded files with script extensions"],
        "logs": ["web access logs", "file integrity monitoring"],
        "investigation": ["List recently written files in upload dirs", "Check content-type vs extension mismatches"],
        "containment": ["Enforce server-side content validation", "Store uploads outside webroot, disable execution"],
    },
    "jwt": {
        "indicators": ["Tokens with alg=none or weak HS256 secrets", "Privilege changes without re-auth"],
        "logs": ["authentication logs"],
        "investigation": ["Decode presented JWTs, inspect alg/claims", "Verify signature validation is enforced"],
        "containment": ["Reject alg=none", "Rotate signing keys, enforce strong secrets"],
    },
    "pwn": {
        "indicators": ["Service crashes / restarts", "Anomalous outbound connections from a service"],
        "logs": ["process telemetry", "core dumps", "network telemetry"],
        "investigation": ["Inspect crash logs / core dumps", "Review recent binary/service exploitation CVEs"],
        "containment": ["Enable ASLR/NX/stack canaries", "Restart with hardened flags, patch binary"],
    },
    "rce": {
        "indicators": ["Unexpected child processes from web/service user", "Reverse-shell connections (/dev/tcp, nc)"],
        "logs": ["process telemetry", "network telemetry", "web access logs"],
        "investigation": ["Trace process tree for spawned shells", "Correlate outbound connections with request logs"],
        "containment": ["Kill spawned processes", "Sanitize command inputs, restrict egress"],
    },
    "xss": {
        "indicators": ["Script payloads in stored fields", "Anomalous outbound requests carrying cookies"],
        "logs": ["web access logs"],
        "investigation": ["Search stored fields for <script>/onerror", "Review CSP violation reports"],
        "containment": ["Output-encode user content", "Deploy a strict Content-Security-Policy"],
    },
    "auth": {
        "indicators": ["High volume of failed logins", "Successful login after many failures"],
        "logs": ["authentication logs"],
        "investigation": ["Review auth logs for brute-force bursts", "Check for credential stuffing patterns"],
        "containment": ["Rate-limit / lockout", "Enforce MFA and strong passwords"],
    },
    "rev": {
        "indicators": ["Binary extraction or download from production assets", "Decompilation tool artifacts on disk"],
        "logs": ["file access logs", "download logs"],
        "investigation": ["Check for exfiltrated binaries", "Review build artifacts for leaked symbols"],
        "containment": ["Strip symbols from release builds", "Restrict binary distribution"],
    },
}


def derive_detection_indicators(tags: List[str]) -> Dict[str, List[str]]:
    """Merge blue-team detection knowledge for all matched technique tags (§11)."""
    merged: Dict[str, List[str]] = {"indicators": [], "logs": [], "investigation": [], "containment": []}
    for tag in tags or []:
        block = _DETECTION_BY_TAG.get(tag.lower())
        if not block:
            continue
        for key, vals in block.items():
            for v in vals:
                if v not in merged[key]:
                    merged[key].append(v)
    return merged


# ---------------------------------------------------------------------------
# Environment-requirement inference for environment-aware skills (Phase 2, §12)
# ---------------------------------------------------------------------------

# Tools that only exist / are meaningfully used on Linux CTF hosts. If an
# experience's commands invoke one of these, the technique implicitly requires a
# Linux execution environment (FORGE itself may be driving from Windows).
_LINUX_ONLY_TOOLS = {
    "nmap", "ffuf", "gobuster", "feroxbuster", "dirsearch", "hydra", "john",
    "hashcat", "binwalk", "steghide", "exiftool", "tshark", "objdump", "readelf",
    "gdb", "radare2", "ghidra", "sqlmap", "nikto", "wpscan", "msfvenom", "searchsploit",
}
# Tools that are portable (present or trivially installable on Windows too).
_PORTABLE_TOOLS = {
    "curl", "wget", "python", "python3", "pip", "pip3", "openssl", "base64",
    "git", "nc", "ncat", "7z", "unzip", "tar", "ssh",
}
# Python solver libs worth recording as a prerequisite when a script imports them.
_KNOWN_PY_LIBS = {"pwn", "pwntools", "requests", "bs4", "cryptography", "flask_unsign", "scapy", "z3"}

_TOOL_TOKEN_RE = re.compile(r"\b([a-z][a-z0-9_\-]{1,20})\b")
_PY_IMPORT_RE = re.compile(r"(?:^|\n)\s*(?:import|from)\s+([a-zA-Z0-9_]+)")


def infer_environment_requirements(commands: List[str], category: str = "") -> Dict[str, Any]:
    """Deterministically infer a skill's execution-environment requirements (§12).

    Returns ``{"required_os", "tools", "python_libs"}``. The rule is conservative:
    an ``"any"`` OS is only downgraded to ``"linux"`` when the commands clearly rely
    on a Linux-only tool, so a portable technique (curl/python) stays OS-independent
    and remains runnable from the Windows host itself. This never executes anything.
    """
    tools: List[str] = []
    libs: List[str] = []
    blob = "\n".join(c for c in (commands or []) if c)
    low = blob.lower()

    for tok in _TOOL_TOKEN_RE.findall(low):
        if tok in _LINUX_ONLY_TOOLS or tok in _PORTABLE_TOOLS:
            if tok not in tools:
                tools.append(tok)
    for mod in _PY_IMPORT_RE.findall(blob):
        m = mod.strip().lower()
        if m in _KNOWN_PY_LIBS and m not in libs:
            libs.append(m)

    required_os = "linux" if any(t in _LINUX_ONLY_TOOLS for t in tools) else "any"
    # pwn/pwntools binary exploitation is effectively Linux-bound.
    if (category or "").lower() == "pwn" or any(l in ("pwn", "pwntools") for l in libs):
        required_os = "linux"

    return {"required_os": required_os, "tools": tools[:12], "python_libs": libs[:8]}


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

class AttemptRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    sequence: int = 0
    approach: str = ""
    technique: str = ""
    outcome: str = "failure"           # success | failure
    reason: str = ""
    evidence: str = ""


class ExperienceRecord(BaseModel):
    """Distilled, generalized representation of a completed FORGE run (§2)."""
    model_config = ConfigDict(populate_by_name=True)

    id: Optional[str] = None
    source: str = "forge_run"
    source_run_id: Optional[str] = None
    source_challenge_id: Optional[str] = None
    challenge_name: str = ""

    category: str = "web"
    difficulty: str = "MEDIUM"
    technique: str = ""
    tags: List[str] = Field(default_factory=list)

    target_characteristics: Dict[str, Any] = Field(default_factory=dict)
    initial_observations: str = ""
    observed_conditions: str = ""
    applicable_conditions: str = ""
    discovered_endpoints: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    vulnerabilities: List[str] = Field(default_factory=list)

    successful_techniques: List[str] = Field(default_factory=list)
    failed_techniques: List[Dict[str, str]] = Field(default_factory=list)
    commands_used: List[str] = Field(default_factory=list)
    important_tool_outputs: List[str] = Field(default_factory=list)
    successful_attack_chain: List[str] = Field(default_factory=list)
    verification_evidence: str = ""
    success_indicators: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    generalized_strategy: str = ""

    detection_indicators: Dict[str, Any] = Field(default_factory=dict)

    # Environment requirements for environment-aware skills (Phase 2, Step 12).
    required_os: str = "any"                                   # any | linux | windows | darwin
    required_tools: List[str] = Field(default_factory=list)
    required_python_libs: List[str] = Field(default_factory=list)

    outcome: str = "success"
    confidence: float = 0.6

    attempts: List[AttemptRecord] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RetrievedMemory(BaseModel):
    """Unified retrieval result covering FORGE experiences AND Playbook Vault entries (§10)."""
    model_config = ConfigDict(populate_by_name=True)

    kind: str                      # "experience" | "playbook"
    id: str
    technique: str = ""
    category: str = ""
    applicable_conditions: str = ""
    strategy: str = ""
    outcome: str = "success"       # success | failure | reference
    success_indicators: List[str] = Field(default_factory=list)
    failed_approaches: List[str] = Field(default_factory=list)
    confidence: float = 0.6
    success_rate: float = 1.0
    source: str = "forge_run"
    provenance: Dict[str, Any] = Field(default_factory=dict)
    environment_fit: float = 1.0   # 0..1 fit to the current execution environment (§12)
    score: float = 0.0
