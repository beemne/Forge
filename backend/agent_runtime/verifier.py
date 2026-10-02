"""
FORGE Agent Runtime — Answer & Flag Resolver / Verifier.

WHAT: Generic answer candidate recognition, semantic resolution, and authoritative verification.
HOW:  Model-powered tools (vision_read), deterministic tools, and regex/signature tables.
WHETHER: Privilege, capability, and safety controls (unmodified).

Detection, Resolution, and Verification are distinct concerns:

1. Candidate Extraction:
   A candidate can be discovered from any evidence source (tool output, vision_read,
   reconstructed artifacts, decoded directives, or agent reasoning).
2. Answer Resolution:
   Evaluates "What answer is this challenge actually asking for?" against the challenge's
   semantic question/task context (flag string, username, hash, number, file, key, text),
   filtering placeholders and verifying type alignment.
3. Answer Verification:
   Weighs evidence quality, provenance, and source confidence (e.g., tool_output,
   vision_read on a verified reconstructed artifact, deterministic decode) to promote
   candidates to RESOLVED / VERIFIED, or preserve them as unverified CANDIDATEs without
   losing their provenance.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("forge.verifier")


# Strict flag regex — known CTF platform prefixes only, ≥4 chars, braces excluded
# from the body. No generic catch-all → no CSS/LaTeX/JSON FPs.
FLAG_REGEX = re.compile(
    r"(?:picoCTF\{[^{}]{4,}\}|FLAG\{[^{}]{4,}\}|flag\{[^{}]{4,}\}|HTB\{[^{}]{4,}\}|CTF\{[^{}]{4,}\}|"
    r"DUCTF\{[^{}]{4,}\}|corctf\{[^{}]{4,}\}|TFCCTF\{[^{}]{4,}\}|pwn\.college\{[^{}]{4,}\}|THM\{[^{}]{4,}\})",
    re.IGNORECASE,
)

# Shapes that look like a flag but are placeholders/examples from model prose.
FALSE_FLAG_PATTERNS = re.compile(
    r"(?:picoCTF\{\.\.\.\}|FLAG\{\.\.\.\}|HTB\{\.\.\.\}|CTF\{\.\.\.\}|"
    r"\{[a-z_]+_here\}|\{example[^}]*\}|\{your[^}]*\}|\{placeholder[^}]*\}|"
    r"\{some[^}]*\}|\{flag[^}]*format[^}]*\}|\{insert[^}]*\}|\{\.\.\.\}|"
    # Bare single-word placeholder bodies — the FULL content between braces is one
    # of these common dummy words.  picoCTF{flag} / FLAG{value} / HTB{todo} etc.
    # were previously not caught because the deny-list only covered _here/example/…
    r"\{(?:flag|value|answer|x|todo|redacted|tbd)\})",
    re.IGNORECASE,
)

# Signals that a "candidate value" is actually a fragment of SOURCE CODE / an
# extraction *expression* (e.g. `{resp.text[resp.text.find('picoCTF{')...]}`) rather
# than a literal answer captured from evidence. A real flag/answer read out of tool
# output never contains quote characters, call/index syntax, or an ellipsis — those
# only appear when the model emitted CODE describing how to find the answer instead
# of the answer itself. Conservative on purpose: real flag/hash/username/number/URL
# bodies use none of these, so this cannot reject a genuine captured answer.
SOURCE_CODE_SIGNALS = re.compile(r"""['"]|\.\.\.|[\[\]()]|\.\w+\s*\(""")

# Common hash patterns
MD5_REGEX = re.compile(r"\b[a-f0-9]{32}\b", re.IGNORECASE)
SHA1_REGEX = re.compile(r"\b[a-f0-9]{40}\b", re.IGNORECASE)
SHA256_REGEX = re.compile(r"\b[a-f0-9]{64}\b", re.IGNORECASE)


class AnswerType(str, Enum):
    FLAG = "flag"
    HASH = "hash"
    USERNAME = "username"
    NUMBER = "number"
    FILENAME = "filename"
    KEY = "key"
    STRING = "string"
    CUSTOM = "custom"


class AnswerStatus(str, Enum):
    VERIFIED = "FLAG_VERIFIED"
    RESOLVED = "RESOLVED"
    CANDIDATE = "FLAG_CANDIDATE"
    REJECTED = "REJECTED"


# Backward compatibility aliases
FlagStatus = AnswerStatus


class AnswerSource(str, Enum):
    TOOL_OUTPUT = "tool_output"                    # direct command execution stdout/stderr
    VISION_READ = "vision_read"                    # model-powered visual extraction from image/artifact
    RECONSTRUCTED_ARTIFACT = "reconstructed_artifact"  # extracted from reconstructed file/bytes
    RECONSTRUCTED_ARTIFACT_OCR = "reconstructed_artifact_ocr"  # OCR of reconstructed artifact
    DECODED_ARTIFACT = "decoded_artifact"          # deterministic decode of hidden text/stream
    LLM_PROSE = "llm_prose"                        # asserted by model in prose, not observed in evidence
    UNKNOWN = "unknown"


# Backward compatibility alias
FlagSource = AnswerSource


@dataclass
class AnswerCandidate:
    """Represents an answer candidate with full evidence, provenance, and task context."""
    value: str
    answer_type: AnswerType = AnswerType.FLAG
    source: AnswerSource = AnswerSource.UNKNOWN
    confidence: float = 0.5
    worker_id: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    task_context: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    status: AnswerStatus = AnswerStatus.CANDIDATE
    reasons: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "answer_type": self.answer_type.value,
            "source": self.source.value if isinstance(self.source, Enum) else str(self.source),
            "confidence": self.confidence,
            "worker_id": self.worker_id,
            "evidence": self.evidence,
            "task_context": self.task_context,
            "provenance": self.provenance,
            "status": self.status.value if isinstance(self.status, Enum) else str(self.status),
            "reasons": self.reasons,
            "created_at": self.created_at,
        }


@dataclass
class AnswerVerdict:
    """Verdict returned by AnswerResolver / FlagVerifier."""
    status: AnswerStatus
    confidence: float
    candidate: str
    reasons: List[str] = field(default_factory=list)
    answer_type: AnswerType = AnswerType.FLAG
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_verified(self) -> bool:
        return self.status == AnswerStatus.VERIFIED

    @property
    def is_resolved(self) -> bool:
        return self.status in (AnswerStatus.VERIFIED, AnswerStatus.RESOLVED)

    @property
    def is_confirmed(self) -> bool:
        return self.is_resolved and self.confidence >= 0.7


# Backward compatibility alias
FlagVerdict = AnswerVerdict


def infer_expected_answer_type(
    question: str = "",
    description: str = "",
    name: str = "",
    category: str = "",
    flag_pattern: str = ""
) -> Tuple[AnswerType, Optional[str]]:
    """Determine what kind of answer the challenge is actually asking for.

    Returns (expected_type, optional_expected_format_or_pattern).
    """
    combined = f"{question} {description} {name}".lower()

    # Explicit flag pattern requested
    if flag_pattern and ("{" in flag_pattern or "flag" in flag_pattern.lower()):
        return AnswerType.FLAG, flag_pattern

    # Explicit question analysis
    if re.search(r"\b(?:username|user name|who is the user|login name|account name|user)\b", combined) and "flag" not in combined:
        return AnswerType.USERNAME, None

    if re.search(r"\b(?:sha256|sha1|md5|sha-256|sha-1|hash|checksum)\b", combined) and "flag" not in combined:
        return AnswerType.HASH, None

    if re.search(r"\b(?:port|port number|port\b|number of|how many|integer|count|total|pid)\b", combined) and "flag" not in combined:
        return AnswerType.NUMBER, None

    if re.search(r"\b(?:filename|file name|file path|path to the file|which file)\b", combined) and "flag" not in combined:
        return AnswerType.FILENAME, None

    if re.search(r"\b(?:secret key|api key|encryption key|access token|auth key|secret token|passphrase|password)\b", combined) and "flag" not in combined:
        return AnswerType.KEY, None

    # CTF default: flags
    cat_lower = (category or "").lower()
    if cat_lower in ["web", "forensics", "crypto", "rev", "reverse", "pwn", "misc", "osint"] or "flag" in combined:
        return AnswerType.FLAG, flag_pattern or None

    return AnswerType.STRING, None



class AnswerResolver:
    """Generic resolver and verifier for answer candidates across all challenge types."""

    def looks_like_flag(self, text: str) -> bool:
        return bool(text) and bool(FLAG_REGEX.search(text)) and not FALSE_FLAG_PATTERNS.search(text)

    def looks_like_source_code(self, value: str) -> bool:
        """True when *value* is a source-code / extraction expression, not a captured
        literal answer.

        Two checks:
        1. For a flag envelope the BODY between the braces is inspected for code
           syntax signals (the envelope's own braces are legitimate).
        2. Any non-whitespace characters immediately before or after the matched
           envelope in the raw candidate string are treated as an interpolation
           signal — e.g. a trailing ``")`` from a Python f-string, or a leading
           ``f"`` prefix, are invisible to the body check but clearly not a captured
           literal flag.
        """
        if not value:
            return False
        subject = value.strip()
        m = FLAG_REGEX.search(subject)
        if m:
            inner = m.group(0)
            brace = inner.find("{")
            if brace != -1 and inner.rstrip().endswith("}"):
                subject = inner[brace + 1:inner.rstrip().rfind("}")]
            # Check for non-whitespace context surrounding the envelope — anything
            # outside a clean `prefix{...}` structure signals string interpolation.
            pre = value[:m.start()].strip()
            post = value[m.end():].strip()
            if pre or post:
                return True
        return bool(SOURCE_CODE_SIGNALS.search(subject))

    def normalize_source(self, source: Any) -> AnswerSource:
        if isinstance(source, AnswerSource):
            return source
        if not source:
            return AnswerSource.UNKNOWN
        s = str(source).lower()
        if s in ("tool_output", "tool", "stdout", "command", "step_execution"):
            return AnswerSource.TOOL_OUTPUT
        if s in ("vision_read", "vision", "gemini_vision"):
            return AnswerSource.VISION_READ
        if s in ("reconstructed_artifact", "artifact", "reconstruction"):
            return AnswerSource.RECONSTRUCTED_ARTIFACT
        if s in ("reconstructed_artifact_ocr", "ocr"):
            return AnswerSource.RECONSTRUCTED_ARTIFACT_OCR
        if s in ("decoded_artifact", "decode"):
            return AnswerSource.DECODED_ARTIFACT
        if s in ("llm", "llm_prose", "llm_reported", "prose", "model"):
            return AnswerSource.LLM_PROSE
        return AnswerSource.UNKNOWN

    def extract_candidates(
        self,
        text: str,
        task_context: Optional[Dict[str, Any]] = None,
        source: AnswerSource | str = AnswerSource.TOOL_OUTPUT,
    ) -> List[AnswerCandidate]:
        """Extract candidate answers from text/output based on flags, explicit annotations, and challenge semantics."""
        if not text:
            return []

        task_context = task_context or {}
        candidates: List[AnswerCandidate] = []
        seen_values = set()
        src = self.normalize_source(source)

        def _add(val: str, atype: AnswerType, conf: float = 0.8, reasons: Optional[List[str]] = None):
            v = val.strip()
            if not v or v in seen_values:
                return
            if FALSE_FLAG_PATTERNS.search(v):
                return
            if self.looks_like_source_code(v):
                return
            seen_values.add(v)
            candidates.append(AnswerCandidate(
                value=v,
                answer_type=atype,
                source=src,
                confidence=conf,
                task_context=task_context,
                reasons=reasons or [],
            ))

        # 1. Flag candidates via FLAG_REGEX (high-value extractor)
        for m in FLAG_REGEX.finditer(text):
            flag_val = m.group(0).strip()
            _add(flag_val, AnswerType.FLAG, conf=0.9, reasons=["Matched standard CTF flag regex."])

        # 2. Explicit ANSWER / FLAG / KEY / SECRET prefix claims in text
        for m in re.finditer(r"\b(?:FLAG|ANSWER|SECRET|KEY|SOLUTION)\s*[:=]\s*['\"]?([^\s'\"]+)['\"]?", text, re.IGNORECASE):
            claim = m.group(1).strip()
            _add(claim, AnswerType.CUSTOM, conf=0.8, reasons=["Explicitly prefixed in evidence text."])

        # 3. Context-driven extraction based on expected challenge answer type
        expected_type, custom_pat = infer_expected_answer_type(
            question=task_context.get("description", "") or task_context.get("question", ""),
            description=task_context.get("description", ""),
            name=task_context.get("challenge_name", ""),
            category=task_context.get("category", ""),
            flag_pattern=task_context.get("flag_pattern", ""),
        )

        if custom_pat:
            try:
                pat = re.compile(custom_pat.replace("{...}", r"\{[^\}]+\}"), re.IGNORECASE)
                for m in pat.finditer(text):
                    _add(m.group(0).strip(), expected_type, conf=0.85, reasons=["Matched custom challenge flag pattern."])
            except Exception:
                pass

        if expected_type == AnswerType.HASH:
            for m in SHA256_REGEX.finditer(text):
                _add(m.group(0).lower(), AnswerType.HASH, conf=0.85, reasons=["Matched SHA256 hash format."])
            for m in SHA1_REGEX.finditer(text):
                _add(m.group(0).lower(), AnswerType.HASH, conf=0.8, reasons=["Matched SHA1 hash format."])
            for m in MD5_REGEX.finditer(text):
                _add(m.group(0).lower(), AnswerType.HASH, conf=0.75, reasons=["Matched MD5 hash format."])

        elif expected_type == AnswerType.NUMBER:
            for m in re.finditer(r"\b(?:port|port\s+number|number|id|count|value)\s*[:=]?\s*(\d{1,8})\b", text, re.IGNORECASE):
                _add(m.group(1), AnswerType.NUMBER, conf=0.85, reasons=["Matched labeled numeric value."])
            for line in text.splitlines():
                line_str = line.strip()
                if line_str.isdigit() and len(line_str) <= 10:
                    _add(line_str, AnswerType.NUMBER, conf=0.75, reasons=["Isolated line containing number."])

        elif expected_type == AnswerType.USERNAME:
            for m in re.finditer(r"\b(?:user(?:name)?|login|account|admin|operator)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{3,32})['\"]?", text, re.IGNORECASE):
                cand_u = m.group(1).strip()
                if cand_u.lower() not in ["not", "the", "found", "error", "true", "false", "null", "undefined", "successful", "failed", "access"]:
                    _add(cand_u, AnswerType.USERNAME, conf=0.85, reasons=["Discovered username pattern in evidence."])



        elif expected_type == AnswerType.FILENAME:
            for m in re.finditer(r"\b([A-Za-z0-9_\-/\\]+\.[A-Za-z0-9]{1,6})\b", text):
                _add(m.group(1), AnswerType.FILENAME, conf=0.75, reasons=["Discovered filename path pattern."])

        elif expected_type == AnswerType.KEY:
            for m in re.finditer(r"\b(?:key|api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{8,64})['\"]?", text, re.IGNORECASE):
                _add(m.group(1), AnswerType.KEY, conf=0.85, reasons=["Discovered secret/key pattern."])

        return candidates

    def resolve(self, candidate_obj: AnswerCandidate) -> AnswerVerdict:
        """Resolve a full AnswerCandidate against its task_context and evidence."""
        task_ctx = candidate_obj.task_context or {}
        return self.assess(
            candidate_obj.value,
            source=candidate_obj.source,
            command=candidate_obj.provenance.get("command", ""),
            action_succeeded=candidate_obj.provenance.get("action_succeeded", True),
            target_scope=task_ctx.get("target_scope", ""),
            expected_format=task_ctx.get("flag_pattern", ""),
            description=task_ctx.get("description", "") or task_ctx.get("question", ""),
            challenge_name=task_ctx.get("challenge_name", ""),
            category=task_ctx.get("category", ""),
            evidence=candidate_obj.evidence,
            worker_id=candidate_obj.worker_id,
        )

    def assess(
        self,
        candidate: str,
        *,
        source: AnswerSource | FlagSource | str = AnswerSource.UNKNOWN,
        command: str = "",
        action_succeeded: bool = True,
        target_scope: str = "",
        expected_format: str = "",
        description: str = "",
        challenge_name: str = "",
        category: str = "",
        evidence: Optional[Dict[str, Any]] = None,
        worker_id: str = "",
        authoritative: bool = False,
    ) -> AnswerVerdict:
        """Assess whether a candidate value resolves the challenge."""
        candidate = (candidate or "").strip()
        reasons: List[str] = []
        evidence = evidence or {}
        src = self.normalize_source(source)

        if not candidate:
            return AnswerVerdict(AnswerStatus.REJECTED, 0.0, candidate,
                                 ["Candidate is empty."], AnswerType.CUSTOM, evidence)

        # ── Echoed / constructed gate ──
        # If the exact value already appears in the command the agent issued, it
        # was generated/echoed by the agent, not captured from the target.
        if command and candidate in command and src in (AnswerSource.LLM_PROSE, AnswerSource.UNKNOWN):
            return AnswerVerdict(AnswerStatus.REJECTED, 0.1, candidate,
                                 ["Value appears in the issued command itself (echoed/constructed, not captured)."],
                                 AnswerType.CUSTOM, evidence)

        # ── Placeholder / False-Positive Gate ──
        if FALSE_FLAG_PATTERNS.search(candidate):
            return AnswerVerdict(AnswerStatus.REJECTED, 0.0, candidate,
                                 ["Matches a placeholder/example shape, not a real answer."],
                                 AnswerType.CUSTOM, evidence)

        # ── Source-Code / Extraction-Expression Gate ──
        # A candidate that is really a code fragment describing HOW to extract the
        # answer (e.g. `picoCTF{'):resp.text.find('}`) is never a captured answer,
        # regardless of source. Reject it before any evidence-based promotion so a
        # source-code artifact can never be recorded as RESOLVED/VERIFIED.
        if self.looks_like_source_code(candidate):
            return AnswerVerdict(AnswerStatus.REJECTED, 0.0, candidate,
                                 ["Value is a source-code/extraction expression, not a literal captured answer."],
                                 AnswerType.CUSTOM, evidence)

        # ── Infer Expected Answer Type & Semantic Requirements ──
        expected_type, custom_pattern = infer_expected_answer_type(
            question=description,
            description=description,
            name=challenge_name,
            category=category,
            flag_pattern=expected_format,
        )

        resolved_type = expected_type

        # ── Semantic Type Validation ──
        if expected_type == AnswerType.USERNAME:
            # If challenge specifically asks for a username, reject flag envelopes like picoCTF{...}
            if FLAG_REGEX.search(candidate):
                return AnswerVerdict(
                    AnswerStatus.REJECTED, 0.1, candidate,
                    ["Challenge asks for a username, but candidate is formatted as a CTF flag."],
                    AnswerType.FLAG, evidence
                )
            # Username should be an identifier
            if len(candidate) > 64 or " " in candidate:
                return AnswerVerdict(
                    AnswerStatus.CANDIDATE, 0.3, candidate,
                    ["Candidate format is uncertain for a username."],
                    AnswerType.USERNAME, evidence
                )

        elif expected_type == AnswerType.HASH:
            clean_hash = candidate.strip().lower()
            if FLAG_REGEX.search(candidate):
                return AnswerVerdict(
                    AnswerStatus.REJECTED, 0.1, candidate,
                    ["Challenge asks for a hash, but candidate is formatted as a CTF flag."],
                    AnswerType.FLAG, evidence
                )
            if not (MD5_REGEX.fullmatch(clean_hash) or SHA1_REGEX.fullmatch(clean_hash) or SHA256_REGEX.fullmatch(clean_hash)):
                return AnswerVerdict(
                    AnswerStatus.CANDIDATE, 0.3, candidate,
                    ["Candidate does not match known hash length/hex format."],
                    AnswerType.HASH, evidence
                )
            candidate = clean_hash

        elif expected_type == AnswerType.NUMBER:
            if FLAG_REGEX.search(candidate):
                return AnswerVerdict(
                    AnswerStatus.REJECTED, 0.1, candidate,
                    ["Challenge asks for a number, but candidate is formatted as a CTF flag."],
                    AnswerType.FLAG, evidence
                )
            if not candidate.isdigit():
                return AnswerVerdict(
                    AnswerStatus.CANDIDATE, 0.2, candidate,
                    ["Challenge asks for a numeric answer, but candidate is non-numeric."],
                    AnswerType.NUMBER, evidence
                )

        elif expected_type == AnswerType.FLAG:
            # If standard flag regex matches, extract canonical matched substring
            m = FLAG_REGEX.search(candidate)
            if m:
                candidate = m.group(0)
            elif custom_pattern:
                try:
                    pat = re.compile(custom_pattern.replace("{...}", r"\{[^\}]+\}"), re.IGNORECASE)
                    m2 = pat.search(candidate)
                    if m2:
                        candidate = m2.group(0)
                except Exception:
                    pass

        # ── Evidence Quality & Source Weighting ──
        is_high_confidence_source = src in (
            AnswerSource.TOOL_OUTPUT,
            AnswerSource.VISION_READ,
            AnswerSource.RECONSTRUCTED_ARTIFACT,
            AnswerSource.RECONSTRUCTED_ARTIFACT_OCR,
            AnswerSource.DECODED_ARTIFACT,
        )

        if is_high_confidence_source:
            if not action_succeeded:
                reasons.append("From execution evidence, but action reported failure — needs corroboration.")
                return AnswerVerdict(AnswerStatus.CANDIDATE, 0.55, candidate, reasons, resolved_type, evidence)

            reasons.append(f"Directly derived from verified execution evidence ({src.value}).")

            # Soft format corroboration if explicit pattern provided
            if expected_format and "{" in candidate:
                prefix = candidate.split("{", 1)[0].lower()
                if prefix and prefix not in expected_format.lower() and "{...}" not in expected_format:
                    reasons.append(f"Prefix '{prefix}' differs from expected format '{expected_format}'.")

            conf = 0.95 if not any(r.startswith("Prefix") for r in reasons) else 0.85

            # Meaningful separation: Only authoritatively verified submissions become VERIFIED.
            # Evidence-based solutions are RESOLVED.
            if authoritative:
                reasons.append("Authoritatively confirmed via submission/check mechanism.")
                return AnswerVerdict(AnswerStatus.VERIFIED, 1.0, candidate, reasons, resolved_type, evidence)

            return AnswerVerdict(AnswerStatus.RESOLVED, conf, candidate, reasons, resolved_type, evidence)

        if src == AnswerSource.LLM_PROSE:
            reasons.append("Asserted in model prose; not verified against direct tool/artifact evidence.")
            return AnswerVerdict(AnswerStatus.CANDIDATE, 0.35, candidate, reasons, resolved_type, evidence)

        reasons.append("Source of candidate is unknown — treated as unverified candidate.")
        return AnswerVerdict(AnswerStatus.CANDIDATE, 0.40, candidate, reasons, resolved_type, evidence)

    def assess_observation(
        self,
        obs: Any,
        *,
        command: str = "",
        action_succeeded: bool = True,
        target_scope: str = "",
        expected_format: str = "",
        description: str = "",
        challenge_name: str = "",
        category: str = "",
        authoritative: bool = False,
    ) -> Optional[AnswerVerdict]:
        """Verify the first answer/flag candidate found on a structured Observation."""
        candidates = getattr(obs, "flag_candidates", None) or getattr(obs, "answer_candidates", None) or []
        if not candidates:
            return None
        cand_val = candidates[0].value if isinstance(candidates[0], AnswerCandidate) else str(candidates[0])
        return self.assess(
            cand_val,
            source=AnswerSource.TOOL_OUTPUT,
            command=command,
            action_succeeded=action_succeeded,
            target_scope=target_scope,
            expected_format=expected_format,
            description=description,
            challenge_name=challenge_name,
            category=category,
            authoritative=authoritative,
        )


VERIFIER_SYSTEM_PROMPT = """You are FORGE's Verifier Agent (Agent #4).
Your duty is to independently, critically, and objectively evaluate answer candidates against challenge requirements, tool execution evidence, and semantic context.

You must determine whether the candidate value accurately and completely answers the challenge question, or if it is a distractor, placeholder, format mismatch, malformed string, or hallucinated assertion.

Output MUST be a valid JSON object with the following structure:
{
  "verdict": "RESOLVE" | "REJECT" | "NEEDS_MORE_EVIDENCE",
  "confidence": <float between 0.0 and 1.0>,
  "answer_type": "flag" | "hash" | "username" | "number" | "filename" | "key" | "string" | "custom",
  "is_distractor": <boolean>,
  "reasoning": "<concise explanation of decision based on evidence and question intent>",
  "evidence_soundness": "<assessment of tool output, command provenance, or observation>"
}

Strict Verification Rules:
1. If the challenge question specifically asks for a non-flag answer (e.g., username, hash, filename, number, port, key) and the candidate is a flag format (e.g., picoCTF{...}) or irrelevant artifact, mark verdict as "REJECT" and is_distractor as true.
2. If the candidate comes from model prose without verifiable tool command observation, confidence must NOT exceed 0.5.
3. If the candidate contains placeholders (e.g. picoCTF{...}, {flag_here}, {insert_flag_here}), mark verdict as "REJECT".
4. Only mark "RESOLVE" if the candidate directly satisfies the challenge question and is backed by verified execution evidence.
5. If evidence is ambiguous or incomplete, mark verdict as "NEEDS_MORE_EVIDENCE".
"""


class VerifierAgent:
    """Dedicated 4th Verifier Agent in FORGE's agent architecture.

    Independently evaluates:
    - the challenge question / task
    - expected answer semantics & type
    - candidate value
    - candidate source
    - supporting evidence
    - provenance
    - confidence
    - whether the evidence actually answers what the challenge is asking for.

    Trust Boundary:
    - The LLM output is UNTRUSTED.
    - Deterministic FORGE code validates the structured response before applying any verdict.
    - LLM evaluation CANNOT directly declare VERIFIED (only authoritative confirmation can).
    - Discovery agents do not control the verifier; it is invoked by the orchestration layer.
    - Verifier does not mutate swarm state; only authoritative orchestration mutates state.
    """

    def __init__(self, resolver: Optional[AnswerResolver] = None, router: Optional[Any] = None):
        self.resolver = resolver or AnswerResolver()
        self.router = router

    def _extract_candidate_data(
        self,
        candidate: AnswerCandidate | str,
        *,
        task_context: Optional[Dict[str, Any]] = None,
        source: AnswerSource | str = AnswerSource.UNKNOWN,
        evidence: Optional[Dict[str, Any]] = None,
        command: str = "",
        action_succeeded: bool = True,
    ) -> Tuple[str, AnswerSource, str, bool, Dict[str, Any], Dict[str, Any], str]:
        if isinstance(candidate, AnswerCandidate):
            cand_obj = candidate
            tctx = {**(cand_obj.task_context or {}), **(task_context or {})}
            ev = {**(cand_obj.evidence or {}), **(evidence or {})}
            val = cand_obj.value
            src = self.resolver.normalize_source(cand_obj.source)
            cmd = cand_obj.provenance.get("command", command)
            succ = cand_obj.provenance.get("action_succeeded", action_succeeded)
            worker_id = cand_obj.worker_id
        elif isinstance(candidate, AnswerVerdict):
            ver_obj = candidate
            tctx = dict(task_context or {})
            ev = {**(ver_obj.evidence or {}), **(evidence or {})}
            val = ver_obj.candidate
            src = self.resolver.normalize_source(getattr(ver_obj, "source", source if source != AnswerSource.UNKNOWN else AnswerSource.TOOL_OUTPUT))
            cmd = command
            succ = action_succeeded
            worker_id = ""
        else:
            val = str(candidate)
            src = self.resolver.normalize_source(source)
            cmd = command
            succ = action_succeeded
            ev = evidence or {}
            tctx = task_context or {}
            worker_id = ""
        return val, src, cmd, succ, ev, tctx, worker_id

    def _apply_deterministic_distractor_gate(
        self,
        val: str,
        tctx: Dict[str, Any],
        verdict: AnswerVerdict,
        evidence: Dict[str, Any],
    ) -> AnswerVerdict:
        """Apply deterministic distractor and question mismatch checks."""
        question = (tctx.get("description", "") or tctx.get("question", "")).lower()

        # Check if candidate is a flag-shaped distractor that does NOT answer the challenge question
        if verdict.answer_type in (AnswerType.USERNAME, AnswerType.NUMBER, AnswerType.HASH, AnswerType.FILENAME, AnswerType.KEY):
            if FLAG_REGEX.search(val):
                return AnswerVerdict(
                    AnswerStatus.REJECTED,
                    0.05,
                    val,
                    [f"Candidate is formatted as a flag, but challenge specifically asks for a {verdict.answer_type.value}."],
                    verdict.answer_type,
                    evidence,
                )

        # Check for distractor flag strings when challenge question warns about fakes
        if "not the flag" in question or "fake" in question or "distractor" in question:
            if "fake" in val.lower() or "distractor" in val.lower() or "not_the_flag" in val.lower():
                verdict.status = AnswerStatus.REJECTED
                verdict.reasons.append("Identified as a challenge distractor.")
                verdict.confidence = 0.1

        return verdict

    def verify_sync(
        self,
        candidate: AnswerCandidate | str,
        *,
        task_context: Optional[Dict[str, Any]] = None,
        source: AnswerSource | str = AnswerSource.UNKNOWN,
        evidence: Optional[Dict[str, Any]] = None,
        command: str = "",
        action_succeeded: bool = True,
        authoritative: bool = False,
    ) -> AnswerVerdict:
        """Synchronously verify an answer candidate against challenge semantics and evidence."""
        val, src, cmd, succ, ev, tctx, worker_id = self._extract_candidate_data(
            candidate,
            task_context=task_context,
            source=source,
            evidence=evidence,
            command=command,
            action_succeeded=action_succeeded,
        )

        # 1. Evaluate via AnswerResolver
        verdict = self.resolver.assess(
            val,
            source=src,
            command=cmd,
            action_succeeded=succ,
            target_scope=tctx.get("target_scope", ""),
            expected_format=tctx.get("flag_pattern", ""),
            description=tctx.get("description", "") or tctx.get("question", ""),
            challenge_name=tctx.get("challenge_name", ""),
            category=tctx.get("category", ""),
            evidence=ev,
            worker_id=worker_id,
            authoritative=authoritative,
        )

        # 2. Semantic Distractor & Intent Audit
        verdict = self._apply_deterministic_distractor_gate(val, tctx, verdict, ev)
        return verdict

    async def verify(
        self,
        candidate: AnswerCandidate | str,
        *,
        task_context: Optional[Dict[str, Any]] = None,
        source: AnswerSource | str = AnswerSource.UNKNOWN,
        evidence: Optional[Dict[str, Any]] = None,
        command: str = "",
        action_succeeded: bool = True,
        authoritative: bool = False,
    ) -> AnswerVerdict:
        """Asynchronously verify a candidate answer with deterministic gate and untrusted LLM reasoning."""
        val, src, cmd, succ, ev, tctx, worker_id = self._extract_candidate_data(
            candidate,
            task_context=task_context,
            source=source,
            evidence=evidence,
            command=command,
            action_succeeded=action_succeeded,
        )

        # 1. Primary deterministic assessment
        deterministic_verdict = self.resolver.assess(
            val,
            source=src,
            command=cmd,
            action_succeeded=succ,
            target_scope=tctx.get("target_scope", ""),
            expected_format=tctx.get("flag_pattern", ""),
            description=tctx.get("description", "") or tctx.get("question", ""),
            challenge_name=tctx.get("challenge_name", ""),
            category=tctx.get("category", ""),
            evidence=ev,
            worker_id=worker_id,
            authoritative=authoritative,
        )

        # Apply deterministic distractor gate
        verdict = self._apply_deterministic_distractor_gate(val, tctx, deterministic_verdict, ev)

        # If authoritative or already rejected for structural reasons, return immediately
        if authoritative or verdict.status == AnswerStatus.REJECTED:
            return verdict

        # 2. If ModelRouter / ProviderGateway is available, execute LLM-assisted verification (Agent #4)
        if self.router is not None:
            try:
                llm_verdict = await self._llm_evaluate(
                    val=val,
                    src=src,
                    cmd=cmd,
                    succ=succ,
                    ev=ev,
                    tctx=tctx,
                    deterministic_verdict=verdict,
                    authoritative=authoritative,
                )
                if llm_verdict is not None:
                    return llm_verdict
            except Exception as e:
                logger.warning(f"[VerifierAgent] LLM verification failed ({e}); falling back to deterministic verdict.")

        return verdict

    async def _llm_evaluate(
        self,
        *,
        val: str,
        src: AnswerSource,
        cmd: str,
        succ: bool,
        ev: Dict[str, Any],
        tctx: Dict[str, Any],
        deterministic_verdict: AnswerVerdict,
        authoritative: bool,
    ) -> Optional[AnswerVerdict]:
        """Invoke ModelRouter / ProviderGateway with capability='verification' and validate structured response."""
        prompt = (
            f"Challenge Information:\n"
            f"- Name: {tctx.get('challenge_name', 'Unknown')}\n"
            f"- Category: {tctx.get('category', 'Unknown')}\n"
            f"- Description / Question: {tctx.get('description', '') or tctx.get('question', 'None')}\n"
            f"- Expected Format / Pattern: {tctx.get('flag_pattern', 'None')}\n\n"
            f"Candidate Evaluation:\n"
            f"- Candidate Value: {val}\n"
            f"- Candidate Source: {src.value}\n"
            f"- Tool Command: {cmd}\n"
            f"- Command Succeeded: {succ}\n"
            f"- Evidence Details: {json.dumps(ev, default=str)[:500]}\n"
            f"- Deterministic Preliminary Status: {deterministic_verdict.status.value}\n"
            f"- Deterministic Inferred Type: {deterministic_verdict.answer_type.value}\n"
            f"- Deterministic Confidence: {deterministic_verdict.confidence}\n"
        )

        resp = None
        if hasattr(self.router, "route_request"):
            resp = await self.router.route_request(
                prompt=prompt,
                capability="verification",
                system_instruction=VERIFIER_SYSTEM_PROMPT,
            )
        elif hasattr(self.router, "complete"):
            resp = await self.router.complete(
                prompt=prompt,
                system_instruction=VERIFIER_SYSTEM_PROMPT,
                capability="verification",
            )
        else:
            return None

        if resp is None or getattr(resp, "is_refusal", False) is True:
            return None

        content = getattr(resp, "content", "")
        if not isinstance(content, str) or not content.strip():
            return None

        return self._validate_llm_decision(
            llm_text=content,
            val=val,
            src=src,
            ev=ev,
            tctx=tctx,
            deterministic_verdict=deterministic_verdict,
            authoritative=authoritative,
        )

    def _validate_llm_decision(
        self,
        llm_text: str,
        val: str,
        src: AnswerSource,
        ev: Dict[str, Any],
        tctx: Dict[str, Any],
        deterministic_verdict: AnswerVerdict,
        authoritative: bool,
    ) -> Optional[AnswerVerdict]:
        """Deterministically validate untrusted LLM structured output and enforce trust boundaries."""
        if not llm_text:
            return None

        # Extract JSON from response
        try:
            cleaned = llm_text.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
            data = json.loads(cleaned)
        except Exception as e:
            logger.debug(f"[VerifierAgent] Failed to parse LLM JSON: {e}")
            return None

        if not isinstance(data, dict):
            return None

        verdict_str = str(data.get("verdict", "")).strip().upper()
        try:
            llm_conf = float(data.get("confidence", deterministic_verdict.confidence))
            llm_conf = max(0.0, min(1.0, llm_conf))
        except (ValueError, TypeError):
            llm_conf = deterministic_verdict.confidence

        is_distractor = bool(data.get("is_distractor", False))
        reasoning = str(data.get("reasoning", "")).strip()
        ans_type_str = str(data.get("answer_type", deterministic_verdict.answer_type.value)).strip().lower()

        try:
            resolved_type = AnswerType(ans_type_str)
        except ValueError:
            resolved_type = deterministic_verdict.answer_type

        reasons = list(deterministic_verdict.reasons)
        if reasoning:
            reasons.append(f"Agent #4 Verifier: {reasoning}")

        # Enforce Trust Boundary Invariants:
        # 1. Distractor detection / REJECT
        if is_distractor or verdict_str == "REJECT":
            return AnswerVerdict(
                status=AnswerStatus.REJECTED,
                confidence=min(llm_conf, 0.1),
                candidate=val,
                reasons=reasons,
                answer_type=resolved_type,
                evidence=ev,
            )

        # 2. LLM prose can NEVER be elevated to RESOLVED or VERIFIED
        if src == AnswerSource.LLM_PROSE:
            return AnswerVerdict(
                status=AnswerStatus.CANDIDATE,
                confidence=min(llm_conf, 0.5),
                candidate=val,
                reasons=reasons,
                answer_type=resolved_type,
                evidence=ev,
            )

        # 3. Only authoritative external checks produce VERIFIED status
        if authoritative:
            return AnswerVerdict(
                status=AnswerStatus.VERIFIED,
                confidence=1.0,
                candidate=val,
                reasons=reasons,
                answer_type=resolved_type,
                evidence=ev,
            )

        # 4. Validated RESOLVED or CANDIDATE
        if verdict_str == "RESOLVE" and llm_conf >= 0.7:
            final_status = AnswerStatus.RESOLVED
            final_conf = llm_conf
        else:
            final_status = AnswerStatus.CANDIDATE
            final_conf = min(llm_conf, 0.6)
            if verdict_str == "NEEDS_MORE_EVIDENCE":
                reasons.append("Agent #4: Needs more corroborating evidence.")

        return AnswerVerdict(
            status=final_status,
            confidence=final_conf,
            candidate=val,
            reasons=reasons,
            answer_type=resolved_type,
            evidence=ev,
        )


# Backward compatibility class
FlagVerifier = AnswerResolver

