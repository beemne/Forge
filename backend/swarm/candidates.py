"""Phase 5 §9/§11/§16/§18-21 — candidate next-action generation.

Given everything currently known (the shared mission state, fresh evidence,
relevant historical experience, and applicable playbooks), produce a set of
structured :class:`~backend.swarm.reasoning.CandidateAction` proposals. The
:class:`~backend.swarm.scoring.ActionScorer` then ranks them and the supervisor /
coordinator turns the best into a schedulable task.

Four complementary sources feed the generator — deterministic, so it runs with no
LLM in tests:

  1. **State-gap actions (§11)** — from what is *unknown*: no services → fingerprint;
     no surface → enumerate; a known vuln not yet exploited → exploit; an artifact →
     analyse/decode. This is the reusable information-gain mechanism.
  2. **Evidence actions (§16)** — an ELF artifact justifies file/strings/checksum
     actions; a versioned service justifies targeted vuln research; etc.
  3. **Memory actions (§18/§19)** — relevant past experience is *advisory*: it
     proposes a candidate carrying its historical confidence/success-rate, but the
     candidate still competes on the current evidence and is never auto-executed.
  4. **Playbook actions (§20/§21)** — an applicable playbook technique is *adapted*
     to the current target/evidence, not blindly executed; if the current target is
     the wrong KIND for the technique, or its capability is unavailable, the
     candidate is dropped or its capability recorded so the pre-dispatch gate handles
     it (reusing Phase 4.x controls).

Memory/playbook retrieval is bounded (§18) and goes exclusively through the EXISTING
``memory_retriever`` singleton — no second memory database is created.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from backend.swarm.reasoning import (
    CandidateAction, profile_for,
)
from backend.swarm.scoring import information_gain_for

logger = logging.getLogger("forge.swarm.candidates")

# Keyword → canonical action_type, used to adapt a free-text technique (from memory,
# a playbook, or evidence prose) onto one of the known ACTION_PROFILES (§20).
_TECHNIQUE_KEYWORDS = {
    "directory_enum": ("director", "gobuster", "ffuf", "dirb", "wordlist", "endpoint enum", "fuzz path"),
    "parameter_discovery": ("parameter", "param fuzz", "arjun", "query string"),
    "http_inspect": ("http header", "response header", "curl", "inspect response", "robots.txt", "sitemap"),
    "auth_test": ("login", "credential", "default password", "auth bypass", "brute"),
    "web_exploit": ("sqli", "sql injection", "xss", "ssti", "idor", "lfi", "rfi", "ssrf",
                    "command injection", "deserialization", "upload bypass", "file upload"),
    "vuln_research": ("cve", "known vulnerabilit", "exploit-db", "version exploit"),
    "service_fingerprint": ("fingerprint", "service version", "banner", "nmap -sv"),
    "port_scan": ("port scan", "nmap", "open port", "masscan"),
    "artifact_analysis": ("strings", "exiftool", "binwalk", "file type", "metadata", "carve", "pcap"),
    "decode": ("base64", "rot13", "hex decode", "decode", "cipher", "xor", "caesar", "encoding"),
    "binary_analysis": ("checksec", "elf", "buffer overflow", "rop", "gdb", "pwntools", "format string"),
    "reverse_analysis": ("ghidra", "objdump", "decompile", "disassemble", "reverse engineer"),
    "ocr_extract": ("ocr", "text from image", "tesseract", "read the text in the image"),
    "interactive_probe": ("interact", "netcat", "nc ", "send input", "pwntools remote", "connect to"),
}


def technique_to_action_type(text: str) -> str:
    """Best-effort map free-text technique/strategy prose onto a known action type."""
    low = (text or "").lower()
    for atype, kws in _TECHNIQUE_KEYWORDS.items():
        if any(kw in low for kw in kws):
            return atype
    return "generic"


class CandidateGenerator:
    """Produces candidate next actions from state, evidence, memory and playbooks."""

    def __init__(self, *, memory_retriever: Any = None, capabilities: Any = None,
                 technique_stats: Any = None):
        # Injected for tests; defaults to the EXISTING singletons in production.
        self._memory = memory_retriever
        self._capabilities = capabilities
        # Phase 6 §9 — optional contextual success statistics. Left None by default so
        # candidate generation is byte-for-byte the Phase-5 behaviour unless a caller
        # (the coordinator in production, or a test) explicitly wires it in; when wired,
        # a technique's historical/contextual success rate refines — never dictates —
        # the candidate's success_probability, and is recorded as observability
        # provenance (§13).
        self._stats = technique_stats

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(
        self,
        mission_state: Any,
        *,
        recent_evidence: Optional[Iterable[Any]] = None,
        use_memory: bool = True,
        max_memory: int = 4,
    ) -> List[CandidateAction]:
        """Generate a de-duplicated list of candidate actions from all sources."""
        out: List[CandidateAction] = []
        out.extend(self._from_state_gaps(mission_state))
        for ev in (recent_evidence or []):
            out.extend(self._from_evidence(ev, mission_state))
        if use_memory:
            out.extend(self._from_memory(mission_state, max_memory=max_memory))
        return self._dedupe(out)

    # ------------------------------------------------------------------ #
    # §11 state-gap → information-seeking actions
    # ------------------------------------------------------------------ #

    def _from_state_gaps(self, ms: Any) -> List[CandidateAction]:
        cands: List[CandidateAction] = []
        target = getattr(ms, "target", "") or ""
        target_type = getattr(ms, "target_type", "") or ""
        category = (getattr(ms, "category", "") or "").lower()

        services = _get(ms, "services", "known_services")
        technologies = list(getattr(ms, "technologies", []) or [])
        endpoints = _get(ms, "endpoints", "known_endpoints")
        vulns = list(getattr(ms, "vulnerabilities", []) or [])
        creds = list(getattr(ms, "credentials", []) or [])
        artifacts = list(getattr(ms, "artifacts", []) or [])

        networked = _looks_networked(target, target_type)

        # No services/tech known on a networked target → fingerprint first (HIGH gain).
        if networked and not services and not technologies:
            cands.append(self._mk("service_fingerprint", ms,
                                   objective=("Fingerprint the target's services and versions to "
                                              "reduce uncertainty about the attack surface."),
                                   rationale="No services or technologies are known yet; "
                                             "fingerprinting has the highest information gain.",
                                   evidence_support=0.4, success_probability=0.7))

        # Networked but surface unknown → directory enumeration.
        if networked and not endpoints:
            cands.append(self._mk("directory_enum", ms,
                                   objective=("Enumerate directories and endpoints to map the "
                                              "application surface."),
                                   rationale="HTTP surface is unknown; enumeration reveals where to look.",
                                   evidence_support=0.4, success_probability=0.6))

        # Discovered endpoints present -> generate web_exploit actions for them
        for ep in endpoints[:3]:
            cands.append(self._mk("web_exploit", ms,
                                   objective=f"Inspect and interact with the discovered endpoint/path '{ep}' to advance exploitation or extract the flag.",
                                   rationale=f"Endpoint '{ep}' is known — probe or exploit it.",
                                   evidence_support=0.75, success_probability=0.6, target=ep))

        # A vulnerability is known but not yet exploited → exploit it (exploitation).
        for v in vulns[:3]:
            cands.append(self._mk("vuln_exploit", ms,
                                   objective=f"Develop and run an exploit for the identified vulnerability: {v}. Extract the flag.",
                                   rationale=f"A concrete vulnerability ({v}) is confirmed — exploiting it is high-value.",
                                   evidence_support=0.85, success_probability=0.55, target=target))

        # Credentials known → use them.
        for c in creds[:2]:
            cands.append(self._mk("auth_test", ms,
                                   objective=f"Authenticate using discovered credentials ({c}) and access protected resources.",
                                   rationale="Credentials are known but not yet used.",
                                   evidence_support=0.8, success_probability=0.6, target=target))

        # Artifacts/files present → analyse/decode/exploit them (forensics/crypto/web surface).
        all_artifacts = list(getattr(ms, "artifacts", []) or [])
        for f in _get(ms, "files", "known_files"):
            if f not in all_artifacts:
                all_artifacts.append(f)

        for a in all_artifacts[:3]:
            low = a.lower()
            if category == "web" or any(k in low for k in (".php", ".phtml", ".jsp", ".asp", ".aspx", "uploads/", "/uploads", ".html", ".js")):
                atype = "web_exploit"
                cands.append(self._mk(atype, ms,
                                       objective=f"Access and inspect/exploit the discovered artifact/path '{a}' for output or flag extraction.",
                                       rationale=f"A discovered artifact/path ({a}) is available to exploit.",
                                       evidence_support=0.8, success_probability=0.65, target=a))
            else:
                atype = "decode" if category == "crypto" else "artifact_analysis"
                cands.append(self._mk(atype, ms,
                                       objective=f"Analyze the artifact '{a}' for embedded data, encodings, or a hidden flag.",
                                       rationale="An unanalysed artifact is a concrete lead.",
                                       evidence_support=0.7, success_probability=0.55, target=a))

        # Category-driven seed when nothing else applies (keeps a bare mission moving).
        if not cands:
            for atype in self._category_actions(category, networked):
                cands.append(self._mk(atype, ms,
                                       objective=profile_for(atype).action_type.replace("_", " ").capitalize()
                                       + " to make initial progress.",
                                       rationale="Initial action for this challenge category.",
                                       evidence_support=0.2, success_probability=profile_for(atype).base_success))
        return cands

    @staticmethod
    def _category_actions(category: str, networked: bool) -> List[str]:
        table = {
            "web": ["http_inspect", "directory_enum"],
            "crypto": ["decode"],
            "forensics": ["artifact_analysis"],
            "pwn": ["binary_analysis"],
            "rev": ["reverse_analysis"],
            "reversing": ["reverse_analysis"],
        }
        if category in table:
            return table[category]
        return ["service_fingerprint"] if networked else ["artifact_analysis"]

    # ------------------------------------------------------------------ #
    # §16 evidence → candidate actions
    # ------------------------------------------------------------------ #

    def _from_evidence(self, ev: Any, ms: Any) -> List[CandidateAction]:
        etype = getattr(ev, "evidence_type", "note")
        title = (getattr(ev, "title", "") or getattr(ev, "description", "") or "").strip()
        cands: List[CandidateAction] = []
        target = getattr(ms, "target", "") or ""
        category = (getattr(ms, "category", "") or "").lower()

        if etype in ("artifact", "file"):
            label = getattr(ev, "artifact_id", None) or getattr(ev, "title", "") or getattr(ev, "description", "") or title
            low = label.lower()
            if any(k in low for k in (".elf", "elf ", "binary", "executable")):
                for atype in ("artifact_analysis", "binary_analysis"):
                    cands.append(self._mk(atype, ms,
                                          objective=f"Run {atype.replace('_', ' ')} on the ELF artifact '{label}'.",
                                          rationale="An ELF binary was discovered — identify its behaviour.",
                                          evidence_support=0.8, success_probability=0.55, target=label))
            elif any(k in low for k in (".jpg", ".jpeg", ".png", ".gif", ".bmp", "image", "jpeg")):
                # An image artifact: analyse it, and — if the objective is to read text
                # embedded in it — recognise OCR as the required next capability (§29
                # Binary Digits). The candidate carries capability="ocr" so the existing
                # Phase 4.x capability gate decides whether it is actually runnable.
                cands.append(self._mk("artifact_analysis", ms,
                                      objective=f"Analyze the image artifact '{label}' for embedded/hidden data.",
                                      rationale="An image artifact was discovered.",
                                      evidence_support=0.7, success_probability=0.55, target=label))
                if _wants_text(ms):
                    cands.append(self._mk("ocr_extract", ms, capability="ocr",
                                          objective=f"Extract the text from the image '{label}' (OCR) to recover the flag.",
                                          rationale=("The image likely contains hidden text; OCR is the required "
                                                     "next capability to read it."),
                                          evidence_support=0.75, success_probability=0.5, target=label))
            elif category == "web" or any(k in low for k in (".php", ".phtml", ".jsp", ".asp", ".aspx", "uploads/", "/uploads", ".html", ".js", "script")):
                cands.append(self._mk("web_exploit", ms,
                                      objective=f"Access and execute/inspect the discovered web path/artifact '{label}' to trigger execution or extract the flag.",
                                      rationale=f"A new web artifact/path '{label}' was confirmed — inspect/request it.",
                                      evidence_support=0.85, success_probability=0.65, target=label))
            else:
                cands.append(self._mk("artifact_analysis", ms,
                                      objective=f"Analyze the discovered artifact '{label}'.",
                                      rationale="A new artifact was discovered.",
                                      evidence_support=0.7, success_probability=0.55, target=label))
        elif etype in ("service", "technology"):
            tech = (getattr(ev, "related_technology", "") or title).strip()
            if tech:
                cands.append(self._mk("vuln_research", ms,
                                      objective=f"Research known vulnerabilities for the detected service '{tech}'. Extract the flag.",
                                      rationale=f"Service '{tech}' identified — target its known weaknesses.",
                                      evidence_support=0.75, success_probability=0.55, target=target))
        elif etype == "endpoint":
            ep = (getattr(ev, "related_endpoint", "") or title).strip()
            if ep:
                cands.append(self._mk("web_exploit", ms,
                                      objective=f"Test the discovered endpoint '{ep}' for access-control and injection flaws.",
                                      rationale=f"Endpoint '{ep}' discovered — probe it.",
                                      evidence_support=0.7, success_probability=0.5, target=ep))
        elif etype == "vulnerability":
            vuln = (getattr(ev, "related_vulnerability", "") or title).strip()
            if vuln:
                cands.append(self._mk("vuln_exploit", ms,
                                      objective=f"Develop and run an exploit for the identified vulnerability: {vuln}. Extract the flag.",
                                      rationale=f"Vulnerability '{vuln}' confirmed — exploit it.",
                                      evidence_support=0.85, success_probability=0.55, target=target))
        # Everything produced here reacts to concrete evidence — tag it so the coordinator
        # treats it as auto-injectable (memory/state-gap candidates are advisory only).
        for c in cands:
            c.source = "evidence"
        return cands

    # ------------------------------------------------------------------ #
    # §18/§19 memory → advisory candidate actions   §20/§21 playbook adaptation
    # ------------------------------------------------------------------ #

    def _from_memory(self, ms: Any, *, max_memory: int = 4) -> List[CandidateAction]:
        retriever = self._memory
        if retriever is None:
            try:
                from backend.knowledge.memory_retriever import memory_retriever as retriever  # type: ignore
            except Exception:
                return []
        category = getattr(ms, "category", "") or None
        target_type = getattr(ms, "target_type", "") or None
        technologies = list(getattr(ms, "technologies", []) or [])
        try:
            evidence_str = ms.summary() if hasattr(ms, "summary") else ""
            memories = retriever.retrieve(
                evidence=evidence_str,
                category=category,
                technologies=technologies,
                top_k=max_memory,
                include_failures=False,
                capabilities=self._capabilities,
            )
        except Exception as e:
            logger.debug(f"[CandidateGenerator] memory retrieval skipped: {e}")
            return []

        cands: List[CandidateAction] = []
        for m in memories or []:
            technique = getattr(m, "technique", "") or getattr(m, "strategy", "")
            atype = technique_to_action_type(f"{technique} {getattr(m, 'strategy', '')}")
            prof = profile_for(atype)
            kind = getattr(m, "kind", "experience")
            # §19 confidence is advisory: memory confidence & historical success rate
            # feed success_probability, but current evidence_support stays modest so a
            # memory alone never dominates a strongly evidence-backed action.
            hist_conf = float(getattr(m, "confidence", 0.6) or 0.6)
            succ_rate = float(getattr(m, "success_rate", 1.0) or 1.0)
            # Existing (Phase-5) formula — also the fallback when there is no contextual
            # statistical evidence for this technique yet.
            base_prob = _clamp(0.35 + 0.4 * hist_conf * succ_rate)
            success_prob = base_prob
            source = "playbook" if kind == "playbook" else "memory"

            # Phase 6 §9/§13 — observability provenance for THIS candidate. Always carries
            # the memory's own signal; enriched with contextual statistics when wired.
            hist_support: Dict[str, Any] = {
                "source": source,
                "memory_ids": [x for x in [getattr(m, "id", "")] if x],
                "memory_confidence": round(hist_conf, 3),
                "memory_success_rate": round(succ_rate, 3),
                "memory_outcome": getattr(m, "outcome", ""),
            }
            # §9 contextual success statistics refine — never dictate — the probability.
            if self._stats is not None:
                try:
                    ctx_rate, stats = self._stats.contextual_success_rate(
                        technique, category=category, target_type=target_type,
                        technologies=technologies)
                    glob = (stats or {}).get("global", {})
                    ctx = (stats or {}).get("contextual", {})
                    hist_support.update({
                        "similar_challenges": int(glob.get("sample_size", 0) or 0),
                        "historical_success_rate": glob.get("success_rate"),
                        "contextual_similar_challenges": int(ctx.get("sample_size", 0) or 0),
                        "contextual_success_rate": ctx.get("success_rate")
                        if int(ctx.get("sample_size", 0) or 0) else None,
                        "context": (stats or {}).get("context"),
                    })
                    if ctx_rate is not None:
                        # History is ONE input (Part 7): blend 50/50 with the memory-derived
                        # base so a strong track record lifts the candidate and a poor one
                        # tempers it, but neither ever fully controls the score.
                        success_prob = _clamp(0.5 * base_prob + 0.5 * float(ctx_rate))
                        hist_support["applied_rate"] = round(float(ctx_rate), 3)
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug(f"[CandidateGenerator] technique stats skipped: {e}")
            hist_support["applied_success_probability"] = round(success_prob, 3)

            # §21 playbook adaptation: only propose it if the environment can run its
            # capability; otherwise record the capability so the pre-dispatch gate (Phase
            # 4.x) can decide, rather than silently executing or silently dropping.
            cand = self._mk(
                atype, ms,
                objective=(f"Adapt a known {kind} technique to the current target: "
                           f"{technique}. {getattr(m, 'strategy', '')}".strip())[:400],
                rationale=(f"Relevant {kind} (confidence {hist_conf:.2f}, "
                           f"success rate {succ_rate:.2f}) suggests '{technique}'. "
                           f"Advisory — validated against current evidence."),
                evidence_support=0.35, success_probability=success_prob,
                source=source, capability=prof.capability,
                target=getattr(ms, "target", "") or "",
            )
            # A memory-sourced candidate is slightly less novel than a fresh idea.
            cand.novelty = 0.85
            cand.historical_support = hist_support
            cands.append(cand)
        return cands

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _mk(self, action_type: str, ms: Any, *, objective: str, rationale: str = "",
            evidence_support: float = 0.3, success_probability: float = 0.5,
            source: str = "reasoning", capability: Optional[str] = None,
            target: str = "") -> CandidateAction:
        prof = profile_for(action_type)
        ig = information_gain_for(action_type, ms)   # §11 gain is relative to current knowledge
        return CandidateAction(
            action_type=action_type,
            objective=objective,
            role=prof.role,
            capability=(capability if capability is not None else prof.capability),
            rationale=rationale,
            information_gain=ig,
            cost=prof.cost,
            risk=prof.risk,
            success_probability=success_probability,
            evidence_support=evidence_support,
            source=source,
            target=target or (getattr(ms, "target", "") or ""),
        )

    @staticmethod
    def _dedupe(cands: List[CandidateAction]) -> List[CandidateAction]:
        seen = set()
        out = []
        for c in cands:
            if c.signature in seen:
                continue
            seen.add(c.signature)
            out.append(c)
        return out


def _get(ms: Any, *names: str) -> list:
    for n in names:
        v = getattr(ms, n, None)
        if v:
            return list(v)
    return []


_TEXT_INTENT = ("text", "read", "hidden", "ocr", "message", "steg", "written", "words", "flag in the image")


def _wants_text(ms: Any) -> bool:
    """Whether the mission objective/description implies reading text out of an image
    (so OCR becomes the required next capability, §29)."""
    hay = " ".join(str(getattr(ms, k, "") or "") for k in ("objective", "description")).lower()
    return any(kw in hay for kw in _TEXT_INTENT)


def _looks_networked(target: str, target_type: str) -> bool:
    tt = (target_type or "").lower()
    if tt in ("live_http", "live_tcp", "interactive_process", "remote_service"):
        return True
    if tt in ("static_file", "local_artifact", "local_executable", "local_script"):
        return False
    t = (target or "").strip()
    if not t or t.startswith("/"):
        return False
    return bool(__import__("re").search(r"(https?://|:\d{2,5}\b|\b\d{1,3}(?:\.\d{1,3}){3}\b|\.[a-z]{2,})", t, __import__("re").I))


def _clamp(x: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo
