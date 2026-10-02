"""
FORGE Experience Extractor.

Transforms a completed run's messy execution trace (the SwarmBlackboard) into a
single, GENERALIZED :class:`ExperienceRecord` — the "distill a solve into reusable
knowledge" step of the intelligence loop (§3).

Design constraints:
* Deterministic and fast — no LLM calls (rule §17). Everything is derived from
  the blackboard's already-collected evidence.
* Duck-typed on the board (no ``import swarm_orchestrator``) to avoid an import
  cycle and to keep the extractor unit-testable with a lightweight stub.
* Every free-text / command / output field is passed through the
  :class:`Generalizer` so challenge-specific targets, flags, credentials and
  tokens never survive into reusable memory (rule §4).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from backend.knowledge.memory_models import (
    ExperienceRecord,
    AttemptRecord,
    Generalizer,
    classify_technique,
    derive_detection_indicators,
    infer_environment_requirements,
)

logger = logging.getLogger("forge.experience_extractor")

# Rejection / failure signals in tool output → the attempt that produced them
# was a FAILED approach worth remembering ("don't immediately repeat this") (§5).
_REJECTION_SIGNALS = [
    "not found", "rejected", "invalid", "forbidden", "denied", "unauthorized",
    "403", "401", "not allowed", "bad request", "error", "failed", "no such",
    "not permitted", "disallowed", "blocked",
]

# Lightweight technology fingerprints scanned across all evidence.
_TECH_FINGERPRINTS = {
    "nginx": ["nginx"], "apache": ["apache", "httpd"], "iis": ["microsoft-iis", "iis/"],
    "php": ["php", "x-powered-by: php", ".php"], "flask": ["flask", "werkzeug"],
    "werkzeug": ["werkzeug"], "django": ["django", "csrftoken"], "express": ["express", "x-powered-by: express"],
    "node.js": ["node.js", "nodejs"], "wordpress": ["wp-content", "wordpress", "wp-json"],
    "tomcat": ["tomcat", "jsessionid"], "mysql": ["mysql", "mariadb"],
    "postgresql": ["postgres", "psql"], "sqlite": ["sqlite"], "python": ["python", "gunicorn", "uvicorn"],
    "ruby": ["ruby", "rails", "puma"], "jwt": ["eyj", "jwt"], "graphql": ["graphql", "__schema"],
    "redis": ["redis"], "mongodb": ["mongodb", "mongo"],
}


class ExperienceExtractor:
    """Build a generalized :class:`ExperienceRecord` from a completed swarm run."""

    def extract_from_board(self, board: Any, flag: str = "", outcome: str = "success") -> ExperienceRecord:
        """Distill *board* into an ExperienceRecord. Never raises — returns a best-effort
        record so a distillation failure can never break run completion."""
        try:
            return self._extract(board, flag, outcome)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"[ExperienceExtractor] extraction fell back to minimal record: {e}")
            return ExperienceRecord(
                source="forge_run",
                source_run_id=getattr(board, "run_id", None),
                source_challenge_id=getattr(board, "challenge_id", None),
                challenge_name=getattr(board, "challenge_name", "") or "",
                category=(getattr(board, "category", "") or "web").lower(),
                difficulty=getattr(board, "difficulty", "") or "MEDIUM",
                technique=f"{(getattr(board, 'category', '') or 'general').upper()} solve methodology",
                tags=[(getattr(board, "category", "") or "general").lower()],
                outcome=outcome,
                confidence=0.6,
            )

    # ------------------------------------------------------------------ #

    def _extract(self, board: Any, flag: str, outcome: str) -> ExperienceRecord:
        category = (getattr(board, "category", "") or "web").lower()
        difficulty = getattr(board, "difficulty", "") or "MEDIUM"
        target_scope = getattr(board, "target_scope", "") or ""
        target_tokens = [t.strip() for t in target_scope.split("+") if t.strip()]

        endpoints = sorted(getattr(board, "discovered_endpoints", set()) or [])
        headers: Dict[str, str] = dict(getattr(board, "extracted_headers", {}) or {})
        exec_history: List[Dict[str, Any]] = list(getattr(board, "execution_history", []) or [])

        # Secrets to scrub: decoded secrets, candidate tokens, and observed cookie values.
        secrets: List[str] = []
        for s in (getattr(board, "deobfuscated_secrets", []) or []):
            if isinstance(s, dict):
                secrets.extend(str(v) for v in s.values() if v)
            elif s:
                secrets.append(str(s))
        secrets.extend(str(t) for t in (getattr(board, "candidate_tokens", set()) or []))
        for ck, cv in (getattr(board, "observed_cookies", {}) or {}).items():
            if cv:
                secrets.append(str(cv))
        usernames = [str(u) for u in (getattr(board, "candidate_usernames", set()) or [])]

        def g(text: str) -> str:
            return Generalizer.generalize(
                text, target_tokens=target_tokens, flag=flag, secrets=secrets, usernames=usernames
            )

        # Evidence corpus for classification & tech fingerprinting.
        # Commands (the agent's chosen technique) are weighted 3× over output (the
        # target's response) so the classifier's needle-count scoring reflects what
        # the agent DID, not noise in what the target returned (Bug 5 fix).
        corpus_parts: List[str] = [getattr(board, "description", "") or ""]
        corpus_parts.extend(f"{k}: {v}" for k, v in headers.items())
        corpus_parts.extend(endpoints)
        for step in exec_history:
            cmd = str(step.get("command", ""))
            out = str(step.get("output", ""))
            # Repeat commands to weight them higher in needle-count scoring.
            if cmd:
                corpus_parts.extend([cmd] * 3)
            if out:
                corpus_parts.append(out)
        evidence_text = "\n".join(p for p in corpus_parts if p)

        # ── Split the trace into the winning chain vs failed attempts (§5) ──
        chain, attempts, failed_techniques = self._split_trace(exec_history, flag, g)

        # Build raw_commands using only the FIRST LINE of each command string.
        # Multi-line commands (heredocs, inline Python cat-writes) embed file content
        # in subsequent lines; those lines may contain technique-rule keywords (e.g.
        # '<script>' in a print() call) that are NOT attack actions.
        # The shell verb + flags always appear on line 0 — that is the reliable signal.
        raw_commands = [
            str(step.get("command", "")).strip().splitlines()[0].strip()
            for step in exec_history
            if str(step.get("command", "")).strip()
        ]
        raw_commands = [c for c in raw_commands if c]  # drop any blanks after split

        clf = classify_technique(
            evidence_text,
            category,
            commands=raw_commands,
            winning_chain=chain,
        )
        technique = clf["technique"]
        tags = clf["tags"]
        success_indicators = list(clf["success_indicators"])

        technologies = self._detect_technologies(evidence_text)
        vulnerabilities = [technique] if outcome == "success" else []

        commands_used = []
        for step in exec_history:
            cmd = g(str(step.get("command", "")).strip())
            if cmd and cmd not in commands_used:
                commands_used.append(cmd)
        commands_used = commands_used[:25]

        important_outputs: List[str] = []
        verification_evidence = ""
        for step in exec_history:
            out = str(step.get("output", ""))
            if flag and flag in out:
                verification_evidence = g(out[:600])
            if out and self._has_signal(out):
                snippet = g(out[:300])
                if snippet and snippet not in important_outputs:
                    important_outputs.append(snippet)
        important_outputs = important_outputs[:8]
        if not verification_evidence and outcome == "success":
            verification_evidence = "Flag verified from live tool output (value redacted as {FLAG})."

        observed_conditions = self._summarize_conditions(endpoints, headers, technologies, g)
        applicable_conditions = self._applicable_conditions(technique, technologies, clf["signatures"])
        generalized_strategy = self._strategy_for(technique, category)
        prerequisites = self._prereqs_for(tags, technologies)
        detection = derive_detection_indicators(tags)
        env_reqs = infer_environment_requirements(commands_used, category)

        target_characteristics = {
            "category": category,
            "difficulty": difficulty,
            "endpoint_count": len(endpoints),
            "technologies": technologies,
            "has_binary_artifact": bool(getattr(board, "artifact_classification", None)),
        }
        initial_observations = g("; ".join(endpoints[:8])) if endpoints else observed_conditions

        # Confidence: a single verified FORGE solve is HIGH-ish; downgrade for failure.
        if outcome == "success":
            confidence = 0.78 if chain else 0.7
        else:
            confidence = 0.4

        return ExperienceRecord(
            source="forge_run",
            source_run_id=getattr(board, "run_id", None),
            source_challenge_id=getattr(board, "challenge_id", None),
            challenge_name=getattr(board, "challenge_name", "") or "",
            category=category,
            difficulty=difficulty,
            technique=technique,
            tags=tags,
            target_characteristics=target_characteristics,
            initial_observations=initial_observations,
            observed_conditions=observed_conditions,
            applicable_conditions=applicable_conditions,
            discovered_endpoints=[g(e) for e in endpoints[:20]],
            technologies=technologies,
            vulnerabilities=vulnerabilities,
            successful_techniques=[technique] if outcome == "success" else [],
            failed_techniques=failed_techniques,
            commands_used=commands_used,
            important_tool_outputs=important_outputs,
            successful_attack_chain=chain,
            verification_evidence=verification_evidence,
            success_indicators=success_indicators or [r"flag pattern present in output"],
            prerequisites=prerequisites,
            generalized_strategy=generalized_strategy,
            detection_indicators=detection,
            required_os=env_reqs["required_os"],
            required_tools=env_reqs["tools"],
            required_python_libs=env_reqs["python_libs"],
            outcome=outcome,
            confidence=confidence,
            attempts=attempts,
        )

    # ------------------------------------------------------------------ #

    def _split_trace(self, exec_history, flag, g):
        """Return (winning_chain, attempts, failed_techniques), all generalized."""
        chain: List[str] = []
        attempts: List[AttemptRecord] = []
        failed_techniques: List[Dict[str, str]] = []

        flag_idx = None
        if flag:
            for i, step in enumerate(exec_history):
                if flag in str(step.get("output", "")):
                    flag_idx = i
                    break

        seq = 0
        for i, step in enumerate(exec_history):
            cmd_raw = str(step.get("command", "")).strip()
            if not cmd_raw:
                continue
            out = str(step.get("output", ""))
            note = str(step.get("note", ""))
            gcmd = g(cmd_raw)
            low = (out + " " + note).lower()
            rejected = any(sig in low for sig in _REJECTION_SIGNALS)
            is_winning = (flag_idx is not None and i <= flag_idx) and not rejected

            seq += 1
            if rejected:
                reason = self._reason_from(out, note, g)
                attempts.append(AttemptRecord(
                    sequence=seq, approach=gcmd[:180], technique="", outcome="failure", reason=reason,
                    evidence=g(out[:200]),
                ))
                entry = {"approach": gcmd[:180], "reason": reason}
                if entry not in failed_techniques:
                    failed_techniques.append(entry)
            else:
                attempts.append(AttemptRecord(
                    sequence=seq, approach=gcmd[:180], technique="", outcome="success" if is_winning else "neutral",
                    reason="", evidence=g(out[:200]),
                ))
                if is_winning and gcmd not in chain:
                    chain.append(gcmd)

        # If we never located the flag in output (e.g. solver-script solve), fall back to
        # the last few non-rejected commands as the effective winning chain.
        if not chain:
            for step in exec_history[-6:]:
                cmd_raw = str(step.get("command", "")).strip()
                out = str(step.get("output", "")).lower()
                if cmd_raw and not any(sig in out for sig in _REJECTION_SIGNALS):
                    gcmd = g(cmd_raw)
                    if gcmd not in chain:
                        chain.append(gcmd)

        return chain[:12], attempts[:30], failed_techniques[:15]

    @staticmethod
    def _reason_from(out: str, note: str, g) -> str:
        low = (out + " " + note).lower()
        for sig in ("403", "401", "forbidden", "unauthorized", "not found", "rejected",
                    "invalid", "denied", "not allowed"):
            if sig in low:
                return f"Target responded with '{sig}' — approach did not work."
        return g((note or out)[:160]) or "Approach produced no useful result."

    @staticmethod
    def _has_signal(out: str) -> bool:
        low = out.lower()
        keys = ("http/1", "server:", "uid=", "root:x:", "<title", "200 ok", "flag",
                "select", "error", "version", "<!--", "set-cookie", "www-authenticate")
        return any(k in low for k in keys)

    @staticmethod
    def _detect_technologies(evidence_text: str) -> List[str]:
        low = evidence_text.lower()
        found = []
        for tech, needles in _TECH_FINGERPRINTS.items():
            if any(n in low for n in needles):
                found.append(tech)
        return found[:12]

    @staticmethod
    def _summarize_conditions(endpoints, headers, technologies, g) -> str:
        parts = []
        if technologies:
            parts.append("Stack fingerprinted: " + ", ".join(technologies))
        if endpoints:
            parts.append(f"{len(endpoints)} endpoint(s) discovered (e.g. " + ", ".join(g(e) for e in endpoints[:4]) + ")")
        interesting = [k for k in headers if k.lower() not in ("content-type", "content-length", "date", "connection")]
        if interesting:
            parts.append("Notable headers: " + ", ".join(interesting[:6]))
        return ". ".join(parts) if parts else "No distinctive conditions recorded."

    @staticmethod
    def _applicable_conditions(technique: str, technologies: List[str], signatures: List[str]) -> str:
        cond = [f"Consider when the target exhibits: {technique.split('(')[0].strip()}"]
        if signatures:
            cond.append("Trigger signatures: " + ", ".join(signatures[:5]))
        if technologies:
            cond.append("Observed stack overlaps: " + ", ".join(technologies[:5]))
        return ". ".join(cond)

    @staticmethod
    def _strategy_for(technique: str, category: str) -> str:
        base = {
            "Server-Side Template Injection (SSTI)": "Probe template delimiters ({{7*7}}, ${7*7}) across reflected inputs; escalate a confirmed engine to RCE via its object model.",
            "SQL Injection": "Fingerprint the DB via error/boolean/time-based probes, then enumerate schema and exfiltrate the flag column with a UNION or stacked query.",
            "Local File Inclusion / Path Traversal": "Test traversal depth and PHP wrappers on file-parameterized endpoints; read app source, then sensitive/flag files.",
            "File Upload Validation Bypass": "Enumerate what the upload filter checks (extension, MIME, magic bytes) and defeat the weakest layer, then reach the uploaded file's URL.",
            "JWT Forgery / Weak Signature": "Inspect the token header/claims; test alg=none and weak-secret brute force, then forge an elevated token.",
            "Binary Exploitation / Memory Corruption": "checksec the binary, find the overflow/primitive, build the chain (ret2libc/ROP) with pwntools, and pop a shell.",
            "Command Injection / Reverse Shell": "Find the injection point, confirm execution with id, then stage a bounded reverse/bind interaction to read the flag.",
            "Cryptographic Weakness Exploitation": "Identify the scheme and its misuse (key reuse, weak params, oracle) and recover plaintext/key deterministically.",
        }.get(technique)
        return base or f"Apply standard {category.upper()} methodology: recon, identify the weakness, exploit deterministically, verify the flag from real output."

    @staticmethod
    def _prereqs_for(tags: List[str], technologies: List[str]) -> List[str]:
        pre = []
        tagset = set(t.lower() for t in tags)
        if "upload" in tagset:
            pre.append("A file upload endpoint reachable by the tester")
        if "sqli" in tagset:
            pre.append("A parameter reflected into a database query")
        if "ssti" in tagset:
            pre.append("User input reflected through a server-side template")
        if "lfi" in tagset:
            pre.append("A file/path parameter used in an include/read")
        if "jwt" in tagset:
            pre.append("Session/authorization carried in a JWT")
        if not pre:
            pre.append("Network reachability to the target service")
        return pre


experience_extractor = ExperienceExtractor()
