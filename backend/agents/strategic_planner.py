import json
import logging
import re
import uuid
from backend.utils.time import utcnow
from typing import Dict, Any, List, Optional
from backend.providers.router import model_router

logger = logging.getLogger("forge.planner")

class StrategicPlanner:
    """Manages pre-flight mission Todo Lists, dynamic task progress, and multi-model stuck reviews."""

    def _generate_fallback_plan(self, category: str, target: str, challenge_name: str) -> List[Dict[str, Any]]:
        """Provides category-specific structured fallback tasks if LLM generation is unavailable."""
        cat = (category or "WEB").upper()
        
        if cat in ["WEB", "GENERAL"]:
            return [
                {
                    "id": "task-1",
                    "phase": "RECON",
                    "title": "Initial Port & Service Fingerprinting",
                    "tool": "nmap / curl",
                    "reasoning": f"Map open ports and HTTP banners on target {target}",
                    "playbook_ref": "web-recon-fingerprint",
                    "status": "IN_PROGRESS",
                    "output_summary": ""
                },
                {
                    "id": "task-2",
                    "phase": "SURFACE_ANALYSIS",
                    "title": "Endpoint Discovery & Technology Profiling",
                    "tool": "curl / ffuf",
                    "reasoning": "Fuzz endpoints, inspect robots.txt, HTML comments, and headers",
                    "playbook_ref": "web-directory-discovery",
                    "status": "PENDING",
                    "output_summary": ""
                },
                {
                    "id": "task-3",
                    "phase": "EXPLOITATION",
                    "title": "Vulnerability Testing & Weaponization",
                    "tool": "python_script (requests/bs4)",
                    "reasoning": "Test for auth bypass, injection (SQLi/SSTI/XSS), or IDOR vulnerabilities",
                    "playbook_ref": "web-vuln-exploit",
                    "status": "PENDING",
                    "output_summary": ""
                },
                {
                    "id": "task-4",
                    "phase": "FLAG_EXTRACTION",
                    "title": "Flag Retrieval from Session / Storage",
                    "tool": "python_script",
                    "reasoning": "Extract the CTF flag string from response body, cookie, or file system",
                    "playbook_ref": "flag-pattern-extract",
                    "status": "PENDING",
                    "output_summary": ""
                },
                {
                    "id": "task-5",
                    "phase": "VERIFICATION",
                    "title": "Flag Validation & Exploit Proof Documentation",
                    "tool": "report_generator",
                    "reasoning": "Verify captured flag format and compile automated walkthrough writeup",
                    "playbook_ref": "solve-verification",
                    "status": "PENDING",
                    "output_summary": ""
                }
            ]
        elif cat in ["PWN", "REV", "REVERSE"]:
            return [
                {
                    "id": "task-1",
                    "phase": "RECON",
                    "title": "Binary Architecture & Security Mitigation Check",
                    "tool": "file / checksec",
                    "reasoning": f"Determine target binary architecture (ELF32/64), NX, ASLR, PIE, Canary",
                    "playbook_ref": "pwn-binary-audit",
                    "status": "IN_PROGRESS",
                    "output_summary": ""
                },
                {
                    "id": "task-2",
                    "phase": "SURFACE_ANALYSIS",
                    "title": "Static Disassembly & Decompilation Review",
                    "tool": "objdump / strings / ghidra",
                    "reasoning": "Examine symbols, hardcoded strings, main control flow, and dangerous functions",
                    "playbook_ref": "rev-decompilation",
                    "status": "PENDING",
                    "output_summary": ""
                },
                {
                    "id": "task-3",
                    "phase": "EXPLOITATION",
                    "title": "Payload Synthesis & Buffer Offset Calculation",
                    "tool": "python_script (pwntools)",
                    "reasoning": "Construct exploit payload (ROP chain, format string, or buffer overflow)",
                    "playbook_ref": "pwn-rop-payload",
                    "status": "PENDING",
                    "output_summary": ""
                },
                {
                    "id": "task-4",
                    "phase": "FLAG_EXTRACTION",
                    "title": "Remote Shell / Flag File Read",
                    "tool": "python_script (remote)",
                    "reasoning": "Execute remote exploit to trigger flag dump or interactive shell cat flag.txt",
                    "playbook_ref": "pwn-shell-read",
                    "status": "PENDING",
                    "output_summary": ""
                },
                {
                    "id": "task-5",
                    "phase": "VERIFICATION",
                    "title": "Flag Validation & Writeup Compilation",
                    "tool": "report_generator",
                    "reasoning": "Verify flag string format and finalize exploit report",
                    "playbook_ref": "solve-verification",
                    "status": "PENDING",
                    "output_summary": ""
                }
            ]
        elif cat in ["CRYPTO", "CRYPTOGRAPHY"]:
            return [
                {
                    "id": "task-1",
                    "phase": "RECON",
                    "title": "Cryptographic Primitive Identification",
                    "tool": "python_script",
                    "reasoning": f"Inspect ciphertext format, modulus size, cipher type (RSA/AES/ECC/XOR)",
                    "playbook_ref": "crypto-primitive-id",
                    "status": "IN_PROGRESS",
                    "output_summary": ""
                },
                {
                    "id": "task-2",
                    "phase": "SURFACE_ANALYSIS",
                    "title": "Mathematical Weakness & Key Reuse Analysis",
                    "tool": "python_script (pycryptodome)",
                    "reasoning": "Check for small exponent (e=3), shared primes, ECB mode, or weak seed PRNG",
                    "playbook_ref": "crypto-math-weakness",
                    "status": "PENDING",
                    "output_summary": ""
                },
                {
                    "id": "task-3",
                    "phase": "EXPLOITATION",
                    "title": "Decryption Solver Synthesis",
                    "tool": "python_script",
                    "reasoning": "Implement mathematical attack (Wiener, Coppersmith, Padding Oracle, Sage)",
                    "playbook_ref": "crypto-attack-solver",
                    "status": "PENDING",
                    "output_summary": ""
                },
                {
                    "id": "task-4",
                    "phase": "FLAG_EXTRACTION",
                    "title": "Plaintext Flag Decoding",
                    "tool": "python_script",
                    "reasoning": "Decode decrypted bytes into flag format string",
                    "playbook_ref": "crypto-flag-decode",
                    "status": "PENDING",
                    "output_summary": ""
                },
                {
                    "id": "task-5",
                    "phase": "VERIFICATION",
                    "title": "Solve Verification & Report",
                    "tool": "report_generator",
                    "reasoning": "Validate decoded flag and generate writeup",
                    "playbook_ref": "solve-verification",
                    "status": "PENDING",
                    "output_summary": ""
                }
            ]
        elif cat in ["FORENSICS", "OSINT"]:
            return [
                {
                    "id": "task-1",
                    "phase": "RECON",
                    "title": "File Format & Metadata Triage",
                    "tool": "file / exiftool / binwalk",
                    "reasoning": f"Inspect container format, headers, timestamps, and embedded streams",
                    "playbook_ref": "forensics-file-triage",
                    "status": "IN_PROGRESS",
                    "output_summary": ""
                },
                {
                    "id": "task-2",
                    "phase": "SURFACE_ANALYSIS",
                    "title": "Deep Artifact / Packet Stream Carving",
                    "tool": "tshark / volatility / strings",
                    "reasoning": "Carve streams, PCAP packet conversations, or memory dumps",
                    "playbook_ref": "forensics-stream-carving",
                    "status": "PENDING",
                    "output_summary": ""
                },
                {
                    "id": "task-3",
                    "phase": "EXPLOITATION",
                    "title": "Hidden Data / Stego Extraction",
                    "tool": "python_script / zsteg",
                    "reasoning": "Extract hidden payloads, LSB steganography, or compressed archives",
                    "playbook_ref": "forensics-stego-extract",
                    "status": "PENDING",
                    "output_summary": ""
                },
                {
                    "id": "task-4",
                    "phase": "FLAG_EXTRACTION",
                    "title": "Flag String Reconstruction",
                    "tool": "python_script",
                    "reasoning": "Reassemble discovered fragments into full flag",
                    "playbook_ref": "flag-reassemble",
                    "status": "PENDING",
                    "output_summary": ""
                },
                {
                    "id": "task-5",
                    "phase": "VERIFICATION",
                    "title": "Flag Verification & Report",
                    "tool": "report_generator",
                    "reasoning": "Verify flag string format and finalize report",
                    "playbook_ref": "solve-verification",
                    "status": "PENDING",
                    "output_summary": ""
                }
            ]
        else:
            return [
                {
                    "id": "task-1",
                    "phase": "RECON",
                    "title": f"Target Profiling & Assessment ({target})",
                    "tool": "nmap / file",
                    "reasoning": "Assess target scope and identify core technologies",
                    "playbook_ref": "general-recon",
                    "status": "IN_PROGRESS",
                    "output_summary": ""
                },
                {
                    "id": "task-2",
                    "phase": "SURFACE_ANALYSIS",
                    "title": "Vulnerability Hypotheses Formulation",
                    "tool": "python_script",
                    "reasoning": "Analyze inputs, endpoints, or binary functions",
                    "playbook_ref": "general-analysis",
                    "status": "PENDING",
                    "output_summary": ""
                },
                {
                    "id": "task-3",
                    "phase": "EXPLOITATION",
                    "title": "Exploit Execution & Payload Delivery",
                    "tool": "python_script",
                    "reasoning": "Deliver targeted exploit to capture flag",
                    "playbook_ref": "general-exploit",
                    "status": "PENDING",
                    "output_summary": ""
                },
                {
                    "id": "task-4",
                    "phase": "FLAG_EXTRACTION",
                    "title": "Flag Extraction & Verification",
                    "tool": "python_script",
                    "reasoning": "Verify extracted flag string matches CTF pattern",
                    "playbook_ref": "general-flag",
                    "status": "PENDING",
                    "output_summary": ""
                }
            ]

    async def generate_initial_plan(
        self,
        challenge_id: str,
        challenge_name: str,
        category: str,
        difficulty: str,
        target: str,
        description: str = "",
        playbook_snippets: str = "",
        turbo_recon_summary: str = ""
    ) -> Dict[str, Any]:
        """Generates pre-flight mission plan / todo list upfront using AI reasoning."""
        logger.info(f"Generating pre-flight mission plan for challenge {challenge_id} ({challenge_name})")

        prompt = (
            f"You are the Lead Security Analysis Planner for an authorized educational CTF lab challenge.\n"
            f"Create an actionable, tactical pre-flight Mission Todo List for the following target:\n\n"
            f"Challenge: {challenge_name}\n"
            f"Category: {category}\n"
            f"Difficulty: {difficulty}\n"
            f"Target Scope: {target}\n"
            f"Description: {description or 'Standard CTF Target'}\n"
            f"{f'Turbo Recon Summary: {turbo_recon_summary}' if turbo_recon_summary else ''}\n"
            f"{f'Relevant Playbooks: {playbook_snippets}' if playbook_snippets else ''}\n\n"
            f"OUTPUT REQUIREMENTS:\n"
            f"Return a strict JSON object with the following schema:\n"
            f"{{\n"
            f'  "summary": "1-sentence high-level analysis strategy",\n'
            f'  "tasks": [\n'
            f'    {{\n'
            f'      "id": "task-1",\n'
            f'      "phase": "RECON" | "SURFACE_ANALYSIS" | "EXPLOITATION" | "FLAG_EXTRACTION" | "VERIFICATION",\n'
            f'      "title": "Clear concise task name",\n'
            f'      "tool": "Primary tool or script to use (e.g. nmap, curl, python_script, pwntools, ffuf)",\n'
            f'      "reasoning": "Why this specific step is critical for this target",\n'
            f'      "playbook_ref": "Optional playbook tag or id"\n'
            f'    }}\n'
            f'  ]\n'
            f"}}\n"
            f"Provide exactly 4 to 6 focused, sequential tasks. The first task must cover initial reconnaissance and target fingerprinting."
        )

        model_used = "model_router"
        tasks: List[Dict[str, Any]] = []
        summary = f"Autonomous {category} security analysis and flag discovery on {target}"

        try:
            llm_response = await model_router.route_request(
                prompt=prompt,
                capability="general_reasoning",
                system_instruction="You are an expert security analysis and CTF lab planner. Always output valid JSON only.",
                speed_tier="fast"
            )
            raw_output = (llm_response.content or "").strip()
            model_used = getattr(llm_response, "model", getattr(llm_response, "provider_name", "AI Planner"))

            # Parse JSON
            json_match = re.search(r"\{[\s\S]*\}", raw_output)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, dict) and "tasks" in parsed and isinstance(parsed["tasks"], list) and len(parsed["tasks"]) > 0:
                    summary = parsed.get("summary", summary)
                    for idx, t in enumerate(parsed["tasks"]):
                        tasks.append({
                            "id": t.get("id") or f"task-{idx+1}",
                            "phase": (t.get("phase") or "RECON").upper(),
                            "title": t.get("title") or f"Step #{idx+1}",
                            "tool": t.get("tool") or "auto",
                            "reasoning": t.get("reasoning") or "Strategic progression",
                            "playbook_ref": t.get("playbook_ref") or "",
                            "status": "IN_PROGRESS" if idx == 0 else "PENDING",
                            "output_summary": ""
                        })
        except Exception as e:
            logger.warning(f"AI Planner generation failed: {e}. Utilizing tailored category fallback.")

        if not tasks:
            tasks = self._generate_fallback_plan(category, target, challenge_name)

        mission_plan = {
            "challenge_id": challenge_id,
            "status": "IN_PROGRESS",
            "summary": summary,
            "model": model_used,
            "created_at": utcnow().isoformat(),
            "updated_at": utcnow().isoformat(),
            "tasks": tasks,
            "strategic_reviews": []
        }

        return mission_plan

    def calculate_progress(self, mission_plan: Optional[Dict[str, Any]], flag_captured: bool = False) -> int:
        """Calculates dynamic progress percentage based on task completion milestones."""
        if flag_captured:
            return 100
        if not mission_plan or "tasks" not in mission_plan or not mission_plan["tasks"]:
            return 10

        tasks = mission_plan["tasks"]
        total_tasks = len(tasks)
        if total_tasks == 0:
            return 10

        completed = sum(1 for t in tasks if t.get("status") == "COMPLETED")
        in_progress = sum(0.5 for t in tasks if t.get("status") == "IN_PROGRESS")
        revised = sum(0.3 for t in tasks if t.get("status") == "REVISED")

        calc_pct = int(((completed + in_progress + revised) / total_tasks) * 100)
        return min(95, max(10, calc_pct))

    def update_task_progress(
        self,
        mission_plan: Dict[str, Any],
        turn: int,
        executed_command: str,
        output_snippet: str,
        is_success: bool = True
    ) -> Dict[str, Any]:
        """Dynamically advances task status in the todo list during execution turns."""
        if not mission_plan or "tasks" not in mission_plan:
            return mission_plan

        tasks = mission_plan["tasks"]
        
        # Find currently active task
        active_idx = -1
        for idx, t in enumerate(tasks):
            if t.get("status") == "IN_PROGRESS":
                active_idx = idx
                break

        if active_idx == -1:
            for idx, t in enumerate(tasks):
                if t.get("status") == "PENDING":
                    t["status"] = "IN_PROGRESS"
                    active_idx = idx
                    break

        if active_idx != -1:
            curr_task = tasks[active_idx]
            curr_task["output_summary"] = f"[Turn #{turn}] `{executed_command[:60]}` -> {output_snippet[:120]}"

            # Determine if this task milestone is achieved or should transition to next task
            # E.g. Turn 1-2 finishes Recon, Turn 3-5 finishes Surface Analysis, etc.
            phase = curr_task.get("phase", "").upper()
            should_advance = False

            if phase == "RECON" and (turn >= 2 or "nmap" in executed_command or "banner" in output_snippet.lower()):
                should_advance = True
            elif phase == "SURFACE_ANALYSIS" and (turn >= 5 or "endpoint" in output_snippet.lower() or "cookie" in output_snippet.lower() or "200 OK" in output_snippet):
                should_advance = True
            elif phase == "EXPLOITATION" and (turn >= 9 or "flag" in output_snippet.lower() or "solve.py" in executed_command):
                should_advance = True

            if should_advance:
                curr_task["status"] = "COMPLETED"
                # Activate next pending task
                for nxt_idx in range(active_idx + 1, len(tasks)):
                    if tasks[nxt_idx].get("status") == "PENDING":
                        tasks[nxt_idx]["status"] = "IN_PROGRESS"
                        break

        mission_plan["updated_at"] = utcnow().isoformat()
        return mission_plan

    async def review_and_adapt_plan(
        self,
        challenge_name: str,
        category: str,
        target: str,
        mission_plan: Dict[str, Any],
        stuck_reason: str,
        recent_history: List[str],
        primary_model: str
    ) -> Dict[str, Any]:
        """Multi-Model Review: Invokes a secondary deep-reasoning model to diagnose blockages and adapt the plan."""
        logger.info(f"Triggering Multi-Model Strategic Review for {challenge_name} (Stuck Reason: {stuck_reason})")

        history_str = "\n".join(recent_history[-5:]) if recent_history else "No history recorded."
        current_tasks_str = json.dumps(mission_plan.get("tasks", []), indent=2)

        prompt = (
            f"You are the Senior Strategic Review Model for the FORGE Autonomous Pentest System.\n"
            f"The primary agent ({primary_model}) is currently STUCK or repeating actions without progress.\n\n"
            f"TARGET CONTEXT:\n"
            f"Challenge: {challenge_name} | Category: {category} | Target: {target}\n"
            f"STUCK REASON: {stuck_reason}\n\n"
            f"RECENT COMMAND & OUTPUT HISTORY:\n{history_str}\n\n"
            f"CURRENT MISSION TODO LIST:\n{current_tasks_str}\n\n"
            f"YOUR TASK:\n"
            f"1. Perform Root Cause Analysis on why the primary approach failed or deadlocked.\n"
            f"2. Propose a concrete strategic pivot (e.g. bypass, alternative protocol, custom Python exploit script, parameter manipulation).\n"
            f"3. Return a revised task list incorporating the strategic pivot.\n\n"
            f"OUTPUT FORMAT (STRICT JSON ONLY):\n"
            f"{{\n"
            f'  "diagnosis": "Detailed 1-2 sentence root cause diagnosis",\n'
            f'  "pivot_strategy": "Concrete pivot instructions for the agent",\n'
            f'  "new_tasks": [\n'
            f'    {{\n'
            f'      "id": "task-pivot-1",\n'
            f'      "phase": "EXPLOITATION",\n'
            f'      "title": "Task title for the pivot",\n'
            f'      "tool": "Specific tool or python library",\n'
            f'      "reasoning": "Why this bypasses the blocker",\n'
            f'      "status": "IN_PROGRESS"\n'
            f'    }}\n'
            f'  ]\n'
            f"}}"
        )

        reviewer_model = "Gemini 1.5 Pro / Strategic Reviewer"
        diagnosis = "Primary agent reached repetition limit on current endpoint. Pivoting attack methodology."
        pivot_strategy = "Construct specialized Python script to inspect alternative endpoints, response headers, and session tokens."
        new_tasks = []

        # 1. Primary Stuck Analyzer: Query Gemini first
        gemini_success = False
        try:
            gemini_provider = model_router.providers.get("gemini")
            if gemini_provider and await gemini_provider.is_available():
                logger.info(f"[StrategicPlanner] Invoking Gemini 1.5 Pro to diagnose stuck state for {challenge_name}...")
                gemini_resp = await gemini_provider.generate_response(
                    prompt=prompt,
                    system_instruction="You are a Principal Cyber Operations & Exploit Strategist. Analyze why the agent is stuck and output valid JSON only.",
                    capability="general_reasoning"
                )
                if not gemini_resp.is_refusal and gemini_resp.content:
                    raw_text = gemini_resp.content.strip()
                    # Verify Gemini did not state inability to handle the problem
                    if not any(neg in raw_text.lower() for neg in ["i cannot", "i can't", "unable to assist", "refuse", "against policy"]):
                        json_match = re.search(r"\{[\s\S]*\}", raw_text)
                        if json_match:
                            parsed = json.loads(json_match.group(0))
                            if isinstance(parsed, dict) and "diagnosis" in parsed:
                                diagnosis = parsed.get("diagnosis", diagnosis)
                                pivot_strategy = parsed.get("pivot_strategy", pivot_strategy)
                                if "new_tasks" in parsed and isinstance(parsed["new_tasks"], list) and len(parsed["new_tasks"]) > 0:
                                    new_tasks = parsed["new_tasks"]
                                reviewer_model = getattr(gemini_resp, "model_name", "Gemini 1.5 Pro")
                                gemini_success = True
        except Exception as e:
            logger.warning(f"[StrategicPlanner] Gemini stuck analysis encountered error: {e}. Falling back to next best model.")

        # 2. Fallback to Next Best Model (AgentRouter Codex DeepSeek-V4-Flash / GLM-5.3)
        if not gemini_success:
            try:
                logger.info(f"[StrategicPlanner] Gemini unavailable or escalated; routing stuck review to AgentRouter Codex...")
                review_resp = await model_router.route_request(
                    prompt=prompt,
                    capability="code_analysis",
                    target_model="deepseek-v4-flash",
                    system_instruction="You are a Principal Security Researcher conducting strategic exploit review. Output valid JSON only.",
                    speed_tier="deep"
                )
                raw_text = (review_resp.content or "").strip()
                reviewer_model = getattr(review_resp, "model", getattr(review_resp, "model_name", "DeepSeek-V4-Flash (Codex)"))

                json_match = re.search(r"\{[\s\S]*\}", raw_text)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, dict):
                        diagnosis = parsed.get("diagnosis", diagnosis)
                        pivot_strategy = parsed.get("pivot_strategy", pivot_strategy)
                        if "new_tasks" in parsed and isinstance(parsed["new_tasks"], list) and len(parsed["new_tasks"]) > 0:
                            new_tasks = parsed["new_tasks"]
            except Exception as e:
                logger.warning(f"Strategic review model query failed: {e}. Applying rule-based adaptation.")

        if not new_tasks:
            new_tasks = [
                {
                    "id": f"pivot-{uuid.uuid4().hex[:4]}",
                    "phase": "EXPLOITATION",
                    "title": "Methodology Pivot — Specialized Python Solver",
                    "tool": "python_script (requests.Session / pwntools / flask_unsign)",
                    "reasoning": "Bypass repeated endpoint deadlock by writing structured session manipulation or payload script",
                    "status": "IN_PROGRESS"
                }
            ]

        # Log review record
        review_entry = {
            "id": f"rev-{uuid.uuid4().hex[:8]}",
            "timestamp": utcnow().strftime("%H:%M:%S"),
            "reviewer_model": reviewer_model,
            "stuck_reason": stuck_reason,
            "diagnosis": diagnosis,
            "pivot_strategy": pivot_strategy
        }

        if "strategic_reviews" not in mission_plan:
            mission_plan["strategic_reviews"] = []
        mission_plan["strategic_reviews"].append(review_entry)

        # Merge new pivot tasks into plan
        existing_tasks = mission_plan.get("tasks", [])
        # Mark previous in_progress tasks as REVISED
        for t in existing_tasks:
            if t.get("status") == "IN_PROGRESS":
                t["status"] = "REVISED"

        # Insert new pivot tasks at front of pending list
        merged_tasks = []
        inserted = False
        for t in existing_tasks:
            if t.get("status") in ["COMPLETED", "REVISED"]:
                merged_tasks.append(t)
            elif not inserted:
                for nt in new_tasks:
                    merged_tasks.append({
                        "id": nt.get("id") or f"task-p-{uuid.uuid4().hex[:4]}",
                        "phase": (nt.get("phase") or "EXPLOITATION").upper(),
                        "title": nt.get("title") or "Strategic Pivot Task",
                        "tool": nt.get("tool") or "python_script",
                        "reasoning": nt.get("reasoning") or "Pivot from stuck state",
                        "playbook_ref": nt.get("playbook_ref") or "",
                        "status": "IN_PROGRESS" if len(merged_tasks) == 0 or merged_tasks[-1]["status"] != "IN_PROGRESS" else "PENDING",
                        "output_summary": f"Injected via Strategic Review ({reviewer_model})"
                    })
                inserted = True
                merged_tasks.append(t)
            else:
                merged_tasks.append(t)

        if not inserted:
            for nt in new_tasks:
                merged_tasks.append(nt)

        mission_plan["tasks"] = merged_tasks
        mission_plan["updated_at"] = utcnow().isoformat()
        return mission_plan

    async def review_swarm_pivot(
        self,
        challenge_name: str,
        category: str,
        target: str,
        exhausted_strategies: List[str],
        recent_history: List[str],
        all_strategy_attempts: Dict[str, int],
    ) -> tuple:
        """Diagnose why a strategy class was exhausted and return a pivot directive.

        Pure / side-effect-free: accepts swarm shared state, returns
        ``(pivot_strategy_text: str, banned_strategies: List[str])`` without
        mutating any caller data.  Called once per exhausted strategy event;
        the caller writes the result to ``board.pivot_directive``.

        Uses the same Gemini-first / DeepSeek-fallback chain as
        ``review_and_adapt_plan``.
        """
        logger.info(
            f"[StrategicPlanner] review_swarm_pivot for {challenge_name} "
            f"| exhausted={exhausted_strategies}"
        )
        history_str = "\n".join(recent_history[-8:]) if recent_history else "No history recorded."
        attempts_str = json.dumps(all_strategy_attempts, indent=2)
        banned_str = ", ".join(exhausted_strategies) if exhausted_strategies else "(none yet)"

        prompt = (
            f"You are the Senior Strategic Review Model for the FORGE Autonomous Pentest System.\n"
            f"The swarm of agents is STUCK: the following approach classes were tried repeatedly "
            f"across ALL workers without producing any new evidence:\n"
            f"  EXHAUSTED STRATEGIES: {banned_str}\n\n"
            f"TARGET CONTEXT:\n"
            f"Challenge: {challenge_name} | Category: {category} | Target: {target}\n\n"
            f"STRATEGY ATTEMPT COUNTS (turns per class):\n{attempts_str}\n\n"
            f"RECENT CROSS-AGENT COMMAND/OUTPUT HISTORY:\n{history_str}\n\n"
            f"YOUR TASK:\n"
            f"1. Identify the root cause: why did the exhausted strategy classes fail?\n"
            f"2. Propose a CONCRETE pivot — a genuinely different class of approach "
            f"(e.g. if credential_stuffing is exhausted, suggest jwt_tamper or idor_bypass).\n"
            f"3. List ALL strategy labels that should remain banned (the exhausted ones + "
            f"any variations that would have the same dead end).\n\n"
            f"OUTPUT FORMAT (STRICT JSON ONLY):\n"
            f"{{\n"
            f'  "diagnosis": "1-2 sentence root cause",\n'
            f'  "pivot_strategy": "Concrete, actionable pivot instruction for the agents",\n'
            f'  "banned_strategies": ["label1", "label2"]\n'
            f"}}"
        )

        pivot_strategy = (
            f"All {banned_str} approaches exhausted. "
            "Switch to a completely different attack class — enumerate the application "
            "source/config, try JWT/session token manipulation, or probe for IDOR/SSRF."
        )
        banned: List[str] = list(exhausted_strategies)

        # Primary: Gemini
        gemini_success = False
        try:
            gemini_provider = model_router.providers.get("gemini")
            if gemini_provider and await gemini_provider.is_available():
                resp = await gemini_provider.generate_response(
                    prompt=prompt,
                    system_instruction=(
                        "You are a Principal Cyber Operations Strategist. "
                        "Analyze the stuck swarm and output valid JSON only."
                    ),
                    capability="general_reasoning",
                )
                if not resp.is_refusal and resp.content:
                    raw = resp.content.strip()
                    if not any(neg in raw.lower() for neg in ["i cannot", "i can't", "unable to assist"]):
                        m = re.search(r"\{[\s\S]*\}", raw)
                        if m:
                            parsed = json.loads(m.group(0))
                            if isinstance(parsed, dict) and "pivot_strategy" in parsed:
                                pivot_strategy = parsed["pivot_strategy"]
                                extra_banned = parsed.get("banned_strategies", [])
                                if isinstance(extra_banned, list):
                                    banned = list(set(banned) | set(extra_banned))
                                gemini_success = True
        except Exception as e:
            logger.warning(f"[StrategicPlanner] review_swarm_pivot Gemini call failed: {e}")

        # Fallback: model router (DeepSeek / best available)
        if not gemini_success:
            try:
                resp = await model_router.route_request(
                    prompt=prompt,
                    capability="code_analysis",
                    system_instruction=(
                        "You are a Principal Security Researcher. "
                        "Analyze the stuck swarm and output valid JSON only."
                    ),
                    speed_tier="deep",
                )
                raw = (resp.content or "").strip()
                m = re.search(r"\{[\s\S]*\}", raw)
                if m:
                    parsed = json.loads(m.group(0))
                    if isinstance(parsed, dict) and "pivot_strategy" in parsed:
                        pivot_strategy = parsed["pivot_strategy"]
                        extra_banned = parsed.get("banned_strategies", [])
                        if isinstance(extra_banned, list):
                            banned = list(set(banned) | set(extra_banned))
            except Exception as e:
                logger.warning(f"[StrategicPlanner] review_swarm_pivot fallback call failed: {e}")

        return pivot_strategy, banned


strategic_planner = StrategicPlanner()
