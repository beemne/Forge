import os
import re
import yaml
import sqlite3
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field

logger = logging.getLogger("forge.playbook_vault")


def _broadcast_knowledge_event(event_type: str, playbook_id: str, category: str, extra: Dict = None):
    """Fire-and-forget WebSocket broadcast for knowledge updates."""
    try:
        import asyncio
        from backend.websocket.manager import ws_manager
        payload = {
            "event": "KNOWLEDGE_UPDATED",
            "type": event_type,
            "playbook_id": playbook_id,
            "category": category,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if extra:
            payload.update(extra)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(ws_manager.broadcast(payload))
        except RuntimeError:
            pass  # No event loop running (CLI context) — skip WS broadcast
    except Exception:
        pass  # Non-critical — never break vault operations for WS


PLAYBOOKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "playbooks"))
CATEGORIES = ["web", "pwn", "crypto", "reverse", "forensics", "osint", "auto_generated", "pending_review"]


from pydantic import BaseModel, Field, ConfigDict


class PlaybookSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    category: str
    tags: List[str] = Field(default_factory=list)
    trigger_signatures: List[str] = Field(default_factory=list)
    exploit_template: str
    expected_outcome_signatures: List[str] = Field(default_factory=list)
    notes: str = ""
    source_type: str = Field(default="curated", alias="source")
    confidence_score: float = 1.0
    n_applications: int = Field(default=0, alias="times_used")
    success_rate: float = 1.0
    is_promoted: bool = False
    is_sanitized: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Compatibility getters/setters for legacy field names
    @property
    def source(self) -> str:
        return "generated" if self.source_type in ["generated", "auto_generated"] else self.source_type

    @source.setter
    def source(self, value: str):
        self.source_type = value

    @property
    def times_used(self) -> int:
        return self.n_applications

    @times_used.setter
    def times_used(self, value: int):
        self.n_applications = value


class PlaybookVault:
    """File-based Playbook Vault with SQLite FTS5 search index, hard-trigger prefiltering, and self-learning synthesis flywheel."""

    CONFIDENCE_PROMOTION_THRESHOLD = 0.8
    REUSE_PROMOTION_THRESHOLD = 3

    def __init__(self, base_dir: str = PLAYBOOKS_DIR, auto_reload: bool = True):
        self.base_dir = base_dir
        self._playbook_cache: Dict[str, PlaybookSchema] = {}
        self._ensure_directories()
        self.db_conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_fts_index()
        if auto_reload:
            self.reload_index()

    def _ensure_directories(self):
        """Create category subdirectories under playbooks base dir."""
        for cat in CATEGORIES:
            cat_dir = os.path.join(self.base_dir, cat)
            os.makedirs(cat_dir, exist_ok=True)

    def _init_fts_index(self):
        """Initialize in-memory SQLite FTS5 search table."""
        with self.db_conn:
            self.db_conn.execute("DROP TABLE IF EXISTS playbook_fts")
            self.db_conn.execute("""
                CREATE VIRTUAL TABLE playbook_fts USING fts5(
                    id UNINDEXED,
                    category,
                    tags,
                    trigger_signatures,
                    notes,
                    exploit_template,
                    source UNINDEXED,
                    confidence_score UNINDEXED,
                    n_applications UNINDEXED,
                    success_rate UNINDEXED,
                    is_searchable UNINDEXED
                )
            """)

    def _is_searchable(self, playbook: PlaybookSchema) -> bool:
        """Auto-generated playbooks must pass the confidence/reuse threshold to be indexed in default search."""
        if playbook.source_type not in ["generated", "auto_generated"]:
            return True
        if playbook.is_promoted:
            return True
        if playbook.confidence_score >= self.CONFIDENCE_PROMOTION_THRESHOLD:
            return True
        if playbook.n_applications >= self.REUSE_PROMOTION_THRESHOLD:
            return True
        return False

    def save_playbook(self, playbook: PlaybookSchema) -> str:
        """Save a playbook as a YAML file in the appropriate directory and update in-memory cache and FTS index."""
        target_dir = os.path.join(self.base_dir, "auto_generated" if playbook.source_type in ["generated", "auto_generated"] else playbook.category)
        os.makedirs(target_dir, exist_ok=True)

        file_path = os.path.join(target_dir, f"{playbook.id}.yaml")
        data = playbook.model_dump(by_alias=False)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        self._index_single_playbook(playbook)
        logger.info(f"Saved playbook `{playbook.id}` in `{target_dir}` (searchable={self._is_searchable(playbook)})")
        _broadcast_knowledge_event("PLAYBOOK_SAVED", playbook.id, playbook.category)
        return file_path

    def _index_single_playbook(self, playbook: PlaybookSchema):
        """Add or replace a single playbook entry in memory cache and FTS index."""
        self._playbook_cache[playbook.id] = playbook
        searchable = 1 if self._is_searchable(playbook) else 0
        with self.db_conn:
            self.db_conn.execute("DELETE FROM playbook_fts WHERE id = ?", (playbook.id,))
            self.db_conn.execute("""
                INSERT INTO playbook_fts(
                    id, category, tags, trigger_signatures, notes, exploit_template,
                    source, confidence_score, n_applications, success_rate, is_searchable
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                playbook.id,
                playbook.category,
                " ".join(playbook.tags),
                " ".join(playbook.trigger_signatures),
                playbook.notes,
                playbook.exploit_template,
                playbook.source_type,
                playbook.confidence_score,
                playbook.n_applications,
                playbook.success_rate,
                searchable
            ))

    def load_playbook(self, playbook_id: str) -> Optional[PlaybookSchema]:
        """Load a playbook by id (cache-first, falling back to disk)."""
        if playbook_id in self._playbook_cache:
            return self._playbook_cache[playbook_id]

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file == f"{playbook_id}.yaml" or file == f"{playbook_id}.yml":
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f)
                            pb = PlaybookSchema(**data)
                            self._playbook_cache[pb.id] = pb
                            return pb
                    except Exception as e:
                        logger.error(f"Error loading playbook {file_path}: {e}")
        return None

    def reload_index(self):
        """Scan all YAML files in playbooks directory, build in-memory cache and rebuild FTS index."""
        self._init_fts_index()
        self._playbook_cache.clear()
        count = 0
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".yaml") or file.endswith(".yml"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f)
                            if isinstance(data, dict) and "id" in data and "exploit_template" in data:
                                pb = PlaybookSchema(**data)
                                self._index_single_playbook(pb)
                                count += 1
                    except Exception as e:
                        logger.warning(f"Skipping malformed playbook {file_path}: {e}")
        logger.info(f"PlaybookVault indexed {count} playbooks.")

    def search_playbooks(
        self,
        query: str,
        category: Optional[str] = None,
        recon_artifacts: Optional[str] = None,
        candidate_tags: Optional[List[str]] = None,
        excluded_ids: Optional[Set[str]] = None,
        top_k: int = 2,
        include_unpromoted: bool = False
    ) -> List[PlaybookSchema]:
        """
        Multi-stage retrieval pipeline:
        1. Failure exclusion pruning
        2. Exact trigger signature hard pre-filter against recon artifacts
        3. Candidate tag pre-filter
        4. Multi-factor weighted BM25 ranking (down-weighting unproven auto_generated playbooks)
        """
        excluded = set(excluded_ids) if excluded_ids else set()
        clean_category = category.lower() if category else None

        # =========================================================================
        # STAGE 2: EXACT TRIGGER SIGNATURE HARD PRE-FILTER
        # =========================================================================
        if recon_artifacts and recon_artifacts.strip():
            hard_matches = []
            recon_lower = recon_artifacts.lower()

            for pb_id, pb in self._playbook_cache.items():
                if pb_id in excluded:
                    continue
                if clean_category and pb.category != clean_category:
                    continue
                if not include_unpromoted and not self._is_searchable(pb):
                    continue

                matched_sigs = []
                for sig in pb.trigger_signatures:
                    if not sig or len(sig.strip()) < 3:
                        continue
                    sig_clean = sig.strip().lower()
                    if sig_clean in recon_lower:
                        matched_sigs.append(sig_clean)

                if matched_sigs:
                    # Specificity score: longer and more specific signatures have higher precedence
                    specificity = sum(len(s) for s in matched_sigs)
                    hard_matches.append((specificity, pb))

            if hard_matches:
                # Rank hard matches by specificity (precise signatures win), confidence, and success rate
                hard_matches.sort(
                    key=lambda item: (
                        item[0],
                        item[1].success_rate * item[1].confidence_score * (1.2 if item[1].source_type == "curated" else 0.8),
                        item[1].n_applications
                    ),
                    reverse=True
                )
                return [item[1] for item in hard_matches[:top_k]]

        # =========================================================================
        # STAGE 3 & 4: TAG-BASED PREFILTER + MULTI-FACTOR WEIGHTED BM25 RANKING
        # =========================================================================
        if not query or not query.strip():
            return []

        clean_terms = re.findall(r"[A-Za-z0-9_\-\.]+", query)
        if not clean_terms:
            return []

        searchable_clause = "" if include_unpromoted else "AND is_searchable = 1"
        category_clause = "AND category = ?" if clean_category else ""

        fts_base_query = " OR ".join(f'"{t}"' for t in clean_terms)
        tag_query = fts_base_query

        clean_candidate_tags = []
        if candidate_tags and len(candidate_tags) > 0:
            clean_candidate_tags = [re.sub(r"[^A-Za-z0-9_\-\.]", "", t).lower() for t in candidate_tags if t and len(t) > 1]
            if clean_candidate_tags:
                tag_terms = " OR ".join(f'"{t}"' for t in clean_candidate_tags)
                tag_query = f"({fts_base_query}) OR tags:({tag_terms})"

        sql = f"""
            SELECT id, category, source, confidence_score, n_applications, success_rate, bm25(playbook_fts) as rank
            FROM playbook_fts
            WHERE playbook_fts MATCH ? {searchable_clause} {category_clause}
            ORDER BY rank
            LIMIT 50
        """

        try:
            cursor = self.db_conn.cursor()
            params = [tag_query]
            if clean_category:
                params.append(clean_category)

            cursor.execute(sql, params)
            candidates = cursor.fetchall()

            # If tag-constrained query yielded no results, fallback to unconstrained FTS
            if not candidates and tag_query != fts_base_query:
                fallback_params = [fts_base_query]
                if clean_category:
                    fallback_params.append(clean_category)
                cursor.execute(sql, fallback_params)
                candidates = cursor.fetchall()

            scored_results = []
            for row in candidates:
                pb_id, pb_cat, src, conf, n_apps, succ_rate, raw_rank = row
                if pb_id in excluded:
                    continue

                # Auto-generated penalty: down-weight until proven across >=3 runs
                if src in ["generated", "auto_generated"] and n_apps < 3:
                    src_multiplier = 0.3 + (0.2 * max(0, n_apps))
                elif src == "curated" or conf >= 0.9:
                    src_multiplier = 1.15
                else:
                    src_multiplier = 1.0

                # Tag overlap boost
                tag_overlap_multiplier = 1.0
                if clean_candidate_tags:
                    cached_pb = self.load_playbook(pb_id)
                    if cached_pb:
                        pb_tags_lower = [t.lower() for t in cached_pb.tags]
                        overlap_count = sum(1 for ct in clean_candidate_tags if ct in pb_tags_lower)
                        if overlap_count > 0:
                            tag_overlap_multiplier = 1.0 + (0.75 * overlap_count)

                # SQLite FTS5 bm25 gives lower values for better matches (e.g. -5.2 is better than -1.1)
                # Invert so higher composite score = better rank
                bm25_weight = 1.0 / (abs(raw_rank) + 0.001)
                composite_score = bm25_weight * (0.5 + 0.5 * conf) * (0.5 + 0.5 * succ_rate) * src_multiplier * tag_overlap_multiplier
                scored_results.append((composite_score, pb_id))

            scored_results.sort(key=lambda x: x[0], reverse=True)

            results: List[PlaybookSchema] = []
            for _, pb_id in scored_results[:top_k]:
                pb = self.load_playbook(pb_id)
                if pb:
                    results.append(pb)
            return results

        except Exception as e:
            logger.error(f"FTS Search failed for query '{query}': {e}")
            return self._fallback_search(clean_terms, clean_category, candidate_tags, excluded, top_k, include_unpromoted)

    def _fallback_search(
        self,
        terms: List[str],
        category: Optional[str],
        candidate_tags: Optional[List[str]],
        excluded: Set[str],
        top_k: int,
        include_unpromoted: bool
    ) -> List[PlaybookSchema]:
        """Fallback in-memory search if FTS encounters syntax exceptions."""
        matches = []
        for pb_id, pb in self._playbook_cache.items():
            if pb_id in excluded:
                continue
            if category and pb.category != category:
                continue
            if not include_unpromoted and not self._is_searchable(pb):
                continue

            corpus = f"{' '.join(pb.tags)} {' '.join(pb.trigger_signatures)} {pb.notes} {pb.exploit_template[:200]}".lower()
            score = sum(1 for t in terms if t.lower() in corpus)

            if candidate_tags:
                score += sum(2 for t in candidate_tags if t.lower() in [tag.lower() for tag in pb.tags])

            if score > 0:
                src_mult = 0.3 if (pb.source_type in ["generated", "auto_generated"] and pb.n_applications < 3) else 1.0
                composite = score * (0.5 + 0.5 * pb.confidence_score) * (0.5 + 0.5 * pb.success_rate) * src_mult
                matches.append((composite, pb))

        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:top_k]]

    def synthesize_from_run(
        self,
        challenge_id: str,
        challenge_title: str,
        category: str,
        target_endpoint: str,
        winning_payload: str,
        winning_commands: List[str],
        flag: str
    ) -> Optional[PlaybookSchema]:
        """Self-learning synthesis flywheel: parameterize winning solve trajectory into a new auto_generated playbook."""
        try:
            clean_category = category.lower() if category else "web"
            if clean_category not in CATEGORIES:
                clean_category = "web"

            parameterized_code = winning_payload or "\n".join(winning_commands)
            if target_endpoint:
                clean_target = target_endpoint.strip()
                parameterized_code = parameterized_code.replace(clean_target, "{TARGET_URL}")
                port_match = re.search(r":(\d{2,5})", clean_target)
                if port_match:
                    port_str = port_match.group(1)
                    parameterized_code = parameterized_code.replace(f":{port_str}", ":{PORT}")

            if flag and flag in parameterized_code:
                parameterized_code = parameterized_code.replace(flag, "{FLAG}")

            tags = [clean_category, "auto_learned"]
            signatures = []
            expected_outcomes = []

            lower_code = parameterized_code.lower()
            if "ssti" in lower_code or "jinja" in lower_code or "{{" in parameterized_code:
                tags.extend(["ssti", "template_injection"])
                signatures.extend(["Jinja2", "Werkzeug", "render_template_string"])
                expected_outcomes.extend([r"uid=\d+", r"root:x:0:0", r"\{\{.*\}\}"])
            elif "sqli" in lower_code or "union select" in lower_code or "' or 1=1" in lower_code:
                tags.extend(["sqli", "sql_injection"])
                signatures.extend(["SQL syntax", "UNION SELECT", "database error"])
                expected_outcomes.extend([r"(?:SQL syntax|mysql_fetch|sqlite3\.OperationalError|PG::SyntaxError)"])
            elif "pwntools" in lower_code or "p32(" in lower_code or "p64(" in lower_code:
                tags.extend(["pwn", "buffer_overflow"])
                signatures.extend(["ELF", "checksec", "ROP"])
                expected_outcomes.extend([r"(?:\[\+\] Opening connection|\[\*\] Switching to interactive mode)"])
            elif "jwt" in lower_code or "bearer" in lower_code:
                tags.extend(["jwt", "token_forgery"])
                signatures.extend(["eyJ", "RS256", "none algorithm"])
                expected_outcomes.extend([r"(?:HTTP/1\.[01] 200|Welcome, admin|flag)"])
            else:
                tags.append("custom_exploit")
                signatures.append(challenge_title[:40])

            content_hash = hashlib.md5(f"{challenge_title}_{parameterized_code}".encode("utf-8")).hexdigest()[:8]
            pb_id = f"gen-{clean_category}-{content_hash}"

            playbook = PlaybookSchema(
                id=pb_id,
                category=clean_category,
                tags=list(set(tags)),
                trigger_signatures=list(set(signatures)),
                exploit_template=parameterized_code,
                expected_outcome_signatures=expected_outcomes,
                notes=f"Auto-synthesized from successful solve of challenge '{challenge_title}' (Flag: {flag[:12]}...).",
                source="auto_generated",
                confidence_score=0.3,
                times_used=0,
                success_rate=0.0,
                is_promoted=False,
                is_sanitized=True
            )

            self.save_playbook(playbook)
            logger.info(f"[Flywheel] Successfully synthesized auto_generated playbook `{pb_id}` from challenge {challenge_id}")
            return playbook
        except Exception as e:
            logger.error(f"[Flywheel] Failed to synthesize playbook: {e}", exc_info=True)
            return None

    def record_playbook_use(self, playbook_id: str, success: bool):
        """Update usage stats and auto-promote if confidence threshold reached."""
        pb = self.load_playbook(playbook_id)
        if not pb:
            return

        pb.n_applications += 1
        alpha = 0.3
        current_val = 1.0 if success else 0.0
        pb.success_rate = round((1 - alpha) * pb.success_rate + alpha * current_val, 2)

        if success:
            pb.confidence_score = min(1.0, round(pb.confidence_score + 0.2, 2))
        else:
            pb.confidence_score = max(0.1, round(pb.confidence_score - 0.1, 2))

        if pb.source_type in ["generated", "auto_generated"] and (pb.n_applications >= self.REUSE_PROMOTION_THRESHOLD or pb.confidence_score >= self.CONFIDENCE_PROMOTION_THRESHOLD):
            pb.is_promoted = True
            logger.info(f"[Flywheel] Auto-promoting playbook `{pb.id}` to primary searchable vault! (uses={pb.n_applications}, conf={pb.confidence_score})")

        self.save_playbook(pb)
        _broadcast_knowledge_event(
            "PLAYBOOK_STATS_UPDATED", pb.id, pb.category,
            {"is_promoted": pb.is_promoted, "confidence": pb.confidence_score, "n_applications": pb.n_applications}
        )

    async def ingest_writeup(self, text_or_markdown: str, category: str = "web", auto_approve: bool = False) -> Optional[PlaybookSchema]:
        """Ingestion pipeline: extract structured playbook from raw writeup text using fast provider tier."""
        from backend.providers.router import model_router
        from backend.knowledge.ingest_writeup import sanitize_playbook_content
        clean_category = category.lower() if category in CATEGORIES else "web"

        prompt = f"""
You are an expert CTF exploit extractor. Extract a structured, reusable exploit playbook from the following CTF write-up.
Return a STRICT JSON object with these exact keys:
{{
  "id": "short-kebab-case-identifier",
  "tags": ["tag1", "tag2", "tag3"],
  "trigger_signatures": ["signature1", "signature2", "error_message_or_technology"],
  "exploit_template": "parameterized Python/bash exploit code with {{TARGET_URL}}, {{PORT}}, etc.",
  "expected_outcome_signatures": ["regex_or_string_indicating_success"],
  "notes": "key insights on how this exploit works and bypass mechanisms"
}}

Writeup Content:
\"\"\"
{text_or_markdown[:4000]}
\"\"\"
"""
        try:
            resp = await model_router.route_and_generate(
                capability="fast_reasoning",
                prompt=prompt,
                system_prompt="You are a JSON-only CTF exploit synthesizer. Output pure JSON without markdown code fences."
            )
            raw = resp.text.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"^```\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            import json
            parsed = json.loads(raw)

            pb_id = parsed.get("id", f"ingested-{clean_category}-{hashlib.md5(text_or_markdown[:100].encode()).hexdigest()[:6]}")
            raw_template = parsed.get("exploit_template", "# No exploit template extracted")
            sanitized_template = sanitize_playbook_content(raw_template)
            sanitized_notes = sanitize_playbook_content(parsed.get("notes", "Ingested from CTF write-up"))

            playbook = PlaybookSchema(
                id=pb_id,
                category=clean_category,
                tags=parsed.get("tags", [clean_category]),
                trigger_signatures=parsed.get("trigger_signatures", []),
                exploit_template=sanitized_template,
                expected_outcome_signatures=parsed.get("expected_outcome_signatures", []),
                notes=sanitized_notes,
                source="ingested",
                confidence_score=0.9 if auto_approve else 0.5,
                times_used=0,
                success_rate=1.0,
                is_promoted=auto_approve,
                is_sanitized=True
            )

            target_cat = clean_category if auto_approve else "pending_review"
            target_dir = os.path.join(self.base_dir, target_cat)
            os.makedirs(target_dir, exist_ok=True)
            file_path = os.path.join(target_dir, f"{playbook.id}.yaml")
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(playbook.model_dump(by_alias=False), f, default_flow_style=False)

            if auto_approve:
                self._index_single_playbook(playbook)

            logger.info(f"Ingested write-up playbook `{pb_id}` to `{target_cat}`")
            return playbook
        except Exception as e:
            logger.error(f"Failed to ingest write-up: {e}")
            return None


playbook_vault = PlaybookVault()
