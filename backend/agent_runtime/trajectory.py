"""
FORGE Agent Runtime — trajectory store + local FTS5 recall (Steps 2 & 4).

`TrajectoryStore` persists every meaningful agent event to the ``trajectory_events``
table (the durable source of truth — never rely solely on in-memory history) and
streams it over the existing ``ws_manager`` (Step 13).

`TrajectorySearch` is a local SQLite **FTS5** index (in-memory, mirroring the
persistent table) — the same lightweight approach already proven by
``experience_memory``. It answers CTF-oriented cross-session recall queries
(previous commands, important observations, successful techniques, failed
approaches, prior sessions) WITHOUT shipping the whole database to the LLM.

No second database is introduced: the persistent store is the existing SQLite DB,
and the FTS index is an in-process mirror rebuilt from it.
"""

from __future__ import annotations

import re
import sqlite3
import logging
from datetime import datetime, timezone
from backend.utils.time import utcnow
from typing import Any, Dict, List, Optional

from backend.database.session import SessionLocal
from backend.database.models import TrajectoryEventModel, AgentSessionModel

logger = logging.getLogger("forge.agent_runtime.trajectory")


def _broadcast(event_type: str, payload: Dict[str, Any]) -> None:
    """Fire-and-forget WebSocket broadcast (Step 13). Never raises, never blocks."""
    try:
        import asyncio
        from backend.websocket.manager import ws_manager
        msg = {"event": "TRAJECTORY_EVENT", "trajectory_event": event_type}
        msg.update(payload)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(ws_manager.broadcast(msg))
        except RuntimeError:
            pass  # No running loop (sync/CLI/test) — skip silently.
    except Exception:
        pass


class TrajectoryStore:
    """Persist + read trajectory events. Owns sequence allocation per session."""

    # Cap very large captures so a single noisy command can't bloat the row/prompt.
    MAX_STDOUT = 20000
    MAX_STDERR = 8000

    def __init__(self, search: Optional["TrajectorySearch"] = None):
        self._search = search  # resolved lazily via the module singleton if None

    def _search_index(self):
        if self._search is not None:
            return self._search
        return trajectory_search

    def next_sequence(self, session_id: str, db=None) -> int:
        own = db is None
        db = db or SessionLocal()
        try:
            row = (db.query(TrajectoryEventModel.sequence)
                   .filter(TrajectoryEventModel.session_id == session_id)
                   .order_by(TrajectoryEventModel.sequence.desc()).first())
            return (row[0] + 1) if row and row[0] is not None else 1
        finally:
            if own:
                db.close()

    def record(
        self,
        *,
        session_id: str,
        event_type: str,
        run_id: Optional[str] = None,
        challenge_id: Optional[str] = None,
        agent_id: str = "orchestrator",
        sequence: Optional[int] = None,
        action_type: str = "",
        command: str = "",
        tool_name: str = "",
        stdout: str = "",
        stderr: str = "",
        exit_code: Optional[int] = None,
        duration_ms: float = 0.0,
        observation: Optional[Dict[str, Any]] = None,
        state_delta: Optional[Dict[str, Any]] = None,
        decision_summary: str = "",
        strategy: str = "",
        result: str = "",
        provider: str = "",
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        broadcast: bool = True,
    ) -> Optional[str]:
        """Persist one trajectory event; return its id. Also indexes it for FTS + broadcasts."""
        db = SessionLocal()
        try:
            if sequence is None:
                sequence = self.next_sequence(session_id, db=db)
            ev = TrajectoryEventModel(
                session_id=session_id, run_id=run_id, challenge_id=challenge_id,
                agent_id=agent_id, sequence=sequence, event_type=event_type,
                action_type=action_type, command=command[:8000] if command else "",
                tool_name=tool_name, stdout=(stdout or "")[: self.MAX_STDOUT],
                stderr=(stderr or "")[: self.MAX_STDERR], exit_code=exit_code,
                duration_ms=float(duration_ms or 0.0), observation=observation or {},
                state_delta=state_delta or {}, decision_summary=decision_summary or "",
                strategy=strategy or "", result=result or "", provider=provider or "",
                model=model or "", prompt_tokens=int(prompt_tokens or 0),
                completion_tokens=int(completion_tokens or 0),
            )
            db.add(ev)
            # Keep the session's last_sequence pointer in step (resume uses it).
            sess = db.query(AgentSessionModel).filter(AgentSessionModel.id == session_id).first()
            if sess and sequence > (sess.last_sequence or 0):
                sess.last_sequence = sequence
            db.commit()
            db.refresh(ev)
            ev_id = ev.id
            try:
                self._search_index().index_event(ev)
            except Exception:
                pass
            if broadcast:
                _broadcast(event_type, {
                    "session_id": session_id, "run_id": run_id, "challenge_id": challenge_id,
                    "agent_id": agent_id, "sequence": sequence, "action_type": action_type,
                    "command": (command or "")[:400], "result": result,
                    "summary": decision_summary or (observation or {}).get("summary", ""),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            return ev_id
        except Exception as e:
            logger.warning(f"[TrajectoryStore] record failed ({event_type}): {e}")
            db.rollback()
            return None
        finally:
            db.close()

    def get_events(self, session_id: str, limit: int = 0) -> List[TrajectoryEventModel]:
        db = SessionLocal()
        try:
            q = (db.query(TrajectoryEventModel)
                 .filter(TrajectoryEventModel.session_id == session_id)
                 .order_by(TrajectoryEventModel.sequence.asc()))
            if limit:
                q = q.limit(limit)
            return q.all()
        finally:
            db.close()

    def get_recent(self, session_id: str, n: int = 8,
                   event_types: Optional[List[str]] = None) -> List[TrajectoryEventModel]:
        db = SessionLocal()
        try:
            q = db.query(TrajectoryEventModel).filter(TrajectoryEventModel.session_id == session_id)
            if event_types:
                q = q.filter(TrajectoryEventModel.event_type.in_(event_types))
            rows = q.order_by(TrajectoryEventModel.sequence.desc()).limit(n).all()
            return list(reversed(rows))
        finally:
            db.close()

    def count(self, session_id: str) -> int:
        db = SessionLocal()
        try:
            return db.query(TrajectoryEventModel).filter(
                TrajectoryEventModel.session_id == session_id).count()
        finally:
            db.close()


class TrajectorySearch:
    """In-memory SQLite FTS5 mirror of trajectory events for fast cross-session recall."""

    _STOP = {"the", "and", "for", "http", "https", "com", "www", "with", "this", "that", "was", "are"}

    def __init__(self, auto_load: bool = True):
        self.db_conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_fts()
        if auto_load:
            self.reload_index()

    def _init_fts(self):
        with self.db_conn:
            self.db_conn.execute("DROP TABLE IF EXISTS trajectory_fts")
            self.db_conn.execute("""
                CREATE VIRTUAL TABLE trajectory_fts USING fts5(
                    id UNINDEXED,
                    session_id UNINDEXED,
                    challenge_id,
                    agent_id UNINDEXED,
                    event_type,
                    command,
                    output,
                    observation,
                    decision,
                    strategy,
                    result UNINDEXED,
                    sequence UNINDEXED,
                    created_at UNINDEXED
                )
            """)

    def reload_index(self):
        self._init_fts()
        db = SessionLocal()
        try:
            rows = db.query(TrajectoryEventModel).all()
            for ev in rows:
                self._index(ev)
            logger.info(f"[TrajectorySearch] Indexed {len(rows)} trajectory events into FTS.")
        except Exception as e:
            logger.debug(f"[TrajectorySearch] reload_index skipped: {e}")
        finally:
            db.close()

    def index_event(self, ev: TrajectoryEventModel):
        self._index(ev)

    def _index(self, ev: TrajectoryEventModel):
        obs = ev.observation or {}
        obs_text = obs.get("summary", "") if isinstance(obs, dict) else ""
        if isinstance(obs, dict):
            obs_text += " " + " ".join(str(x) for x in (obs.get("new_technologies") or []))
            obs_text += " " + " ".join(str(x) for x in (obs.get("new_vulnerabilities") or []))
        output = f"{ev.stdout or ''}\n{ev.stderr or ''}"[:4000]
        with self.db_conn:
            self.db_conn.execute("DELETE FROM trajectory_fts WHERE id = ?", (ev.id,))
            self.db_conn.execute(
                """INSERT INTO trajectory_fts(id, session_id, challenge_id, agent_id, event_type,
                       command, output, observation, decision, strategy, result, sequence, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ev.id, ev.session_id or "", ev.challenge_id or "", ev.agent_id or "",
                 ev.event_type or "", ev.command or "", output, obs_text.strip(),
                 ev.decision_summary or "", ev.strategy or "", ev.result or "",
                 int(ev.sequence or 0), (ev.created_at or utcnow()).isoformat()),
            )

    def search(
        self, query: str = "", *, challenge_id: Optional[str] = None,
        event_type: Optional[str] = None, exclude_session: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return the most relevant trajectory events (BM25-ranked) as dicts."""
        terms = re.findall(r"[A-Za-z0-9_]{2,}", query or "")
        terms = [t for t in terms if t.lower() not in self._STOP][:24]
        clauses, params = [], []
        if terms:
            clauses.append("trajectory_fts MATCH ?")
            params.append(" OR ".join(f'"{t}"' for t in terms))
        if challenge_id:
            clauses.append("challenge_id = ?")
            params.append(challenge_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        where = (" AND ".join(clauses)) if clauses else "1"
        order = "ORDER BY bm25(trajectory_fts)" if terms else "ORDER BY created_at DESC"
        sql = f"""SELECT id, session_id, challenge_id, event_type, command, output, observation,
                         decision, strategy, result, sequence, created_at
                  FROM trajectory_fts WHERE {where} {order} LIMIT ?"""
        params.append(max(1, top_k) * 3)
        try:
            cur = self.db_conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
        except Exception as e:
            logger.debug(f"[TrajectorySearch] search failed ('{query}'): {e}")
            return []
        cols = ["id", "session_id", "challenge_id", "event_type", "command", "output",
                "observation", "decision", "strategy", "result", "sequence", "created_at"]
        results = []
        for r in rows:
            if exclude_session and r[1] == exclude_session:
                continue
            results.append(dict(zip(cols, r)))
            if len(results) >= top_k:
                break
        return results


# Module singletons (mirror the experience_memory pattern). Indexed from the DB
# named by DATABASE_URL at import time; tests set that env var first.
trajectory_search = TrajectorySearch()
trajectory_store = TrajectoryStore(search=trajectory_search)
