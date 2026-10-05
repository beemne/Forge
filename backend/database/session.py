import logging
import os
from typing import Optional
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker
from backend.config import settings
from backend.database.guard import DatabaseIsolationError, assert_database_permitted
from backend.database.models import Base, SchemaVersionModel

logger = logging.getLogger("forge.database.session")

_engine_cache = {}


def _resolve_database_url() -> str:
    """The database URL this process will use.

    The environment is consulted first on purpose: tests/test_schema_migrations.py
    repoints DATABASE_URL per test (setUp/tearDown) at a temp-directory database, and
    settings.DATABASE_URL is fixed at import time so it cannot reflect that switch.
    """
    return os.getenv("DATABASE_URL", settings.DATABASE_URL)


# Database engine initialization (Supports SQLite out-of-the-box and PostgreSQL)
def get_engine(url: Optional[str] = None):
    current_url = url or _resolve_database_url()
    # Structural safety check, before the engine exists and before the cache is
    # consulted -- a cached production engine must not be handed out either. Raises
    # before create_engine(), so a denial opens no connection and caches nothing.
    assert_database_permitted(current_url)
    if current_url in _engine_cache:
        return _engine_cache[current_url]
    kwargs = {}
    if current_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(current_url, **kwargs)
    if current_url.startswith("sqlite"):
        # WAL lets concurrent swarm writers and API readers coexist; busy_timeout
        # makes a writer wait instead of failing with "database is locked".
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    _engine_cache[current_url] = engine
    return engine

def SessionLocal():
    eng = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=eng)()

# ============================================================================ #
# Schema migrations — a versioned list, deliberately NOT Alembic
#
# `Base.metadata.create_all()` only creates tables that don't exist yet; it never
# alters an existing one. So every column added to a model AFTER a database was
# first created needs an explicit ALTER TABLE for that database. That used to be
# 20 bare `try/except: pass` blocks, which swallowed a genuinely broken migration
# (locked db, typo'd type, malformed schema) exactly as quietly as the expected
# "column already exists" case — and left no record of what had been applied, so
# no database could tell you what schema it was actually at.
#
# Now: an ordered, numbered list, applied once each and recorded in the
# `schema_version` table.
#
# ORDERED AND APPEND-ONLY. Never renumber or reword an existing entry: live
# databases record version numbers, so editing a statement under an
# already-recorded number means the edit never runs anywhere. New migrations go
# at the end, with the next number.
# ============================================================================ #
MIGRATIONS = [
    (1, "ALTER TABLE challenges ADD COLUMN progress INTEGER DEFAULT 0"),
    (2, "ALTER TABLE challenges ADD COLUMN flag_status VARCHAR DEFAULT 'UNFOUND'"),
    (3, "ALTER TABLE challenges ADD COLUMN working_directory VARCHAR DEFAULT ''"),
    (4, "ALTER TABLE challenges ADD COLUMN platform_name VARCHAR DEFAULT ''"),
    (5, "ALTER TABLE challenges ADD COLUMN difficulty VARCHAR DEFAULT 'MEDIUM'"),
    (6, "ALTER TABLE challenges ADD COLUMN started_at TIMESTAMP"),
    (7, "ALTER TABLE challenges ADD COLUMN completed_at TIMESTAMP"),
    (8, "ALTER TABLE challenges ADD COLUMN duration_seconds INTEGER DEFAULT 0"),
    (9, "ALTER TABLE challenges ADD COLUMN mission_plan JSON"),
    (10, "ALTER TABLE challenges ADD COLUMN requires_root BOOLEAN DEFAULT 0"),
    (11, "ALTER TABLE findings ADD COLUMN severity VARCHAR DEFAULT 'HIGH'"),
    (12, "ALTER TABLE findings ADD COLUMN endpoint VARCHAR DEFAULT ''"),
    # Phase 2: environment-aware skill columns on experiences.
    (13, "ALTER TABLE experiences ADD COLUMN required_os VARCHAR DEFAULT 'any'"),
    (14, "ALTER TABLE experiences ADD COLUMN required_tools JSON"),
    (15, "ALTER TABLE experiences ADD COLUMN required_python_libs JSON"),
    # Phase 4.x hardening: persist swarm task coordination hints so the
    # capability/target gates survive checkpoint/resume (§5/§6).
    (16, "ALTER TABLE swarm_tasks ADD COLUMN required_capabilities JSON"),
    (17, "ALTER TABLE swarm_tasks ADD COLUMN target_type VARCHAR DEFAULT ''"),
    (18, "ALTER TABLE targets ADD COLUMN technologies JSON"),
    (19, "ALTER TABLE targets ADD COLUMN address_history JSON"),
    (20, "ALTER TABLE targets ADD COLUMN discovery_method VARCHAR DEFAULT 'FORGE Auto Ingest'"),
    (21, "ALTER TABLE challenges ADD COLUMN approval_mode VARCHAR DEFAULT NULL"),
    # Workstream B2: per-model discovery state (present/callable/last-probe).
    (22, "ALTER TABLE models ADD COLUMN present_in_catalog BOOLEAN DEFAULT 1"),
    (23, "ALTER TABLE models ADD COLUMN callable BOOLEAN DEFAULT 1"),
    (24, "ALTER TABLE models ADD COLUMN last_probe_ts FLOAT DEFAULT 0"),
]

# SQLite's exact wording for ADD COLUMN on a column that already exists
# (verified on this project's SQLite 3.50.4: "duplicate column name: b").
_DUPLICATE_COLUMN_MARKER = "duplicate column name"


class SchemaMigrationError(RuntimeError):
    """A migration failed for a reason OTHER than its column already existing.

    This propagates out of init_db() to stop startup. A database FORGE merely
    believes is correct is worse than a refused boot: continuing would surface
    the same problem later as a confusing column error far from its cause, in
    the middle of a run. Refusing to start names the migration and the reason.
    """


def _current_schema_version(conn) -> int:
    """Highest applied migration version, 0 when nothing has been recorded.

    A database created before this mechanism existed has no `schema_version`
    table at all, which is indistinguishable from "nothing applied yet" — both
    mean replay from the top. That replay is safe because the already-applied
    case is recognised per statement (see _run_schema_migrations).
    """
    try:
        row = conn.execute(text("SELECT MAX(version) FROM schema_version")).scalar()
    except Exception:
        return 0
    return int(row) if row is not None else 0


def _record_schema_version(conn, version: int) -> None:
    """Persist an applied migration so it is never attempted again.

    Inserted through the ORM table rather than a raw bound statement: CPython
    removed sqlite3's implicit datetime adapters in 3.14, so binding a
    `datetime` via text() would raise. SQLAlchemy's DateTime type performs the
    conversion itself.
    """
    conn.execute(SchemaVersionModel.__table__.insert().values(version=version))
    conn.commit()


def _run_schema_migrations(conn) -> None:
    """Apply every migration newer than the recorded version, in order."""
    current = _current_schema_version(conn)
    pending = [(v, sql) for v, sql in MIGRATIONS if v > current]
    if not pending:
        logger.debug("Schema already at version %s; nothing to apply", current)
        return
    logger.debug(
        "Applying %d schema migration(s): version %s -> %s",
        len(pending), current, pending[-1][0],
    )
    for version, statement in pending:
        try:
            conn.execute(text(statement))
            conn.commit()
        except Exception as exc:
            if _DUPLICATE_COLUMN_MARKER not in str(exc).lower():
                logger.error(
                    "Schema migration %s FAILED — FORGE will not start.\n"
                    "  statement: %s\n"
                    "  error: %s",
                    version, statement, exc,
                )
                raise SchemaMigrationError(
                    f"Schema migration {version} failed: {exc}\n"
                    f"  statement: {statement}\n"
                    f"Fix the database or the migration, then restart FORGE."
                ) from exc
            # Expected whenever the column is already there — including every
            # forge.db created before `schema_version` existed, where all 20
            # land here on first boot. The end state is already correct, so
            # record the version and move on. Also the normal path on a BRAND
            # NEW database, where create_all() has just created the column from
            # the current model, making the ALTER a no-op.
            logger.debug(
                "Schema migration %s already applied (column exists): %s",
                version, statement,
            )
        _record_schema_version(conn, version)


def init_db():
    # Resolved once and handed to get_engine(): the branch below decides whether to
    # apply the SQLite-flavoured migrations, so an independent second read here could
    # disagree with the database the engine was actually built for.
    current_url = _resolve_database_url()
    eng = get_engine(current_url)
    Base.metadata.create_all(bind=eng)
    # These statements are SQLite-flavoured column additions, and have always
    # been applied on SQLite only; PostgreSQL deployments take their schema from
    # create_all(). Unchanged by the move to versioned migrations.
    if current_url.startswith("sqlite"):
        with eng.connect() as conn:
            _run_schema_migrations(conn)


    # Phase 2: warm the in-memory FTS indexes AFTER create_all + migration. The
    # experience/trajectory singletons build their indexes at import time, which in
    # the app happens BEFORE this function runs (routes are imported before startup
    # calls init_db). On a pre-existing DB whose `experiences` table predates the
    # environment-aware columns, that first import-time index build fails on the
    # missing column and leaves an EMPTY index for the process lifetime. Reloading
    # here — once the schema is correct — repairs it. Lazy-imported + non-fatal so
    # this never affects fresh installs or non-app entrypoints.
    try:
        from backend.knowledge.experience_memory import experience_memory as _em
        _em.reload_index()
    except DatabaseIsolationError:
        # A guard denial is never downgraded to "skip the warm-up". Continuing with an
        # empty index that silently stands in for real data is the same failure shape as
        # the DATABASE_URL bug this guard exists to stop. Every other failure below
        # stays non-fatal by design.
        raise
    except Exception:
        pass
    try:
        from backend.agent_runtime.trajectory import trajectory_search as _ts
        _ts.reload_index()
    except DatabaseIsolationError:
        raise
    except Exception:
        pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
