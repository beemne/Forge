"""Structural guard: the production database is unreachable without authorization.

Why this exists
---------------
FORGE lost production data because ``backend/config.py`` called
``load_dotenv(dotenv_path=".env", override=True)`` at import time. ``override=True``
let the ``.env`` value rewrite ``os.environ["DATABASE_URL"]`` *after* a test module
had pinned it to the isolated test database, so the suite ran against production
``forge.db`` and test ``tearDown``s deleted real rows.

The first fix moved the work to the call sites (pin DATABASE_URL, import
``backend.config`` to let the override happen, then pin again). That fixes the
modules that remember the idiom and nothing else: it depends on import ordering
inside every file, and it does not cover ``py tests/test_x.py`` or an IDE "run
file" run at all -- there, ``backend.agent_runtime``'s import-time singletons open
a database before the module's pin line is ever reached.

This module closes the hole by construction rather than by discipline:

* The production database's identity is read from the repository's own ``.env``,
  never from ``settings`` -- ``settings.DATABASE_URL`` is exactly the value an
  override would have changed, so it cannot be used to detect one.
* :func:`assert_database_permitted` refuses to build an engine for that database
  unless a production entrypoint has called :func:`authorize_production_database`.

No bypass
---------
There is deliberately no environment variable, flag, or config key that turns this
check off. Authorization is an in-process call made by a production entrypoint
(``backend/main.py``'s startup hook), so a process that never runs that hook --
every test run, however it was launched -- cannot open the production database
regardless of what its environment says. ``tests/test_database_isolation.py``
asserts that this module reads no environment variable at all, so the property
cannot erode quietly in a later edit.

A denial raises before ``create_engine`` is reached, so no connection is opened and
no entry is added to the engine cache: every subsequent call raises again. Callers
that swallow exceptions therefore do not get a working database, they get a factory
that keeps refusing.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Set, Tuple

from sqlalchemy.engine import make_url

logger = logging.getLogger("forge.database.guard")

# backend/database/guard.py -> backend/database -> backend -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Used when the repository's .env does not name a database. Mirrors the Settings
# default in backend/config.py, so the identity has a value even on a checkout with
# no .env at all.
DEFAULT_PRODUCTION_URL = "sqlite:///./forge.db"

# Set only by authorize_production_database(), called from production entrypoints.
_production_authorized = False

# Decisions already made, keyed by (url, cwd). get_engine() runs on the request path
# via get_db(), so the filesystem work below must not repeat for a URL already judged.
_permitted: Set[Tuple[str, str]] = set()


class DatabaseIsolationError(RuntimeError):
    """An engine was requested for the production database without authorization.

    A RuntimeError subclass, matching SchemaMigrationError in
    backend/database/session.py, so existing ``assertRaises(Exception)`` call sites
    keep working unchanged.
    """


def authorize_production_database() -> None:
    """Permit this process to open the production database.

    Called by production entrypoints only -- currently ``backend/main.py``'s startup
    hook, which both launchers (``backend/main.py`` and ``launch_forge.py``) run
    because both serve ``backend.main:app``.
    """
    global _production_authorized
    _production_authorized = True


def is_production_authorized() -> bool:
    """Whether a production entrypoint has authorized this process."""
    return _production_authorized


def _read_production_url() -> str:
    """DATABASE_URL exactly as written in the repository's .env.

    Parsed with ``dotenv_values`` rather than ``load_dotenv`` so that reading the
    file has no effect on ``os.environ``: this module must never influence the value
    it is checking. Falls back to the Settings default when .env is absent or does
    not set DATABASE_URL.
    """
    try:
        from dotenv import dotenv_values

        value = (dotenv_values(REPO_ROOT / ".env").get("DATABASE_URL") or "").strip()
        if value:
            return value
    except Exception:
        # python-dotenv missing, or .env unreadable. The default identity still
        # protects the database FORGE creates out of the box.
        logger.debug("Could not read .env for the production database identity", exc_info=True)
    return DEFAULT_PRODUCTION_URL


def _parse(url: str):
    """make_url() as a soft failure -- engine construction will report a bad URL itself."""
    try:
        return make_url(url)
    except Exception:
        return None


def _is_memory_database(database: Optional[str]) -> bool:
    """True for the spellings SQLAlchemy treats as in-memory rather than a file."""
    if not database:
        # `sqlite://` (no path) is a valid in-memory URL whose database is None.
        return True
    return database == ":memory:" or database.startswith("file::memory:")


def _sqlite_target(database: str) -> str:
    """Absolute, case-normalized path a SQLite file URL resolves to.

    ``os.path.realpath`` resolves a relative path against the current working
    directory -- which is what SQLAlchemy itself will do at connect time -- and
    normalizes symlinks, ``..`` and drive-letter case in one step.
    """
    return os.path.normcase(os.path.realpath(database))


def _production_targets() -> Set[str]:
    """Every file path the production database could be opened under.

    A relative URL such as ``sqlite:///./forge.db`` is resolved by SQLAlchemy against
    the current working directory, while ``.env`` lives at the repository root. Those
    agree when FORGE is launched from the root and disagree otherwise, so both anchors
    are treated as production -- comparing only one would leave the other's file
    openable without authorization.
    """
    parsed = _parse(_read_production_url())
    if parsed is None or parsed.get_backend_name() != "sqlite":
        return set()
    database = parsed.database
    if _is_memory_database(database):
        return set()
    if os.path.isabs(database):
        return {_sqlite_target(database)}
    return {
        _sqlite_target(os.path.join(os.getcwd(), database)),
        _sqlite_target(os.path.join(str(REPO_ROOT), database)),
    }


def _is_production_url(url: str) -> bool:
    """Whether this URL names the production database."""
    parsed = _parse(url)
    production_parsed = _parse(_read_production_url())
    if parsed is None or production_parsed is None:
        # Unparseable: not this guard's call to make, let create_engine report it.
        return False

    if parsed.get_backend_name() != production_parsed.get_backend_name():
        return False

    if parsed.get_backend_name() != "sqlite":
        # Non-file backends have no path to compare, so identity is the URL itself
        # (host/port/database), with the password excluded from the comparison.
        return _render(parsed) == _render(production_parsed)

    database = parsed.database
    if _is_memory_database(database):
        return False
    return _sqlite_target(database) in _production_targets()


def _render(parsed) -> str:
    """A stable, case-normalized string form of a URL for equality comparison."""
    try:
        return os.path.normcase(parsed.render_as_string(hide_password=True))
    except Exception:
        return os.path.normcase(str(parsed))


def assert_database_permitted(url: str) -> None:
    """Refuse to build an engine for the production database without authorization.

    Raises DatabaseIsolationError before any connection is made. Every other
    database -- the isolated test file, a temp-directory database, in-memory, an
    unrelated server -- is permitted, so this only ever forbids the one database
    whose loss is unrecoverable.
    """
    if _production_authorized:
        return

    key = (url, os.getcwd())
    if key in _permitted:
        return

    if _is_production_url(url):
        production = _read_production_url()
        raise DatabaseIsolationError(
            "Refusing to open the production database without authorization.\n"
            f"  requested:  {url}\n"
            f"  production: {production}  (from {REPO_ROOT / '.env'})\n"
            "This process never authorized the production database. Production "
            "entrypoints authorize it in backend/main.py's startup hook; a test run "
            "must point DATABASE_URL at the isolated database "
            "(sqlite:///./test_forge.db) instead.\n"
            "No engine was created and no connection was made."
        )

    _permitted.add(key)
