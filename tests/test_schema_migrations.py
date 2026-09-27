"""
Tests for the lightweight schema-version migration runner
(backend/database/session.py: MIGRATIONS / _run_schema_migrations / init_db).

Three things the versioned migration list has to get right:

  1. A fresh database ends up with exactly one `schema_version` row per migration,
     numbered 1..N with no gaps.
  2. A database that predates this mechanism — every column already present (the
     old try/except loop applied them silently), no `schema_version` rows — is
     backfilled on the next boot without raising and without operator action.
  3. A migration that fails for a reason OTHER than "column already exists" stops
     startup loudly with a specific message, instead of being swallowed the way the
     old bare `except Exception: pass` swallowed it.

Everything runs against a throwaway SQLite file in a temp directory. No API key,
network, subprocess, or production database.
"""
import logging
import os
import shutil
import tempfile
import unittest

from sqlalchemy import text

# No module-level DATABASE_URL pin on purpose: setUp() points DATABASE_URL at a throwaway
# file per test and get_engine() re-reads the variable on every call, so the engine is built
# from that per-test value. backend/config.py no longer writes to os.environ (real environment
# variables win over .env), and backend/database/guard.py refuses the production forge.db
# outright, so this module cannot reach production even if setUp() were wrong -- setUp()
# asserts the redirect landed anyway, for the subtler case of landing on another file.
from backend.database import session as session_module
from backend.database.models import Base
from backend.database.session import (
    MIGRATIONS,
    SchemaMigrationError,
    get_engine,
    init_db,
)

# Kept as a literal on purpose: it is the canary for a migration appended to
# backend/database/session.py's MIGRATIONS list without updating this file. Deriving it
# from len(MIGRATIONS) would make the test agree with whatever it found, which is the
# opposite of what it is for.
MIGRATION_COUNT = 21
LOGGER_NAME = "forge.database.session"


class SchemaMigrationTestBase(unittest.TestCase):
    """Runs each test against a throwaway SQLite database file."""

    def setUp(self):
        self._original_url = os.environ.get("DATABASE_URL")
        self._tmpdir = tempfile.mkdtemp(prefix="forge_schema_test_")
        self.db_path = os.path.join(self._tmpdir, "schema_test.db")
        self.url = "sqlite:///" + self.db_path.replace("\\", "/")
        os.environ["DATABASE_URL"] = self.url

        # Prove the redirect landed rather than trusting it: without this check a mistake
        # here would quietly point every assertion below at the wrong database. (The guard
        # would refuse production outright; this catches landing on another file.)
        self.assertEqual(os.getenv("DATABASE_URL"), self.url)
        self.assertNotEqual(
            os.path.realpath("forge.db"), os.path.realpath(self.db_path),
            "the throwaway test database resolved onto the production forge.db path",
        )

        self.addCleanup(self._restore_environment)

    def _restore_environment(self):
        # Release the pooled connections before deleting the file: on Windows an
        # open handle makes rmtree fail, and WAL leaves -wal/-shm siblings open too.
        # The engine cache is keyed by URL, so dropping our entry stops it growing
        # across tests.
        try:
            get_engine().dispose()
        except Exception:
            pass
        session_module._engine_cache.pop(self.url, None)
        if self._original_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = self._original_url
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _recorded_versions(self):
        """Version numbers currently recorded in schema_version, ascending."""
        with get_engine().connect() as conn:
            return list(
                conn.execute(
                    text("SELECT version FROM schema_version ORDER BY version")
                ).scalars().all()
            )

    def _table_names(self):
        with get_engine().connect() as conn:
            return set(
                conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).scalars().all()
            )

    def _drop_schema_version_table(self):
        with get_engine().connect() as conn:
            conn.execute(text("DROP TABLE schema_version"))
            conn.commit()


class FreshDatabaseTests(SchemaMigrationTestBase):
    """Acceptance criterion 1 — a new database records every migration once."""

    def test_migration_list_is_contiguous_and_ordered(self):
        self.assertEqual(len(MIGRATIONS), MIGRATION_COUNT)
        self.assertEqual(
            [version for version, _ in MIGRATIONS], list(range(1, MIGRATION_COUNT + 1)),
            "MIGRATIONS must be numbered 1..N with no gaps, duplicates or reordering",
        )

    def test_init_db_records_every_migration_exactly_once(self):
        init_db()

        self.assertEqual(
            self._recorded_versions(), list(range(1, MIGRATION_COUNT + 1)),
        )
        # One row per version — no migration recorded twice.
        self.assertEqual(
            len(self._recorded_versions()), MIGRATION_COUNT,
        )
        self.assertIn("schema_version", self._table_names())

    def test_second_boot_reapplies_nothing(self):
        init_db()

        with self.assertLogs(LOGGER_NAME, level="DEBUG") as captured:
            init_db()

        self.assertEqual(
            self._recorded_versions(), list(range(1, MIGRATION_COUNT + 1)),
            "a second init_db() must not duplicate schema_version rows",
        )
        self.assertTrue(
            any("nothing to apply" in r.getMessage() for r in captured.records),
            "a fully-migrated database should report nothing pending",
        )

    def test_applied_at_is_populated(self):
        init_db()

        with get_engine().connect() as conn:
            applied = conn.execute(
                text("SELECT applied_at FROM schema_version ORDER BY version")
            ).scalars().all()

        self.assertEqual(len(applied), MIGRATION_COUNT)
        self.assertTrue(all(a is not None for a in applied))


class PreexistingDatabaseTests(SchemaMigrationTestBase):
    """Acceptance criterion 2 — backward compatibility with an old forge.db."""

    def _build_pre_change_database(self):
        """Build a database shaped like one created before this change.

        `create_all()` builds the tables from the CURRENT models, so every column is
        already present — precisely the state an old database reached by having
        those ALTER TABLEs silently applied. Dropping `schema_version` afterwards is
        what makes it faithful: no version has ever been recorded.
        """
        eng = get_engine()
        Base.metadata.create_all(bind=eng)
        self._drop_schema_version_table()
        self.assertNotIn("schema_version", self._table_names())

    def test_preexisting_database_is_backfilled_without_raising(self):
        self._build_pre_change_database()

        with self.assertLogs(LOGGER_NAME, level="DEBUG") as captured:
            init_db()   # must not raise — this is the common upgrade path

        self.assertEqual(
            self._recorded_versions(), list(range(1, MIGRATION_COUNT + 1)),
            "every already-present migration must still be recorded",
        )
        # Proves the duplicate-column branch — not the failure branch — handled every
        # migration, rather than the runner silently skipping them.
        already_present = [
            r for r in captured.records if "already applied" in r.getMessage()
        ]
        self.assertEqual(len(already_present), MIGRATION_COUNT)
        self.assertEqual(
            [r for r in captured.records if r.levelno >= logging.ERROR], [],
            "an already-migrated database must not log any errors",
        )

    def test_missing_schema_version_table_reads_as_version_zero(self):
        """The 'table does not exist yet' path, before create_all() recreates it."""
        eng = get_engine()
        Base.metadata.create_all(bind=eng)
        self._drop_schema_version_table()

        with eng.connect() as conn:
            self.assertEqual(session_module._current_schema_version(conn), 0)

    def test_resulting_schema_is_correct_after_backfill(self):
        self._build_pre_change_database()
        init_db()

        with get_engine().connect() as conn:
            columns = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(challenges)")).fetchall()
            }

        self.assertIn("progress", columns)
        self.assertIn("flag_status", columns)
        self.assertIn("mission_plan", columns)


class BrokenMigrationTests(SchemaMigrationTestBase):
    """Acceptance criterion 3 — real failures are distinguished and stop startup."""

    def _install_migrations(self, migrations):
        """Swap in a COPY of MIGRATIONS for one test, restoring it afterwards."""
        original = session_module.MIGRATIONS
        self.addCleanup(setattr, session_module, "MIGRATIONS", original)
        session_module.MIGRATIONS = migrations

    def test_genuine_failure_stops_startup_with_a_specific_error(self):
        broken_version = MIGRATION_COUNT + 1
        broken_statement = "ALTER TABLE challenges ADD COLUMN this is not valid sql"
        self._install_migrations(list(MIGRATIONS) + [(broken_version, broken_statement)])

        with self.assertLogs(LOGGER_NAME, level="DEBUG") as captured:
            with self.assertRaises(SchemaMigrationError) as ctx:
                init_db()

        message = str(ctx.exception)
        self.assertIn(str(broken_version), message)   # names the migration
        self.assertIn(broken_statement, message)      # names the statement

        errors = [r for r in captured.records if r.levelno >= logging.ERROR]
        self.assertEqual(len(errors), 1, "exactly one real failure should be logged")
        self.assertIn(broken_statement, errors[0].getMessage())

        # Every real migration still counts as applied; the broken one is NOT
        # recorded, so the next boot retries it instead of skipping over it.
        self.assertEqual(
            self._recorded_versions(), list(range(1, MIGRATION_COUNT + 1)),
        )

    def test_already_applied_column_is_recorded_not_raised(self):
        """The distinction the old `except Exception: pass` could not make.

        Same shape as the failure above — a migration whose column is already
        present must be recorded and passed over, not treated as an error.
        """
        self._install_migrations(
            list(MIGRATIONS)
            + [(MIGRATION_COUNT + 1, "ALTER TABLE challenges ADD COLUMN progress INTEGER DEFAULT 0")]
        )

        init_db()   # `progress` exists, so this is the expected branch

        self.assertEqual(
            self._recorded_versions(), list(range(1, MIGRATION_COUNT + 2)),
        )


if __name__ == "__main__":
    unittest.main()
