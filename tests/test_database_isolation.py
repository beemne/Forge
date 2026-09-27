"""Proofs that no test process can reach the production database.

The incident these exist for: backend/config.py ran load_dotenv(override=True) at import
time, which rewrote os.environ["DATABASE_URL"] to the production value from .env *after* a
test module had pinned it to the isolated database. The suite then ran against production
forge.db and test tearDowns deleted real rows.

Every proof that depends on import ordering runs in a CHILD process, because this module
has already imported backend by the time it executes and so cannot reproduce an ordering
bug in its own interpreter. The two defining cases:

  * env-set-then-import resolves to the test database   (config.py no longer overrides),
  * a module that forgot to pin at all is REFUSED rather than pointed at production
    (backend/database/guard.py).

Nothing here writes to forge.db. The refusal happens before create_engine() is reached.
"""

import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.database.guard import (  # noqa: E402
    DatabaseIsolationError,
    _is_production_url,
    assert_database_permitted,
)

ROOT = Path(__file__).resolve().parent.parent
TEST_URL = "sqlite:///./test_forge.db"
PRODUCTION_URL = "sqlite:///./forge.db"
GUARD_SOURCE = ROOT / "backend" / "database" / "guard.py"

# Environment variables a future "skip the guard" switch might plausibly be called. None of
# them may change the outcome; they exist here so the day someone adds one, this fails.
BYPASS_NAMES = {
    "FORGE_ALLOW_PROD_DB": "1",
    "FORGE_ALLOW_PRODUCTION_DB": "1",
    "FORGE_ALLOW_PROD_DATABASE": "1",
    "FORGE_SKIP_DB_GUARD": "1",
    "FORGE_DB_GUARD": "off",
    "FORGE_DISABLE_DB_GUARD": "1",
    "FORGE_TEST_MODE": "0",
}

# A module that never pins DATABASE_URL: exactly the shape of a test written by someone who
# did not know about the old idiom. Under discovery it must be refused, not pointed at prod.
PROBE = '''"""Simulates a test module that forgot to pin DATABASE_URL."""
import unittest

from backend.database.guard import DatabaseIsolationError
from backend.database.session import get_engine

class ProductionDatabaseProbe(unittest.TestCase):
    def test_production_database_is_refused(self):
        with self.assertRaises(DatabaseIsolationError):
            get_engine()

    def test_no_connection_was_possible_after_the_refusal(self):
        # The refusal must be repeatable: nothing was cached, so a retry refuses too.
        for _ in range(3):
            with self.assertRaises(DatabaseIsolationError):
                get_engine()
'''

def _child_env(strip_database_url=True, extra=None):
    env = os.environ.copy()
    if strip_database_url:
        # The whole point: a process with NO test pin resolves .env, i.e. production.
        env.pop("DATABASE_URL", None)
    if extra:
        env.update(extra)
    return env

def _run(argv, env, timeout=180):
    return subprocess.run(
        argv, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout
    )

def _run_discovery(tmpdir, env, timeout=180):
    """Run a real `-m unittest discover` over tmpdir, from the repository root.

    A real runner matters: it is what a per-module run is, and it is what the four
    heuristic signals in an earlier design failed to detect.
    """
    return _run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(tmpdir), "-t", str(tmpdir)],
        env,
        timeout=timeout,
    )

def _write_probe(tmpdir):
    (Path(tmpdir) / "test_forge_probe.py").write_text(PROBE, encoding="utf-8")

class ConfigPrecedenceTests(unittest.TestCase):
    """backend/config.py must let the process environment win over .env."""

    def test_process_env_wins_over_env_file(self):
        """The exact failure mode, inverted: pin first, then import backend.config."""
        child = (
            "import os\n"
            f"os.environ['DATABASE_URL'] = {TEST_URL!r}\n"
            "import backend.config\n"
            "print('RESOLVED=' + backend.config.settings.DATABASE_URL)\n"
        )
        result = _run([sys.executable, "-c", child], _child_env(strip_database_url=False))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"RESOLVED={TEST_URL}", result.stdout)

    def test_env_file_is_still_read_when_env_unset(self):
        """Dropping load_dotenv must not stop .env from being loaded at all."""
        child = "import backend.config\nprint('RESOLVED=' + backend.config.settings.DATABASE_URL)\n"
        result = _run([sys.executable, "-c", child], _child_env())
        self.assertEqual(result.returncode, 0, result.stderr)
        # .env names production; with no override from the environment that is what
        # settings resolves to, and the guard's refusal below is what makes it harmless.
        self.assertIn("RESOLVED=" + PRODUCTION_URL, result.stdout)

    def test_env_file_read_has_no_side_effect_on_os_environ(self):
        """Reading .env must not inject its keys into the child's environment."""
        child = (
            "import os, backend.config\n"
            "print('ENV_DB=' + repr(os.environ.get('DATABASE_URL')))\n"
        )
        result = _run([sys.executable, "-c", child], _child_env())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ENV_DB=None", result.stdout)

class GuardRefusalTests(unittest.TestCase):
    """A module that forgot to pin must be refused, however it was launched."""

    def test_guard_refuses_production_under_real_discovery(self):
        with tempfile.TemporaryDirectory(prefix="forge_guard_probe_") as tmpdir:
            _write_probe(tmpdir)
            result = _run_discovery(tmpdir, _child_env())
        self.assertEqual(
            result.returncode,
            0,
            "the probe failed to observe a refusal -- a module with no DATABASE_URL pin "
            f"reached production.\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )
        self.assertIn("OK", result.stderr + result.stdout)

    def test_no_environment_variable_can_disable_the_guard(self):
        with tempfile.TemporaryDirectory(prefix="forge_guard_probe_") as tmpdir:
            _write_probe(tmpdir)
            result = _run_discovery(tmpdir, _child_env(extra=BYPASS_NAMES))
        self.assertEqual(
            result.returncode,
            0,
            "an environment variable disabled the guard.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

    def test_production_database_is_refused_in_process(self):
        absolute = "sqlite:///" + str(ROOT / "forge.db").replace("\\", "/")
        with self.assertRaises(DatabaseIsolationError):
            assert_database_permitted(absolute)
        with self.assertRaises(DatabaseIsolationError):
            assert_database_permitted(PRODUCTION_URL)

class GuardAuthorizationTests(unittest.TestCase):
    """The guard must not be a blanket ban: production still works when authorized."""

    def test_authorized_production_database_is_permitted(self):
        child = (
            "from backend.database.guard import authorize_production_database\n"
            "from backend.database.session import get_engine\n"
            "authorize_production_database()\n"
            "engine = get_engine()\n"
            "print('ENGINE=' + str(engine.url))\n"
            "engine.dispose()\n"
        )
        result = _run([sys.executable, "-c", child], _child_env())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ENGINE=" + PRODUCTION_URL, result.stdout)

    def test_authorization_is_not_inherited(self):
        """Authorization is per process: a child gets no grant from its parent."""
        with tempfile.TemporaryDirectory(prefix="forge_guard_probe_") as tmpdir:
            _write_probe(tmpdir)
            result = _run_discovery(tmpdir, _child_env(extra={"FORGE_ALLOW_PROD_DB": "1"}))
        self.assertEqual(result.returncode, 0, result.stderr)

class ProductionIdentityTests(unittest.TestCase):
    """Identity is a path comparison, so every spelling of forge.db is caught."""

    def setUp(self):
        # Relative spellings resolve against the working directory, so pin it: the
        # predicate must be judged exactly as the server would see it.
        self._cwd = os.getcwd()
        os.chdir(ROOT)

    def tearDown(self):
        os.chdir(self._cwd)

    def test_production_spellings_are_recognized(self):
        absolute = "sqlite:///" + str(ROOT / "forge.db").replace("\\", "/")
        for url in (PRODUCTION_URL, absolute, "sqlite:///forge.db"):
            with self.subTest(url=url):
                self.assertTrue(_is_production_url(url))

    def test_non_production_databases_are_not_production(self):
        temp_db = "sqlite:///" + os.path.join(
            tempfile.gettempdir(), "forge_schema_test_x", "forge_schema_test.db"
        ).replace("\\", "/")
        for url in (TEST_URL, temp_db, "sqlite://", "sqlite:///:memory:"):
            with self.subTest(url=url):
                self.assertFalse(_is_production_url(url))

    def test_test_database_engine_is_permitted_and_correct(self):
        from backend.database.session import get_engine

        engine = get_engine()
        self.assertTrue(str(engine.url).endswith("test_forge.db"), str(engine.url))

class GuardIsNotBypassableByConstructionTests(unittest.TestCase):
    """The no-bypass property must not be erodable by a later edit."""

    def test_guard_reads_no_environment_variable(self):
        tree = ast.parse(GUARD_SOURCE.read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "environ":
                offenders.append((node.lineno, "os.environ"))
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "getenv":
                    offenders.append((node.lineno, "getenv()"))
                elif isinstance(func, ast.Name) and func.id == "getenv":
                    offenders.append((node.lineno, "getenv()"))
        self.assertEqual(
            offenders,
            [],
            "backend/database/guard.py must decide without reading the environment, so "
            f"that no variable can switch it off. Found: {offenders}",
        )

    def test_production_authorization_starts_false_and_is_set_only_in_the_function(self):
        from backend.database import guard

        # Nothing in this process authorized production.
        self.assertFalse(guard.is_production_authorized())
        tree = ast.parse(GUARD_SOURCE.read_text(encoding="utf-8"))

        # The flag IS assigned at module level -- that assignment is its initialisation.
        # What must not happen there is a *grant*: the only module-level assignment
        # allowed is a literal False, so no statement outside the function can leave the
        # guard open. (An earlier version of this test banned module-level assignment
        # outright and so failed on the initialisation it was meant to permit.)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            else:
                continue
            if "_production_authorized" not in targets:
                continue
            self.assertIsInstance(
                node.value,
                ast.Constant,
                "a module-level assignment to the authorization flag must be a literal",
            )
            self.assertIs(
                node.value.value,
                False,
                "no module-level statement may grant production authorization; the only "
                "writer of True must be authorize_production_database()",
            )

        # And that function grants with a literal True -- not a value read from the
        # environment, a config key, or a parameter.
        granter = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "authorize_production_database"
        )
        grants = [
            node
            for node in ast.walk(granter)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_production_authorized"
                for t in node.targets
            )
        ]
        self.assertEqual(
            len(grants), 1, "authorize_production_database() must set the flag exactly once"
        )
        self.assertIsInstance(grants[0].value, ast.Constant)
        self.assertIs(grants[0].value.value, True)

if __name__ == "__main__":
    unittest.main()
