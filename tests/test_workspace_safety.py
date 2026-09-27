"""Regression tests for challenge working-directory deletion safety.

Guards against the historic bug where a challenge whose working_directory was
"." (resolving to the FORGE project root) caused DELETE /challenges to run
shutil.rmtree() on the entire project. See backend.utils.workspace.
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.utils.workspace import (
    CTF_WORKSPACE_ROOT,
    PROJECT_ROOT,
    is_deletable_working_dir,
    resolve_safe_working_dir,
)

def test_project_root_is_not_deletable():
    assert is_deletable_working_dir(PROJECT_ROOT) is False

def test_dot_resolves_to_project_root_and_is_refused():
    # "." is exactly what poisoned the old rows: abspath(".") == project root.
    assert is_deletable_working_dir(os.path.abspath(".")) is False

def test_home_and_system_roots_not_deletable():
    assert is_deletable_working_dir(os.path.abspath(os.path.expanduser("~"))) is False
    assert is_deletable_working_dir(os.path.abspath(os.sep)) is False

def test_workspace_root_itself_not_deletable():
    assert is_deletable_working_dir(CTF_WORKSPACE_ROOT) is False

def test_sibling_prefix_trick_not_deletable():
    # A directory that merely shares a name prefix with the workspace root
    # must not be treated as inside it.
    assert is_deletable_working_dir(CTF_WORKSPACE_ROOT + "-evil") is False

def test_legit_challenge_dir_is_deletable():
    legit = os.path.join(CTF_WORKSPACE_ROOT, "PicoCTF", "WEB", "EASY", "some_challenge")
    assert is_deletable_working_dir(legit) is True

def test_resolve_safe_working_dir_rejects_dot():
    resolved = resolve_safe_working_dir(".", "abc123", "WEB", "Demo")
    assert resolved != os.path.abspath(".")
    assert is_deletable_working_dir(resolved) is True

def test_resolve_safe_working_dir_keeps_valid_path():
    legit = os.path.join(CTF_WORKSPACE_ROOT, "PicoCTF", "WEB", "EASY", "keepme")
    resolved = resolve_safe_working_dir(legit, "abc123", "WEB", "keepme")
    assert resolved == os.path.abspath(legit)

def test_safe_delete_refuses_project_root(tmp_path):
    # End-to-end: the routes helper must not remove a sentinel placed at the
    # project root when handed ".".
    from backend.api.routes import challenges as routes

    sentinel = os.path.join(PROJECT_ROOT, ".delete_guard_sentinel")
    with open(sentinel, "w") as f:
        f.write("do not delete")
    try:
        routes._safe_delete_working_dir(".")
        routes._safe_delete_working_dir(PROJECT_ROOT)
        assert os.path.exists(sentinel), "project root sentinel was deleted!"
    finally:
        if os.path.exists(sentinel):
            os.remove(sentinel)

if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
