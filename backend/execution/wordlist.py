"""
Environment-aware wordlist resolver for the FORGE execution layer.

Agents request ``web_fuzzing`` or ``directory_enumeration``; this module
resolves the best available wordlist for the current OS without any caller
knowing where the file lives.

Resolution order:
1. FORGE_WORDLIST env-var (user override)
2. Workspace-local wordlist (workspaces/common.txt)
3. OS-detected standard paths (Kali / Parrot / generic Linux, then Windows)
4. Generated minimal fallback — always works, no external dependency

The generated fallback is written once and reused; it is a small but
functionally useful list for CTF-style web enumeration.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("forge.execution.wordlist")

# ── Known system wordlist paths by OS ────────────────────────────────────── #

_LINUX_CANDIDATES = [
    "/usr/share/seclists/Discovery/Web-Content/common.txt",
    "/usr/share/wordlists/seclists/Discovery/Web-Content/common.txt",
    "/usr/share/wordlists/dirb/common.txt",
    "/usr/share/dirb/wordlists/common.txt",
    "/opt/SecLists/Discovery/Web-Content/common.txt",
]

_WINDOWS_CANDIDATES = [
    r"C:\tools\wordlists\common.txt",
    r"C:\SecLists\Discovery\Web-Content\common.txt",
]

_FALLBACK_WORDS = [
    "admin", "login", "api", "v1", "v2", "static", "assets",
    "upload", "uploads", "backup", "backups", "config", "configs",
    "flag", "flags", "secret", "secrets", "token", "tokens",
    "dashboard", "panel", "manage", "manager", "console",
    "index.php", "index.html", "robots.txt", ".git", ".env",
    "server-status", "server-info", "phpinfo.php",
    "wp-admin", "wp-login.php", "xmlrpc.php",
    "test", "debug", "dev", "staging", "old",
]


class WordlistResolver:
    """
    Resolves the best available wordlist path without embedding OS assumptions
    into the callers.
    """

    def __init__(self, workspace_root: Optional[str] = None) -> None:
        self._workspace_root = workspace_root or os.path.join(
            os.getcwd(), "workspaces"
        )

    def resolve(self, kind: str = "web_common") -> str:
        """
        Return an absolute path to a usable wordlist.

        Always returns a path; generates the fallback wordlist if nothing else
        is found.  Callers do not need to handle None.
        """
        # 1. Env-var override (highest priority)
        env_path = os.environ.get("FORGE_WORDLIST", "").strip()
        if env_path and os.path.isfile(env_path):
            logger.debug(f"[WordlistResolver] using FORGE_WORDLIST: {env_path}")
            return env_path

        # 2. Workspace-local wordlist
        local = os.path.join(self._workspace_root, "common.txt")
        if os.path.isfile(local):
            logger.debug(f"[WordlistResolver] using workspace wordlist: {local}")
            return local

        # 3. OS-detected system paths
        import sys
        candidates = _WINDOWS_CANDIDATES if sys.platform == "win32" else _LINUX_CANDIDATES
        for path in candidates:
            if os.path.isfile(path):
                logger.debug(f"[WordlistResolver] using system wordlist: {path}")
                return path

        # 4. Generated fallback — write it once next to the workspace
        return self._ensure_fallback()

    def _ensure_fallback(self) -> str:
        fallback = os.path.join(self._workspace_root, "common.txt")
        os.makedirs(os.path.dirname(fallback), exist_ok=True)
        if not os.path.isfile(fallback):
            with open(fallback, "w", encoding="utf-8") as fh:
                fh.write("\n".join(_FALLBACK_WORDS) + "\n")
            logger.info(f"[WordlistResolver] generated fallback wordlist: {fallback}")
        return fallback


# Module-level singleton.
wordlist_resolver = WordlistResolver()
