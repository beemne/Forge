"""FORGE OS Keep-Awake Engine.
Prevents operating system sleep, hibernation, and display turn-off while CTF challenges or workflows are active.
Uses Windows Win32 API SetThreadExecutionState with graceful fallbacks.
"""

import sys
import logging

logger = logging.getLogger("forge.keep_awake")

# Win32 Execution State Flags
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_CONTINUOUS = 0x80000000

class KeepAwakeManager:
    """Manages system sleep prevention and display state retention across active tasks."""

    def __init__(self):
        self._active_holds: int = 0
        self._is_windows: bool = sys.platform.startswith("win")
        self._kernel32 = None
        
        if self._is_windows:
            try:
                import ctypes
                self._kernel32 = ctypes.windll.kernel32
            except Exception as e:
                logger.warning(f"[KeepAwake] Failed to load kernel32.dll: {e}")
                self._kernel32 = None

    @property
    def is_active(self) -> bool:
        return self._active_holds > 0

    @property
    def active_holds_count(self) -> int:
        return self._active_holds

    def acquire(self, reason: str = "Active CTF Challenge Execution") -> bool:
        """Acquires a keep-awake lock to prevent PC sleep and screen shutoff."""
        self._active_holds += 1
        if self._active_holds == 1:
            self._apply_state(prevent_sleep=True)
            logger.info(f"[KeepAwake] ⚡ Sleep prevention & display lock ENGAGED. Reason: {reason} (Holds: {self._active_holds})")
        else:
            logger.debug(f"[KeepAwake] Hold incremented. Reason: {reason} (Holds: {self._active_holds})")
        return True

    def release(self, reason: str = "Challenge Execution Finished") -> bool:
        """Releases a keep-awake lock. If holds drop to 0, system power defaults are restored."""
        if self._active_holds > 0:
            self._active_holds -= 1

        if self._active_holds == 0:
            self._apply_state(prevent_sleep=False)
            logger.info(f"[KeepAwake] 💤 Sleep prevention RELEASED. System power defaults restored. Reason: {reason}")
        else:
            logger.debug(f"[KeepAwake] Hold decremented. Reason: {reason} (Holds: {self._active_holds})")
        return True

    def _apply_state(self, prevent_sleep: bool):
        if not self._is_windows or not self._kernel32:
            return

        try:
            if prevent_sleep:
                # Continuous + System Required + Display Required
                flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
                res = self._kernel32.SetThreadExecutionState(flags)
                if res == 0:
                    logger.warning("[KeepAwake] SetThreadExecutionState returned 0 (failed to set keep-awake state).")
            else:
                # Reset to continuous normal operation
                flags = ES_CONTINUOUS
                self._kernel32.SetThreadExecutionState(flags)
        except Exception as e:
            logger.error(f"[KeepAwake] Error modifying thread execution state: {e}")

# Global Singleton
keep_awake_manager = KeepAwakeManager()
