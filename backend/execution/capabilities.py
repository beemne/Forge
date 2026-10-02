"""
Capability discovery, alternatives, and resolution (Phase 4.x §12–§16).

FORGE agents should reason about *capabilities* ("I need OCR", "I need to scan a
network") and let the execution system decide *which concrete provider* fulfils
them ("tesseract", "nmap") and *whether it is even possible here*.  Before Phase
4.x a missing tool produced the same error on every retry until the mission budget
was gone; this module replaces that with a single deterministic decision:

    capability → discover → available? ── yes → use provider
                                       └─ no → alternative? ── yes → use alternative
                                                            └─ no → acquirable? ─ yes → request acquisition
                                                                                └─ no → BLOCKED → replan

Availability is discovered from the ACTUAL environment (``shutil.which`` for CLI
tools, ``importlib`` for Python libraries, and the execution backend's capability
report), never hardcoded.  Nothing here installs or executes anything — acquisition
is planned in :mod:`backend.execution.acquisition` and gated by the PrivilegeManager.
"""
from __future__ import annotations

import importlib.util
import logging
import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("forge.execution.capabilities")

_IS_WINDOWS = sys.platform == "win32"

# ── Resolution status ───────────────────────────────────────────────────────── #
AVAILABLE = "AVAILABLE"                       # a preferred provider is installed
ALTERNATIVE_AVAILABLE = "ALTERNATIVE_AVAILABLE"  # a non-preferred provider is installed
ACQUIRABLE = "ACQUIRABLE"                     # nothing installed, but can be acquired (needs approval)
BLOCKED = "BLOCKED"                           # unavailable and not acquirable here
UNKNOWN = "UNKNOWN"


class ProviderKind(str, Enum):
    TOOL = "tool"        # a CLI binary on PATH
    PYLIB = "pylib"      # an importable Python module
    BUILTIN = "builtin"  # a feature the execution backend itself provides
    CLOUD = "cloud"      # a hosted API provider, "present" when its key is configured


@dataclass(frozen=True)
class ProviderSpec:
    """One concrete way to fulfil a capability."""
    name: str
    kind: ProviderKind
    binary: str = ""                 # for TOOL: the executable to look up
    module: str = ""                 # for PYLIB: the import name
    requires_tool: str = ""          # a TOOL dependency a PYLIB needs (e.g. pytesseract→tesseract)
    install_recipe: str = ""
    privilege: str = "SAFE"          # SAFE | PRIVILEGED | DANGEROUS (acquisition routing)


@dataclass
class Capability:
    """The discovered state of one capability in THIS environment."""
    name: str
    status: str = UNKNOWN
    available: bool = False
    provider: str = ""                       # chosen provider name (to use or to acquire)
    providers_checked: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)   # other installed providers
    backend_supported: bool = True
    installable: bool = False
    acquisition_possible: bool = False
    recommended_action: str = "execute"      # execute | request_acquisition | replan
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "status": self.status, "available": self.available,
            "provider": self.provider, "providers_checked": list(self.providers_checked),
            "alternatives": list(self.alternatives), "backend_supported": self.backend_supported,
            "installable": self.installable, "acquisition_possible": self.acquisition_possible,
            "recommended_action": self.recommended_action, "reason": self.reason,
        }

    def observability_line(self) -> str:
        return (f"CAPABILITY_{self.status} capability={self.name} "
                f"provider={self.provider or '-'} "
                f"providers_checked={self.providers_checked} "
                f"alternatives={self.alternatives} "
                f"acquisition_possible={self.acquisition_possible} "
                f"recommended_action={self.recommended_action}")


def _P(name, kind, **kw) -> ProviderSpec:
    return ProviderSpec(name=name, kind=kind, **kw)


# The registry expresses capability→providers in PREFERENCE order.  A provider is
# either a CLI tool, a Python library, or a built-in backend feature.  Recipes are
# advisory strings used only for auditable acquisition planning (never auto-run).
CAPABILITY_REGISTRY: Dict[str, List[ProviderSpec]] = {
    "network_scanning": [
        _P("nmap", ProviderKind.TOOL, binary="nmap", install_recipe="apt-get install nmap", privilege="PRIVILEGED"),
        _P("rustscan", ProviderKind.TOOL, binary="rustscan", install_recipe="cargo install rustscan"),
        _P("masscan", ProviderKind.TOOL, binary="masscan", install_recipe="apt-get install masscan", privilege="PRIVILEGED"),
    ],
    "web_fuzzing": [
        _P("ffuf", ProviderKind.TOOL, binary="ffuf", install_recipe="apt-get install ffuf"),
        _P("gobuster", ProviderKind.TOOL, binary="gobuster", install_recipe="apt-get install gobuster"),
        _P("feroxbuster", ProviderKind.TOOL, binary="feroxbuster", install_recipe="apt-get install feroxbuster"),
        _P("dirsearch", ProviderKind.TOOL, binary="dirsearch", install_recipe="pip install dirsearch"),
    ],
    "http_request": [
        _P("curl", ProviderKind.TOOL, binary="curl", install_recipe="apt-get install curl"),
        _P("wget", ProviderKind.TOOL, binary="wget", install_recipe="apt-get install wget"),
        _P("requests", ProviderKind.PYLIB, module="requests", install_recipe="pip install requests"),
        _P("urllib", ProviderKind.BUILTIN),   # python stdlib, always present with python
    ],
    "interactive_tcp": [
        _P("pwntools", ProviderKind.PYLIB, module="pwn", install_recipe="pip install pwntools"),
        _P("ncat", ProviderKind.TOOL, binary="ncat", install_recipe="apt-get install nmap"),
        _P("nc", ProviderKind.TOOL, binary="nc", install_recipe="apt-get install netcat-openbsd"),
        _P("socat", ProviderKind.TOOL, binary="socat", install_recipe="apt-get install socat"),
        _P("python_socket", ProviderKind.BUILTIN),   # python stdlib socket
    ],
    "interactive_stdin": [
        _P("execution_backend", ProviderKind.BUILTIN),   # LocalBackend scripted-stdin + InteractiveSession
    ],
    "interactive_execution": [
        _P("interactive_manager", ProviderKind.BUILTIN),  # InteractiveSessionManager persistent dialogue
    ],
    "pty": [
        _P("posix_pty", ProviderKind.BUILTIN),           # POSIX only
    ],
    "ocr": [
        # Gemini vision is the preferred OCR provider when a GEMINI_API_KEY is
        # configured: it needs no local binary (killing the recurring
        # "tesseract/pytesseract not installed" dead-ends), reads text directly
        # from the image bytes, and runs on the same multi-key Gemini pool the
        # rest of FORGE already uses. Local providers remain as offline fallbacks.
        _P("gemini", ProviderKind.CLOUD),
        _P("tesseract", ProviderKind.TOOL, binary="tesseract", install_recipe="apt-get install tesseract-ocr"),
        _P("pytesseract", ProviderKind.PYLIB, module="pytesseract", requires_tool="tesseract", install_recipe="pip install pytesseract"),
        _P("easyocr", ProviderKind.PYLIB, module="easyocr", install_recipe="pip install easyocr"),
    ],
    "image_processing": [
        _P("PIL", ProviderKind.PYLIB, module="PIL", install_recipe="pip install pillow"),
        _P("opencv", ProviderKind.PYLIB, module="cv2", install_recipe="pip install opencv-python"),
    ],
    "python_exec": [
        _P("python", ProviderKind.TOOL, binary="python"),
        _P("python3", ProviderKind.TOOL, binary="python3"),
    ],
    "pwntools": [
        _P("pwntools", ProviderKind.PYLIB, module="pwn", install_recipe="pip install pwntools"),
    ],
    "binary_analysis": [
        _P("gdb", ProviderKind.TOOL, binary="gdb", install_recipe="apt-get install gdb"),
        _P("radare2", ProviderKind.TOOL, binary="r2", install_recipe="apt-get install radare2"),
        _P("objdump", ProviderKind.TOOL, binary="objdump", install_recipe="apt-get install binutils"),
    ],
    "steg_analysis": [
        _P("binwalk", ProviderKind.TOOL, binary="binwalk", install_recipe="apt-get install binwalk"),
        _P("zsteg", ProviderKind.TOOL, binary="zsteg", install_recipe="gem install zsteg"),
        _P("steghide", ProviderKind.TOOL, binary="steghide", install_recipe="apt-get install steghide"),
        _P("exiftool", ProviderKind.TOOL, binary="exiftool", install_recipe="apt-get install libimage-exiftool-perl"),
    ],
    "packet_analysis": [
        _P("tshark", ProviderKind.TOOL, binary="tshark", install_recipe="apt-get install tshark"),
        _P("tcpdump", ProviderKind.TOOL, binary="tcpdump", install_recipe="apt-get install tcpdump", privilege="PRIVILEGED"),
        _P("scapy", ProviderKind.PYLIB, module="scapy", install_recipe="pip install scapy"),
    ],
    "password_cracking": [
        _P("john", ProviderKind.TOOL, binary="john", install_recipe="apt-get install john"),
        _P("hashcat", ProviderKind.TOOL, binary="hashcat", install_recipe="apt-get install hashcat"),
        _P("hydra", ProviderKind.TOOL, binary="hydra", install_recipe="apt-get install hydra"),
    ],
    "archive_extract": [
        _P("7z", ProviderKind.TOOL, binary="7z", install_recipe="apt-get install p7zip-full"),
        _P("unzip", ProviderKind.TOOL, binary="unzip", install_recipe="apt-get install unzip"),
        _P("python_zipfile", ProviderKind.BUILTIN),   # python stdlib zipfile/tarfile
    ],
    "docker": [
        _P("docker", ProviderKind.TOOL, binary="docker", install_recipe="apt-get install docker.io", privilege="PRIVILEGED"),
    ],
}


class CapabilityService:
    """Discovers capabilities from the real environment and resolves how to fulfil them."""

    def __init__(self, *, backend=None, registry: Optional[Dict[str, List[ProviderSpec]]] = None):
        self._backend = backend
        self.registry = registry if registry is not None else CAPABILITY_REGISTRY
        self._cache: Dict[str, Capability] = {}

    # ------------------------------------------------------------------ #

    def _caps(self):
        if self._backend is None:
            from backend.agent_runtime.execution_backend import execution_backend
            self._backend = execution_backend
        try:
            return self._backend.capabilities()
        except Exception:
            return None

    # -- provider presence checks (environment-truthful, not hardcoded) -- #

    def _tool_present(self, spec: ProviderSpec) -> bool:
        if shutil.which(spec.binary):
            return True
        # A `python`/`python3` provider is satisfied by the interpreter FORGE itself is
        # running on, even when the venv's Scripts/bin directory is not on PATH — which
        # is the ordinary deployment. This is not a claim that cannot be honoured:
        # LocalBackend._normalise_command() rewrites a leading `python3`/`python` to
        # exactly this interpreter, so such a command really does run. Reporting
        # python_exec as unavailable while FORGE is itself executing inside Python
        # would be false, and would send a solvable task down the acquisition path.
        if spec.binary in ("python", "python3") and sys.executable:
            return True
        caps = self._caps()
        return bool(caps and caps.has_tool(spec.binary))

    def _pylib_present(self, spec: ProviderSpec) -> bool:
        try:
            if importlib.util.find_spec(spec.module) is not None:
                # A PYLIB may still need a companion CLI tool (pytesseract → tesseract).
                if spec.requires_tool and not shutil.which(spec.requires_tool):
                    caps = self._caps()
                    if not (caps and caps.has_tool(spec.requires_tool)):
                        return False
                return True
        except Exception:
            pass
        caps = self._caps()
        return bool(caps and caps.has_python_lib(spec.module))

    def _builtin_present(self, spec: ProviderSpec) -> bool:
        name = spec.name
        if name == "execution_backend":
            return True   # scripted stdin + InteractiveSession are always available now
        if name == "posix_pty":
            return not _IS_WINDOWS
        if name in ("urllib", "python_socket", "python_zipfile"):
            # Python stdlib — available wherever python runs (i.e. here).
            return True
        return True

    def _cloud_present(self, spec: ProviderSpec) -> bool:
        """A CLOUD provider is 'present' when its hosted API key is configured.

        Discovery stays environment-truthful: we do not call out to the API, we
        only report the provider as usable when FORGE actually holds a key for it.
        """
        if spec.name == "gemini":
            try:
                from backend.config import settings
                return bool(getattr(settings, "GEMINI_API_KEY", "")
                            or getattr(settings, "GEMINI_API_KEYS", ""))
            except Exception:
                return False
        return False

    def _present(self, spec: ProviderSpec) -> bool:
        if spec.kind is ProviderKind.TOOL:
            return self._tool_present(spec)
        if spec.kind is ProviderKind.PYLIB:
            return self._pylib_present(spec)
        if spec.kind is ProviderKind.CLOUD:
            return self._cloud_present(spec)
        return self._builtin_present(spec)

    # ------------------------------------------------------------------ #

    def discover(self, name: str, *, use_cache: bool = True) -> Capability:
        """Return the discovered :class:`Capability` for *name*."""
        if use_cache and name in self._cache:
            return self._cache[name]

        specs = self.registry.get(name)
        if not specs:
            cap = Capability(name=name, status=UNKNOWN, backend_supported=False,
                             recommended_action="replan",
                             reason=f"Unknown capability '{name}' — not in the capability registry.")
            self._cache[name] = cap
            return cap

        present: List[str] = []
        checked: List[str] = []
        for spec in specs:
            checked.append(spec.name)
            if self._present(spec):
                present.append(spec.name)

        preferred = specs[0].name
        installable = any(bool(s.install_recipe) for s in specs)
        acq_possible = self._acquisition_possible(specs)

        if present:
            if preferred in present:
                status, action, reason = AVAILABLE, "execute", f"Provider '{preferred}' is installed."
                chosen = preferred
            else:
                chosen = present[0]
                status, action = ALTERNATIVE_AVAILABLE, "execute"
                reason = (f"Preferred provider '{preferred}' is unavailable; using installed "
                          f"alternative '{chosen}'.")
            alternatives = [p for p in present if p != chosen]
            cap = Capability(
                name=name, status=status, available=True, provider=chosen,
                providers_checked=checked, alternatives=alternatives, backend_supported=True,
                installable=installable, acquisition_possible=acq_possible,
                recommended_action=action, reason=reason)
        elif acq_possible:
            cap = Capability(
                name=name, status=ACQUIRABLE, available=False, provider=preferred,
                providers_checked=checked, alternatives=[], backend_supported=True,
                installable=installable, acquisition_possible=True,
                recommended_action="request_acquisition",
                reason=(f"No provider for '{name}' is installed, but '{preferred}' can be "
                        f"acquired under privilege control."))
        else:
            cap = Capability(
                name=name, status=BLOCKED, available=False, provider="",
                providers_checked=checked, alternatives=[], backend_supported=True,
                installable=installable, acquisition_possible=False,
                recommended_action="replan",
                reason=(f"No provider for '{name}' is installed and none can be acquired in "
                        f"this environment. Replan around it."))
        self._cache[name] = cap
        return cap

    def is_available(self, name: str) -> bool:
        return self.discover(name).available

    def alternatives(self, name: str) -> List[str]:
        """Installed providers for *name* (the chosen one first)."""
        cap = self.discover(name)
        if not cap.available:
            return []
        return [cap.provider] + list(cap.alternatives)

    def provider_specs(self, name: str) -> List[ProviderSpec]:
        return list(self.registry.get(name, []))

    def spec_for(self, name: str, provider: str) -> Optional[ProviderSpec]:
        for s in self.registry.get(name, []):
            if s.name == provider:
                return s
        return None

    # ------------------------------------------------------------------ #

    def _acquisition_possible(self, specs: List[ProviderSpec]) -> bool:
        """Whether ANY provider could plausibly be acquired here (delegates to the planner)."""
        try:
            from backend.execution.acquisition import acquisition_planner
            return any(acquisition_planner.can_acquire(s) for s in specs)
        except Exception:
            # Conservative fallback: a pip-installable Python lib is acquirable if pip exists.
            if shutil.which("pip") or shutil.which("pip3"):
                return any(s.kind is ProviderKind.PYLIB for s in specs)
            return False

    def refresh(self, name: Optional[str] = None) -> None:
        """Invalidate cached discovery (e.g. after a successful acquisition)."""
        if name is None:
            self._cache.clear()
            try:
                if self._backend and hasattr(self._backend, "refresh"):
                    self._backend.refresh()
            except Exception:
                pass
        else:
            self._cache.pop(name, None)

    def summary(self) -> Dict[str, dict]:
        """Discover every registered capability (for the API/status view)."""
        return {name: self.discover(name).to_dict() for name in self.registry}


# Module-level singleton — the capability oracle for the intelligence layer.
capability_service = CapabilityService()
