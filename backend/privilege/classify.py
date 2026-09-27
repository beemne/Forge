import re
from backend.tools.registry import tool_registry

DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bdd\s+if=", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r":\(\)\s*\{.*;\s*:\s*\}", re.IGNORECASE),  # fork bomb
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"curl[^|]*\|\s*(sh|bash)\b", re.IGNORECASE),
    re.compile(r"wget[^|]*\|\s*(sh|bash)\b", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bchmod\s+777\b", re.IGNORECASE),
    re.compile(r">\s*/dev/sd", re.IGNORECASE),
    re.compile(r"\biptables\b", re.IGNORECASE),
    re.compile(r"\buserdel\b|\bpasswd\b", re.IGNORECASE),
]

# Interpreters that run an operator/agent-authored SCRIPT.  `python3 exploit.py` is
# FORGE's ordinary automation path, not a privileged side effect, so these are treated
# exactly like an already-registered SAFE tool instead of falling through to the
# PRIVILEGED fail-closed default (which still covers genuinely unknown binaries:
# sqlmap, hydra, nc, socat, and anything else unregistered).
#
# SECURITY — ordering is load-bearing: this allowlist is consulted AFTER the
# DANGEROUS_PATTERNS scan above it in classify_command_privilege(), so it can never
# launder a destructive command.  `python3 -c "import os; os.system('rm -rf /')"`
# still classifies DANGEROUS, because the pattern scan reads the ENTIRE command string
# including text inside a `-c` argument.  Do not move the check above the scan.
AUTOMATION_SAFE_BINARIES = {"python", "python3", "bash", "sh", "node", "perl", "ruby", "php"}


def classify_command_privilege(cmd: str, bin_name: str) -> str:
    """Classify the privilege level required for a command.

    Priority order:
    1. Check cmd against dangerous regex patterns -> 'DANGEROUS'.  This runs FIRST,
       before any allowlist or registry lookup, so nothing can hide a destructive
       command behind a trusted name: `python3 -c "import os; os.system('rm -rf /')"`
       is DANGEROUS even though `python3` is a registered SAFE tool.
    2. If bin_name is in tool_registry.tools, return its privilege_requirement.
    3. Else, if bin_name is a common script interpreter -> 'SAFE' (automation).
    4. Otherwise, return 'PRIVILEGED' (fail-closed).
    """
    if any(pattern.search(cmd) for pattern in DANGEROUS_PATTERNS):
        return "DANGEROUS"

    if bin_name and bin_name in tool_registry.tools:
        return tool_registry.tools[bin_name].privilege_requirement

    if bin_name and bin_name in AUTOMATION_SAFE_BINARIES:
        return "SAFE"

    return "PRIVILEGED"
