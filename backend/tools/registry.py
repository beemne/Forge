from typing import Dict, List, Any, Optional
from pydantic import BaseModel

class ToolMetadata(BaseModel):
    tool_name: str
    capabilities: List[str]
    binary: str
    version_command: str
    installation_recipe: str
    os_compatibility: List[str] # ["linux", "windows", "darwin"]
    privilege_requirement: str # SAFE, PRIVILEGED, DANGEROUS
    risk_level: str
    timeout_seconds: int = 60
    args_template: str

class ToolRegistry:
    """Capability-aware Tool Registry holding pre-approved cybersecurity tools and runtime-added tools."""

    def __init__(self):
        self.tools: Dict[str, ToolMetadata] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        # 1. nmap
        self.register_tool(ToolMetadata(
            tool_name="nmap",
            capabilities=["network_scanning", "recon", "port_scan"],
            binary="nmap",
            version_command="nmap --version",
            installation_recipe="sudo apt-get install -y nmap / choco install -y nmap",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="LOW",
            timeout_seconds=120,
            args_template="{target} -sV -F"
        ))

        # 2. rustscan
        self.register_tool(ToolMetadata(
            tool_name="rustscan",
            capabilities=["network_scanning", "recon", "port_scan", "fast_port_scan"],
            binary="rustscan",
            version_command="rustscan --version",
            installation_recipe="cargo install rustscan / choco install rustscan",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="LOW",
            timeout_seconds=90,
            args_template="-a {target} -- -sV"
        ))

        # 3. masscan
        self.register_tool(ToolMetadata(
            tool_name="masscan",
            capabilities=["network_scanning", "port_scan"],
            binary="masscan",
            version_command="masscan --version",
            installation_recipe="sudo apt-get install -y masscan",
            os_compatibility=["linux"],
            privilege_requirement="PRIVILEGED",
            risk_level="MEDIUM",
            timeout_seconds=90,
            args_template="{target} -p1-65535 --rate 1000"
        ))

        # 4. ffuf
        self.register_tool(ToolMetadata(
            tool_name="ffuf",
            capabilities=["directory_enumeration", "web_testing", "web_fuzzing"],
            binary="ffuf",
            version_command="ffuf -V",
            installation_recipe="sudo apt-get install -y ffuf / go install github.com/ffuf/ffuf@latest",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="LOW",
            timeout_seconds=90,
            args_template="-u {target}/FUZZ -w {wordlist}"  # wordlist resolved at runtime by WordlistResolver
        ))

        # 5. gobuster
        self.register_tool(ToolMetadata(
            tool_name="gobuster",
            capabilities=["directory_enumeration", "web_testing", "dns_recon"],
            binary="gobuster",
            version_command="gobuster version",
            installation_recipe="sudo apt-get install -y gobuster / go install github.com/OJ/gobuster/v3@latest",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="LOW",
            timeout_seconds=90,
            args_template="dir -u {target} -w {wordlist}"
        ))

        # 6. feroxbuster
        self.register_tool(ToolMetadata(
            tool_name="feroxbuster",
            capabilities=["directory_enumeration", "web_testing", "recursive_content_discovery"],
            binary="feroxbuster",
            version_command="feroxbuster -V",
            installation_recipe="sudo apt-get install -y feroxbuster / cargo install feroxbuster",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="LOW",
            timeout_seconds=90,
            args_template="-u {target}"
        ))

        # 7. curl
        self.register_tool(ToolMetadata(
            tool_name="curl",
            capabilities=["web_testing", "recon", "http_request"],
            binary="curl",
            version_command="curl --version",
            installation_recipe="sudo apt-get install -y curl / choco install curl",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=30,
            args_template="-i -s {target}"
        ))

        # 8. httpx
        self.register_tool(ToolMetadata(
            tool_name="httpx",
            capabilities=["web_testing", "http_probing", "recon"],
            binary="httpx",
            version_command="httpx -version",
            installation_recipe="go install -v github.com/projectdiscovery/httpx/cmd/httpx@latest",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=45,
            args_template="-u {target} -status-code -title"
        ))

        # 9. tshark
        self.register_tool(ToolMetadata(
            tool_name="tshark",
            capabilities=["packet_analysis", "network_forensics"],
            binary="tshark",
            version_command="tshark --version",
            installation_recipe="sudo apt-get install -y tshark / choco install wireshark",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=60,
            args_template="-r {target}"
        ))

        # 10. tcpdump
        self.register_tool(ToolMetadata(
            tool_name="tcpdump",
            capabilities=["packet_analysis", "traffic_capture"],
            binary="tcpdump",
            version_command="tcpdump --version",
            installation_recipe="sudo apt-get install -y tcpdump",
            os_compatibility=["linux", "darwin"],
            privilege_requirement="PRIVILEGED",
            risk_level="MEDIUM",
            timeout_seconds=60,
            args_template="-r {target}"
        ))

        # Binary / File Analysis (Strings) — registered BEFORE binwalk deliberately:
        # execute_capability() selects the FIRST installed candidate for a capability,
        # so registration order *is* the priority. `strings -n 8` is the cheap,
        # read-only first pass over an unknown artifact; `binwalk -e` extracts embedded
        # files and writes them to disk, which is the wrong default for a bare
        # file_analysis. The rest of the codebase orders them the same way
        # (agents/artifact_classifier.py, swarm/candidates.py). Reordering these two
        # silently changes which tool every file_analysis run reaches for.
        self.register_tool(ToolMetadata(
            tool_name="strings",
            capabilities=["file_analysis", "reverse_engineering"],
            binary="strings",
            version_command="strings --version",
            installation_recipe="sudo apt-get install -y binutils",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=30,
            args_template="-n 8 {target}"
        ))

        # 11. binwalk
        self.register_tool(ToolMetadata(
            tool_name="binwalk",
            capabilities=["file_analysis", "forensics", "steg_analysis", "firmware_extraction"],
            binary="binwalk",
            version_command="binwalk --help",
            installation_recipe="sudo apt-get install -y binwalk / pip install binwalk",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=60,
            args_template="-e {target}"
        ))

        # 12. ghidra
        self.register_tool(ToolMetadata(
            tool_name="ghidra",
            capabilities=["reverse_engineering", "binary_analysis", "decompiler"],
            binary="ghidra",
            version_command="ghidra --version",
            installation_recipe="sudo apt-get install -y ghidra / choco install ghidra",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=120,
            args_template="{target}"
        ))

        # 13. radare2
        self.register_tool(ToolMetadata(
            tool_name="radare2",
            capabilities=["reverse_engineering", "binary_analysis", "disassembler"],
            binary="r2",
            version_command="r2 -v",
            installation_recipe="sudo apt-get install -y radare2",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=60,
            args_template="-A -c 'aaa; afl; pdf' {target}"
        ))

        # 14. gdb
        self.register_tool(ToolMetadata(
            tool_name="gdb",
            capabilities=["binary_analysis", "dynamic_analysis", "debugger"],
            binary="gdb",
            version_command="gdb --version",
            installation_recipe="sudo apt-get install -y gdb / choco install gdb",
            os_compatibility=["linux", "windows"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=60,
            args_template="-batch -ex 'info files' {target}"
        ))

        # 15. sqlmap
        self.register_tool(ToolMetadata(
            tool_name="sqlmap",
            capabilities=["web_testing", "sql_injection", "database_exploitation"],
            binary="sqlmap",
            version_command="sqlmap --version",
            installation_recipe="sudo apt-get install -y sqlmap / pip install sqlmap",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="PRIVILEGED",
            risk_level="MEDIUM",
            timeout_seconds=120,
            args_template="-u {target} --batch"
        ))

        # 16. john
        self.register_tool(ToolMetadata(
            tool_name="john",
            capabilities=["password_cracking", "hash_cracking"],
            binary="john",
            version_command="john --version",
            installation_recipe="sudo apt-get install -y john / choco install john",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="LOW",
            timeout_seconds=90,
            args_template="--wordlist={wordlist} {target}"
        ))

        # 17. hashcat
        self.register_tool(ToolMetadata(
            tool_name="hashcat",
            capabilities=["password_cracking", "gpu_hash_cracking"],
            binary="hashcat",
            version_command="hashcat --version",
            installation_recipe="sudo apt-get install -y hashcat / choco install hashcat",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="LOW",
            timeout_seconds=90,
            args_template="-m 0 -a 0 {target} {wordlist}"
        ))

        # 18. hydra
        self.register_tool(ToolMetadata(
            tool_name="hydra",
            capabilities=["online_bruteforce", "auth_testing", "password_cracking"],
            binary="hydra",
            version_command="hydra -h",
            installation_recipe="sudo apt-get install -y hydra",
            os_compatibility=["linux", "darwin"],
            privilege_requirement="PRIVILEGED",
            risk_level="MEDIUM",
            timeout_seconds=90,
            args_template="-L {wordlist} -P {wordlist} {target}"
        ))

        # 19. volatility3
        self.register_tool(ToolMetadata(
            tool_name="volatility3",
            capabilities=["memory_forensics", "memory_analysis", "forensics"],
            binary="vol",
            version_command="vol -h",
            installation_recipe="pip install volatility3",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=120,
            args_template="-f {target} windows.pslist"
        ))

        # 20. searchsploit
        self.register_tool(ToolMetadata(
            tool_name="searchsploit",
            capabilities=["exploit_search", "vulnerability_search", "recon"],
            binary="searchsploit",
            version_command="searchsploit -h",
            installation_recipe="sudo apt-get install -y exploitdb",
            os_compatibility=["linux", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=30,
            args_template="{target}"
        ))

        # 21. sublist3r
        self.register_tool(ToolMetadata(
            tool_name="sublist3r",
            capabilities=["subdomain_enumeration", "recon", "dns_recon"],
            binary="sublist3r",
            version_command="sublist3r -h",
            installation_recipe="sudo apt-get install -y sublist3r / pip install sublist3r",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=60,
            args_template="-d {target}"
        ))

        # 22. cyberchef
        self.register_tool(ToolMetadata(
            tool_name="cyberchef",
            capabilities=["data_decoding", "cyberchef_cli", "transformation"],
            binary="cyberchef",
            version_command="cyberchef --version",
            installation_recipe="npm install -g cyberchef-cli",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=30,
            args_template="{target}"
        ))

        # 23. python3
        self.register_tool(ToolMetadata(
            tool_name="python3",
            capabilities=["python_exec", "scripting"],
            binary="python3",
            version_command="python3 --version",
            installation_recipe="sudo apt-get install -y python3 / python installer",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=60,
            args_template="{target}"
        ))

        # 24. claude
        self.register_tool(ToolMetadata(
            tool_name="claude",
            capabilities=["ai_assistant", "claude_code"],
            binary="claude",
            version_command="claude --version",
            installation_recipe="npm install -g @anthropic-ai/claude-code",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=60,
            args_template="{target}"
        ))

        # 25. codex
        self.register_tool(ToolMetadata(
            tool_name="codex",
            capabilities=["ai_assistant", "codex_cli"],
            binary="codex",
            version_command="codex --version",
            installation_recipe="npm install -g @openai/codex-cli",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=60,
            args_template="{target}"
        ))

        # Vision / Multimodal Image Reading
        self.register_tool(ToolMetadata(
            tool_name="vision_read",
            capabilities=["vision_read", "file_analysis", "image_analysis"],
            binary="vision_read",
            version_command="vision_read --version",
            installation_recipe="built-in gemini provider",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=60,
            args_template="{target}"
        ))

        # Persistent Interactive Execution Primitives
        self.register_tool(ToolMetadata(
            tool_name="interactive_open",
            capabilities=["interactive_open", "interactive_execution"],
            binary="interactive_open",
            version_command="interactive_open --version",
            installation_recipe="built-in interactive execution",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=60,
            args_template="{target}"
        ))

        self.register_tool(ToolMetadata(
            tool_name="interactive_send",
            capabilities=["interactive_send", "interactive_execution"],
            binary="interactive_send",
            version_command="interactive_send --version",
            installation_recipe="built-in interactive execution",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=30,
            args_template="{target}"
        ))

        self.register_tool(ToolMetadata(
            tool_name="interactive_read",
            capabilities=["interactive_read", "interactive_execution"],
            binary="interactive_read",
            version_command="interactive_read --version",
            installation_recipe="built-in interactive execution",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=60,
            args_template="{target}"
        ))

        self.register_tool(ToolMetadata(
            tool_name="interactive_send_and_read",
            capabilities=["interactive_send_and_read", "interactive_execution"],
            binary="interactive_send_and_read",
            version_command="interactive_send_and_read --version",
            installation_recipe="built-in interactive execution",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=60,
            args_template="{target}"
        ))

        self.register_tool(ToolMetadata(
            tool_name="interactive_close",
            capabilities=["interactive_close", "interactive_execution"],
            binary="interactive_close",
            version_command="interactive_close --version",
            installation_recipe="built-in interactive execution",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement="SAFE",
            risk_level="SAFE",
            timeout_seconds=30,
            args_template="{target}"
        ))

    def register_tool(self, tool: ToolMetadata):
        self.tools[tool.tool_name] = tool

    def register_dynamic_tool(
        self,
        tool_name: str,
        binary: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        installation_recipe: str = "",
        args_template: str = "{target}",
        privilege_requirement: str = "SAFE",
        timeout_seconds: int = 60
    ) -> ToolMetadata:
        """Dynamically registers a newly installed or custom tool at runtime."""
        bin_name = binary or tool_name
        caps = list(capabilities) if capabilities else [tool_name]
        if tool_name not in caps:
            caps.append(tool_name)

        meta = ToolMetadata(
            tool_name=tool_name,
            capabilities=caps,
            binary=bin_name,
            version_command=f"{bin_name} --version",
            installation_recipe=installation_recipe or f"custom install {tool_name}",
            os_compatibility=["linux", "windows", "darwin"],
            privilege_requirement=privilege_requirement,
            risk_level="LOW",
            timeout_seconds=timeout_seconds,
            args_template=args_template
        )
        self.register_tool(meta)
        return meta

    def get_tool(self, tool_name: str) -> Optional[ToolMetadata]:
        return self.tools.get(tool_name)

    def get_tools_for_capability(self, capability: str) -> List[ToolMetadata]:
        matching = [t for t in self.tools.values() if capability in t.capabilities]
        if not matching and capability in self.tools:
            return [self.tools[capability]]
        return matching

    def get_all_arsenal_tools(self) -> List[tuple[str, str]]:
        """Returns (tool_name, installation_recipe) for all registered CLI/arsenal tools."""
        return [(t.tool_name, t.installation_recipe) for t in self.tools.values()
                if not t.tool_name.startswith("interactive_") and t.tool_name != "vision_read"]

tool_registry = ToolRegistry()

