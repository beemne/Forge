# 🔥 FORGE — Autonomous CTF Intelligence & Exploitation Framework

> ⚠️ **Status: Under Active Development** — Features, agent capabilities, and security workflows are actively being developed and refined.

> **Next-Generation Autonomous CTF Command Center & Intelligence Orchestrator**  
> *Engineered for Real Use in Cybersecurity Competitions *

---

## 📌 Executive Overview

**FORGE** is a local web-based autonomous CTF intelligence & exploitation command center. It unifies high-speed target reconnaissance, vulnerability analysis, capability-driven tool execution, auditable AI decision routing, evidence vault logging, and state checkpointing into a single, high-performance web dashboard.

---

## 🏛️ Core Architecture & Security Principles

1. **Strict Tripartite Separation (WHAT vs HOW vs WHETHER)**:
   - **AI Agents** decide **WHAT** capability is needed (e.g., `request_capability("web_enumeration", target)`).
   - **Tool Manager** resolves **HOW** to execute approved tools safely on host binaries (`nmap`, `ffuf`, `tshark`, `gdb`).
   - **Privilege Manager** decides **WHETHER** elevated/privileged execution is permitted (`SAFE` vs `PRIVILEGED` vs `DANGEROUS`).

2. **Provider Abstraction & Cost Safeguards**:
   - Paid models default to `PAID_MODEL_ALLOWED=false` with session and daily USD budget caps (`DAILY_BUDGET_USD=5.00`).
   - Multi-provider fallback chain (Gemini, OpenRouter, NVIDIA, Cerebras, Cloudflare, Hugging Face, AgentRouter, Local LLMs) guarantees zero single-point-of-failure downtime.

3. **Resilient State Checkpointing**:
   - SQLite state vector database (`forge.db`) persists challenges, targets, runs, evidence artifacts, and state machine snapshots.
   - Operations survive backend restarts, target IP updates, network disconnects, and API quota resets.

4. **Emergency Stop (Kill Switch)**:
   - Immediate cancellation of active agent loops, tool subprocesses, and AI calls (`status = CANCELLED`) broadcasted live via WebSocket.

---

## 💻 Operating System & Hardware Requirements

> **Note**: At the current stage, **FORGE is specifically engineered and intended for Linux environments** (Parrot Security OS, Kali Linux, or Ubuntu 22.04+ LTS).

### Minimum Hardware
- **CPU**: 2+ Logical Cores (4+ Cores recommended for concurrent fuzzing)
- **RAM**: 4.0 GB Physical RAM (8.0 GB+ recommended)
- **Disk**: 2.0 GB free disk space for evidence artifacts and database state

### Software Runtimes
- **Python**: Python 3.10+ (`python3` & `python3-pip`)
- **Node.js**: Node.js 18+ & npm (`npm`)

---

### 📦 Linux Requirement & Tool Installation Guide

To install all system requirements, runtimes, and security binaries on Linux (Parrot OS / Kali / Ubuntu), execute:

```bash
# 1. Update package lists
sudo apt-get update -y

# 2. Install Core System Runtimes & Build Utilities
sudo apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  nodejs \
  npm \
  curl \
  git \
  build-essential

# 3. Install Security Tools & Binaries
sudo apt-get install -y \
  nmap \
  ffuf \
  gobuster \
  feroxbuster \
  masscan \
  tshark \
  tcpdump \
  binwalk \
  gdb \
  radare2 \
  sqlmap \
  john \
  hashcat \
  hydra \
  exploitdb \
  sublist3r

# 4. Install Global AI CLI Agents (Optional)
npm install -g @anthropic-ai/claude-code @openai/codex-cli
```

---

## 🚀 Quick Start — Single Command Launcher

FORGE runs as a **single unified system** (FastAPI backend + React Web Portal static bundle + WebSockets) on port `8000`.

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/your-org/Forge.git
cd Forge

# Install Python backend dependencies
pip install -r backend/requirements.txt

# (Optional) Setup environment configuration
cp .env.example .env
```

### 2. Single System Launch
To start the entire framework with a single command, run:

```bash
# Windows
py launch_forge.py

# Linux / macOS / WSL2
python3 launch_forge.py
```

### 3. Access Web Command Center
Open your web browser to:
👉 **`http://localhost:8000`**

*(Both the web portal UI and API/WebSocket services are served directly from `http://localhost:8000`)*.

---

## 🛠️ Complete Web Portal Capabilities & Feature Walkthrough

### 1. 🎯 Command Center (`/command`)
- **Live Target Radar**: Visual status tracking of target IP addresses, open ports, and active investigation runs.
- **Operational Strategy Selector**: Switch between `⚡ CTF CONTROLLED`, `🔥 FULL AUTONOMOUS`, and `🛡️ STEALTH RECON` strategies.
- **Real-Time WebSocket Feed**: Live streaming events (`RUN_STARTED`, `TOOL_EXECUTED`, `EVIDENCE_CAPTURED`, `FLAG_CAPTURED`, `KILL_SWITCH_ACTIVATED`).

### 2. 🏆 Challenge Management (`/challenges`)
- **Interactive Registration**: Create CTF challenges with custom typeable categories (`Web`, `Pwn`, `Crypto`, `Reverse`, `Forensics`, `OSINT`).
- **Workspace Pipeline**: 7-pane investigation pipeline (Target Radar, Recon, Web, Exploitation, Evidence, Terminal, Writeup Report).
- **Automated Report Generator**: One-click generation of comprehensive Markdown CTF writeups with evidence citations.

### 3. 🎯 Target Identity Architecture (`/targets`)
- **Decoupled Target Identity**: Separate static target identity from dynamic IP addresses.
- **Dynamic Address Re-binding**: Update target IP/domain on the fly without losing past evidence or run history.
- **Verification & Rediscovery**: Trigger manual target connectivity checks or automated port re-scans.

### 4. 🧠 Auditable AI Intelligence (`/ai_intelligence`)
- **Transparent Decision Feed**: Audits every AI routing decision (`Goal` → `Requested Capability` → `Resolved Tool` → `Selected LLM Model` → `Result`).
- **Confidence Scoring**: Model routing decisions annotated with confidence percentages and cost tracking.

### 5. 🗄️ Evidence Vault (`/evidence`)
- **Provenance Tracking**: Log command stdout/stderr dumps, captured HTTP requests, hashes, and flags with timestamps.
- **Markdown & Screenshot Inspector**: Preview formatted Markdown reports and captured image artifacts.
- **Vault Export**: Export evidence vault items into structured JSON or Markdown files.

### 6. 💻 Integrated Terminal Stream (`/terminal`)
- **Live Output Stream**: Real-time terminal logs with copy-to-clipboard, text filtering, and exit code status.
- **Daemon Auto-Start**: Auto-starts initialization streams for `Claude Code CLI` and `OpenAI Codex CLI` daemons upon portal startup.

### 7. 🧰 Tool Arsenal (`/tools`)
- **25 Pentesting & CTF Tools**: Complete capability-driven integration for `nmap`, `rustscan`, `masscan`, `ffuf`, `gobuster`, `feroxbuster`, `curl`, `httpx`, `browser`, `sqlmap`, `pwntools`, `gdb_pwndbg`, `ghidra_headless`, `radare2`, `binwalk`, `tshark`, `tcpdump`, `john`, `hashcat`, `hydra`, `volatility3`, `searchsploit`, `sublist3r`, `cyberchef`, and `python3`.
- **Operator Approvals**: One-click privilege escalation approval for privileged tools (`nmap -sS`, `tshark`).

### 8. 🌐 AI Infrastructure & Providers (`/providers`)
- **Interactive Provider Manager**: Modal interface to update API keys for Gemini, OpenRouter, NVIDIA, Cerebras, Cloudflare, AgentRouter, and custom Local LLMs.
- **AgentRouter Per-Model Keys**: Granular keys for `gpt-5.6-sol`, `deepseek-v4-flash`, `glm-5.3`, `claude-opus-4-8`, and `claude-opus-5`.
- **Live Connection Testing & Cost Analytics**: Real-time HTTP ping tests with latency gauges and usage statistics breakdown.

### 9. 🔍 OS Requirements & Host Dependency Auditor (`/system`)
- **Empirical PATH Resolution**: Real-time system PATH binary location checks for all 25 CTF tools.
- **1-Click Installation Recipes**: Displays copyable installation commands (`sudo apt-get install ...` / `choco install ...` / `pip install ...`) for missing binaries.
- **Hardware Specs**: Audits Python runtime version, physical RAM memory, and logical CPU core availability.

---

## ⚙️ System Settings & Security Safeguards

### Environment Variables (`.env`)
```env
# LLM Providers
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
AGENTROUTER_API_KEY=your_agentrouter_key

# Database
DATABASE_URL=sqlite:///./forge.db

# Security & Budget Controls
PAID_MODEL_ALLOWED=false
DAILY_BUDGET_USD=5.00
SESSION_BUDGET_USD=2.00
AUTO_APPROVE_PRIVILEGED=false
COMMAND_TIMEOUT_SECONDS=300
HOST=127.0.0.1
PORT=8000
```

---

## 🧪 Testing & Automated Verification

Run the comprehensive Python test suite:

```bash
# Run unit & integration test suite
py -m unittest discover tests

# Run end-to-end competition self-test harness
py scripts/competition_test.py
```

All 29/29 automated backend tests pass cleanly.

---

## 📄 License & Legal Notice

FORGE is intended strictly for authorized CTF competitions and defensive security research. Ensure proper authorization before scanning or testing target networks.
