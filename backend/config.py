import logging
import os
from pathlib import Path

logger = logging.getLogger("forge.config")

# Anchored to the repository root rather than the working directory, so `settings`
# and the database guard (backend/database/guard.py) always read the same file and
# cannot disagree about which database is production.
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    class Settings(BaseSettings):
        # pydantic-settings loads .env itself, and gives real process environment
        # variables precedence over it -- which is the behavior FORGE wants.
        #
        # There is deliberately NO load_dotenv(override=True) call here. It used to
        # rewrite os.environ at import time, discarding a DATABASE_URL that a test had
        # already pinned to the isolated database: the suite then ran against
        # production forge.db and test tearDowns deleted real rows. See
        # backend/database/guard.py for the structural guard against that.
        model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

        PROJECT_NAME: str = "FORGE Autonomous CTF Framework"
        VERSION: str = "1.0.0"
        ENVIRONMENT: str = "development"
        HOST: str = "127.0.0.1"
        PORT: int = 8000
        SECRET_KEY: str = "forge-secret-key-change-in-production"

        # API gate. Empty string => auth disabled (local dev mode); any non-empty
        # value is required in the X-Forge-Key header on every /api route.
        # MUST be set before exposing FORGE beyond localhost.
        FORGE_API_KEY: str = ""

        # Comma-separated CORS origin allowlist. Never "*": the API is unauthenticated
        # by default, so a wildcard origin lets any visited site drive this backend.
        FORGE_ALLOWED_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

        # Database
        DATABASE_URL: str = "sqlite:///./forge.db"
        
        # API Keys & Per-Model AgentRouter Keys
        GEMINI_API_KEY: str = ""
        GEMINI_API_KEYS: str = ""
        NVIDIA_API_KEY: str = ""
        CEREBRAS_API_KEY: str = ""
        OPENROUTER_API_KEY: str = ""
        HF_TOKEN: str = ""
        CLOUDFLARE_API_TOKEN: str = ""
        CLOUDFLARE_ACCOUNT_ID: str = ""
        AGENTROUTER_API_KEY: str = ""
        MISTRAL_API_KEY: str = ""
        COHERE_API_KEY: str = ""
        GROQ_API_KEY: str = ""
        GROQ_API_KEYS: str = ""
        RAPIDAPI_KEY: str = ""
        XKIRO_API_KEY: str = ""

        # AgentRouter Per-Model Keys
        AGENTROUTER_CLAUDE_OPUS_5_KEY: str = ""
        AGENTROUTER_CLAUDE_OPUS_4_8_KEY: str = ""
        AGENTROUTER_GPT_5_6_KEY: str = ""
        AGENTROUTER_GPT_5_6_SOL_KEY: str = ""
        AGENTROUTER_GLM_5_3_KEY: str = ""
        AGENTROUTER_DEEPSEEK_V4_FLASH_KEY: str = ""

        # CLI Path Overrides
        CLAUDE_CODE_PATH: str = ""
        CODEX_PATH: str = ""
        
        # Budget Controls
        PAID_MODEL_ALLOWED: bool = True
        DAILY_BUDGET_USD: float = 5.00
        SESSION_BUDGET_USD: float = 2.00

        # Agent execution budget — limits unbounded loops that exhaust shared rate limits.
        # Override per-run via the challenge start form or the /settings endpoint.
        # N tool-call iterations OR M wall-clock minutes, whichever fires first.
        AGENT_MAX_ITERATIONS: int = 40      # default tool-call budget per run
        AGENT_MAX_MINUTES: int = 30         # default wall-clock ceiling per run (minutes)

        # HITL checkpoint interval in seconds (default: 5 minutes).
        CHECKPOINT_INTERVAL_SECONDS: int = 300

        # HITL checkpoint wait timeout in seconds. If no operator guidance is supplied
        # within this window, the checkpoint times out and agents resume autonomously.
        # Set to 0 for non-blocking / immediate resume. Default: 30 seconds.
        CHECKPOINT_TIMEOUT_SECONDS: int = 30

        # Per-command operator-approval mode for PRIVILEGED/DANGEROUS commands.
        #   "manual" (default) — every PRIVILEGED/DANGEROUS command waits (with no
        #                        timeout) for an explicit operator approve/deny.
        #   "auto"            — those commands run unattended; the ONLY thing that
        #                        still halts execution is a command that literally
        #                        needs `sudo`, which waits (no timeout) for the
        #                        password only, not a yes/no decision.
        # Defaults to the safer "manual" so an unconfigured deployment keeps asking.
        # NOTE: this is NOT the cycle-level HITL checkpoint timeout above.
        FORGE_APPROVAL_MODE: str = "manual"
        AUTO_APPROVE_PRIVILEGED: bool = False

        # Default flag-pattern string shown in the challenge form and baked into
        # agent prompts.  Users may override this per-challenge at start time.
        # Pipe-separated list of regex prefixes; the agent uses this as a
        # VALIDATION FILTER only — it must never construct strings to match.
        DEFAULT_FLAG_PATTERNS: str = "picoCTF{...}|FLAG{...}|flag{...}|HTB{...}|CTF{...}"

        # Number of general-purpose agents spawned per run. Integer as a string, or
        # "auto" -> min(cpu_cores-1, 4). Clamped to [1, 6] at dispatch.
        AGENT_POOL_SIZE: str = "3"

        # Max consecutive LOCAL execution failures (a command that never reached the
        # target — Errno 2, SyntaxError, permission denied) before an agent aborts and
        # reports a blocker, instead of burning its budget re-running a broken invocation.
        LOCAL_EXEC_MAX_RETRIES: int = 2

        # Floor on max_tokens for GLM-routed (reasoning) models. Too-low a ceiling
        # starves the reasoning-token budget and returns HTTP 200 with empty content.
        GLM_MIN_MAX_TOKENS: int = 2048

        # Seconds before the platform instance-expiry deadline at which the swarm winds
        # down (reports best findings and stops) rather than being cut off mid-command.
        INSTANCE_WINDDOWN_BUFFER_SECONDS: int = 30

except ImportError:
    class Settings:
        def __init__(self):
            # pydantic-settings is unavailable, so this branch reads .env itself --
            # otherwise the keys below would silently fall back to their defaults.
            # override=False is essential: real process environment variables must
            # win over .env, exactly as pydantic-settings orders them. override=True
            # is what discarded a test's pinned DATABASE_URL and pointed the suite at
            # production forge.db.
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=ENV_FILE, override=False)
            except ImportError:
                pass

            self.PROJECT_NAME = os.getenv("PROJECT_NAME", "FORGE Autonomous CTF Framework")
            self.VERSION = os.getenv("VERSION", "1.0.0")
            self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
            self.HOST = os.getenv("HOST", "127.0.0.1")
            self.PORT = int(os.getenv("PORT", 8000))
            self.SECRET_KEY = os.getenv("SECRET_KEY", "forge-secret-key")
            self.FORGE_API_KEY = os.getenv("FORGE_API_KEY", "")
            self.FORGE_ALLOWED_ORIGINS = os.getenv(
                "FORGE_ALLOWED_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173"
            )
            self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./forge.db")
            self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
            self.NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
            self.CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
            self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
            self.HF_TOKEN = os.getenv("HF_TOKEN", "")
            self.CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
            self.CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
            self.AGENTROUTER_API_KEY = os.getenv("AGENTROUTER_API_KEY", "")
            self.MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
            self.COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
            self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
            self.XKIRO_API_KEY = os.getenv("XKIRO_API_KEY", "")

            # AgentRouter Per-Model Keys
            self.AGENTROUTER_CLAUDE_OPUS_5_KEY = os.getenv("AGENTROUTER_CLAUDE_OPUS_5_KEY", "")
            self.AGENTROUTER_CLAUDE_OPUS_4_8_KEY = os.getenv("AGENTROUTER_CLAUDE_OPUS_4_8_KEY", "")
            self.AGENTROUTER_GPT_5_6_KEY = os.getenv("AGENTROUTER_GPT_5_6_KEY", "")
            self.AGENTROUTER_GPT_5_6_SOL_KEY = os.getenv("AGENTROUTER_GPT_5_6_SOL_KEY", "")
            self.AGENTROUTER_GLM_5_3_KEY = os.getenv("AGENTROUTER_GLM_5_3_KEY", "")
            self.AGENTROUTER_DEEPSEEK_V4_FLASH_KEY = os.getenv("AGENTROUTER_DEEPSEEK_V4_FLASH_KEY", "")

            # CLI Paths
            self.CLAUDE_CODE_PATH = os.getenv("CLAUDE_CODE_PATH", "")
            self.CODEX_PATH = os.getenv("CODEX_PATH", "")

            self.PAID_MODEL_ALLOWED = os.getenv("PAID_MODEL_ALLOWED", "true").lower() == "true"
            self.DAILY_BUDGET_USD = float(os.getenv("DAILY_BUDGET_USD", 5.0))
            self.SESSION_BUDGET_USD = float(os.getenv("SESSION_BUDGET_USD", 2.0))
            self.AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", 40))
            self.AGENT_MAX_MINUTES = int(os.getenv("AGENT_MAX_MINUTES", 30))
            self.CHECKPOINT_INTERVAL_SECONDS = int(os.getenv("CHECKPOINT_INTERVAL_SECONDS", 300))
            self.CHECKPOINT_TIMEOUT_SECONDS = int(os.getenv("CHECKPOINT_TIMEOUT_SECONDS", 30))
            # Per-command operator-approval mode: "manual" (ask, no timeout) or "auto"
            # (run unattended; only a literal `sudo` command waits, for the password).
            self.FORGE_APPROVAL_MODE = os.getenv("FORGE_APPROVAL_MODE", "manual").strip().lower()
            self.DEFAULT_FLAG_PATTERNS = os.getenv(
                "DEFAULT_FLAG_PATTERNS",
                "picoCTF{...}|FLAG{...}|flag{...}|HTB{...}|CTF{...}"
            )
            self.AGENT_POOL_SIZE = os.getenv("AGENT_POOL_SIZE", "3")
            self.LOCAL_EXEC_MAX_RETRIES = int(os.getenv("LOCAL_EXEC_MAX_RETRIES", 2))
            self.GLM_MIN_MAX_TOKENS = int(os.getenv("GLM_MIN_MAX_TOKENS", 2048))
            self.INSTANCE_WINDDOWN_BUFFER_SECONDS = int(os.getenv("INSTANCE_WINDDOWN_BUFFER_SECONDS", 30))

settings = Settings()

# Fail-closed validation of the operator approval mode, applied to BOTH Settings
# definitions above (pydantic and the plain-attribute fallback). Anything that is not
# exactly "auto" or "manual" — a typo, wrong case, stray whitespace, empty string —
# logs a warning and falls back to "manual", the safer mode, which always asks before
# executing rather than silently switching to unattended auto-run.
_resolved_approval_mode = str(getattr(settings, "FORGE_APPROVAL_MODE", "manual") or "").strip().lower()
if _resolved_approval_mode not in ("auto", "manual"):
    logger.warning(
        "Invalid FORGE_APPROVAL_MODE=%r; falling back to 'manual' "
        "(expected exactly 'auto' or 'manual').",
        getattr(settings, "FORGE_APPROVAL_MODE", None),
    )
    _resolved_approval_mode = "manual"
settings.FORGE_APPROVAL_MODE = _resolved_approval_mode
