import os
import time
import asyncio
import tempfile
import logging
from typing import Optional
from backend.providers.base import ProviderResponse
from backend.providers.cli.base import BaseCLIProvider, redact_secrets
from backend.config import settings

logger = logging.getLogger("forge.providers.claude_code")

class AgentRouterClaudeCodeProvider(BaseCLIProvider):
    """Controlled CLI Subprocess Provider for AgentRouter via Claude Code CLI (`claude`)."""

    # Model to Environment Variable Mapping
    MODEL_KEY_MAP = {
        "claude-opus-5": "AGENTROUTER_CLAUDE_OPUS_5_KEY",
        "claude-opus-4-8": "AGENTROUTER_CLAUDE_OPUS_4_8_KEY"
    }

    def __init__(self):
        super().__init__(
            name="agentrouter_claude_code",
            cli_binary_default="claude",
            custom_path=settings.CLAUDE_CODE_PATH
        )

    def resolve_api_key(self, model: str) -> Optional[str]:
        env_var_name = self.MODEL_KEY_MAP.get(model)
        if env_var_name:
            key = getattr(settings, env_var_name, None) or os.environ.get(env_var_name)
            if key and key.strip():
                return key.strip()
        # Fallback to AGENTROUTER_CLAUDE_OPUS_5_KEY or general AGENTROUTER_API_KEY
        return getattr(settings, "AGENTROUTER_CLAUDE_OPUS_5_KEY", None) or settings.AGENTROUTER_API_KEY or os.environ.get("AGENTROUTER_API_KEY")

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        capability: str = "general_reasoning",
        model: Optional[str] = "claude-opus-5",
        timeout_seconds: int = 90,
        **kwargs
    ) -> ProviderResponse:
        model_to_use = model or "claude-opus-5"
        executable = self.find_executable()

        if not executable:
            return ProviderResponse(
                provider_name=self.name,
                model_name=model_to_use,
                content="",
                is_refusal=True,
                refusal_reason="Claude Code CLI executable not found on host system."
            )

        api_key = self.resolve_api_key(model_to_use)
        if not api_key:
            return ProviderResponse(
                provider_name=self.name,
                model_name=model_to_use,
                content="",
                is_refusal=True, refusal_reason=f"No API key configured for model {model_to_use} or AgentRouter."
            )

        # Build isolated non-interactive subprocess environment
        sub_env = os.environ.copy()
        sub_env["ANTHROPIC_API_KEY"] = ""
        sub_env["ANTHROPIC_BASE_URL"] = "https://agentrouter.org"
        sub_env["ANTHROPIC_AUTH_TOKEN"] = api_key

        # Subprocess invocation arguments: non-interactive print mode with tools disabled
        # Flags: -p (print response), --tools "" (disable built-in tools), --no-session-persistence
        cmd = [
            executable,
            "-p", prompt,
            "--tools", "",
            "--no-session-persistence",
            "--output-format", "text"
        ]

        if system_instruction:
            cmd.extend(["--system-prompt", system_instruction])

        start_time = time.time()
        
        temp_dir = tempfile.mkdtemp(prefix="forge_claude_cli_")
        logger.info(f"[AgentRouter CLI] Starting Claude Code Subprocess PID... Model: {model_to_use}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=temp_dir,
                env=sub_env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_seconds
                )
                raw_stdout = stdout_bytes.decode(errors="replace")
                raw_stderr = stderr_bytes.decode(errors="replace")
                exit_code = proc.returncode

            except asyncio.TimeoutError:
                proc.kill()
                return ProviderResponse(
                    provider_name=self.name,
                    model_name=model_to_use,
                    content="",
                    is_refusal=True,
                    refusal_reason=f"Claude Code CLI process timed out after {timeout_seconds}s."
                )

        except Exception as e:
            return ProviderResponse(
                provider_name=self.name,
                model_name=model_to_use,
                content="",
                is_refusal=True,
                refusal_reason=redact_secrets(f"Subprocess launch error: {str(e)}", [api_key])
            )
        finally:
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

        elapsed_ms = (time.time() - start_time) * 1000
        clean_stdout = redact_secrets(raw_stdout, [api_key])
        clean_stderr = redact_secrets(raw_stderr, [api_key])

        combined_output = f"{clean_stdout}\n{clean_stderr}".strip()

        # Error checks in stdout & stderr
        if "401" in combined_output or "unauthorized" in combined_output.lower() or "unauthenticated" in combined_output.lower():
            return ProviderResponse(
                provider_name=self.name,
                model_name=model_to_use,
                content="",
                is_refusal=True,
                refusal_reason=f"AgentRouter Authentication Refusal (401): {combined_output[:200]}"
            )

        if "402" in combined_output or "quota" in combined_output.lower() or "budget pool" in combined_output.lower():
            return ProviderResponse(
                provider_name=self.name,
                model_name=model_to_use,
                content="",
                is_refusal=True,
                refusal_reason=f"AgentRouter Quota Exhausted (402): {combined_output[:200]}"
            )

        if exit_code != 0 and (not clean_stdout.strip() or "error" in clean_stdout.lower()):
            return ProviderResponse(
                provider_name=self.name,
                model_name=model_to_use,
                content="",
                is_refusal=True,
                refusal_reason=f"Claude Code CLI exited with code {exit_code}: {combined_output[:200]}"
            )

        return ProviderResponse(
            provider_name=self.name,
            model_name=model_to_use,
            content=clean_stdout.strip(),
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(clean_stdout) // 4,
            estimated_cost_usd=0.002,
            latency_ms=elapsed_ms,
            is_refusal=False
        )
