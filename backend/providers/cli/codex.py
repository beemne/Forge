import os
import time
import asyncio
import tempfile
import logging
from typing import Optional
from backend.providers.base import ProviderResponse
from backend.providers.cli.base import BaseCLIProvider, redact_secrets
from backend.config import settings

logger = logging.getLogger("forge.providers.codex")

class AgentRouterCodexProvider(BaseCLIProvider):
    """Controlled CLI Subprocess Provider for AgentRouter via Codex CLI (`codex`)."""

    # Model to Environment Variable Mapping
    MODEL_KEY_MAP = {
        "gpt-5.6": "AGENTROUTER_GPT_5_6_KEY",
        "gpt-5.6-sol": "AGENTROUTER_GPT_5_6_SOL_KEY",
        "glm-5.3": "AGENTROUTER_GLM_5_3_KEY",
        "deepseek-v4-flash": "AGENTROUTER_DEEPSEEK_V4_FLASH_KEY"
    }

    def __init__(self):
        super().__init__(
            name="agentrouter_codex",
            cli_binary_default="codex",
            custom_path=settings.CODEX_PATH
        )

    def resolve_api_key(self, model: str) -> Optional[str]:
        env_var_name = self.MODEL_KEY_MAP.get(model)
        if env_var_name:
            key = getattr(settings, env_var_name, None) or os.environ.get(env_var_name)
            if key and key.strip():
                return key.strip()
        # Fallback to general AGENTROUTER_API_KEY
        return settings.AGENTROUTER_API_KEY or os.environ.get("AGENTROUTER_API_KEY")

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        capability: str = "general_reasoning",
        model: Optional[str] = "deepseek-v4-flash",
        timeout_seconds: int = 90,
        **kwargs
    ) -> ProviderResponse:
        model_to_use = model or "deepseek-v4-flash"
        executable = self.find_executable()

        if not executable:
            return ProviderResponse(
                provider_name=self.name,
                model_name=model_to_use,
                content="",
                is_refusal=True,
                refusal_reason="Codex CLI executable not found on host system."
            )

        api_key = self.resolve_api_key(model_to_use)
        if not api_key:
            return ProviderResponse(
                provider_name=self.name,
                model_name=model_to_use,
                content="",
                is_refusal=True,
                refusal_reason=f"No API key configured for model {model_to_use} or AgentRouter."
            )

        sub_env = os.environ.copy()
        sub_env["OPENAI_API_KEY"] = api_key
        sub_env["OPENAI_BASE_URL"] = "https://agentrouter.org/v1"
        sub_env["AGENTROUTER_API_KEY"] = api_key

        cmd = [
            executable,
            "exec",
            "--skip-git-repo-check",
            "--quiet",
            "--model", model_to_use,
            prompt
        ]

        start_time = time.time()

        temp_dir = tempfile.mkdtemp(prefix="forge_codex_cli_")
        logger.info(f"[AgentRouter CLI] Starting Codex Subprocess PID... Model: {model_to_use}")

        try:
            # Use CREATE_NO_WINDOW on Windows to prevent console popups
            creation_flags = 0
            import sys
            if sys.platform == "win32":
                creation_flags = 0x08000000  # CREATE_NO_WINDOW

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=temp_dir,
                env=sub_env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **({'creationflags': creation_flags} if creation_flags else {})
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
                    refusal_reason=f"Codex CLI process timed out after {timeout_seconds}s."
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
        
        # Check for AgentRouter quota exhaustion (402)
        if "402" in combined_output or "quota" in combined_output.lower() or "budget pool" in combined_output.lower():
            return ProviderResponse(
                provider_name=self.name,
                model_name=model_to_use,
                content="",
                is_refusal=True,
                refusal_reason=f"AgentRouter Quota Exhausted (402): {combined_output[:200]}"
            )

        # Check for authentication errors
        if "401" in combined_output or "unauthorized" in combined_output.lower() or "unauthenticated" in combined_output.lower():
            return ProviderResponse(
                provider_name=self.name,
                model_name=model_to_use,
                content="",
                is_refusal=True,
                refusal_reason=f"AgentRouter Authentication Refusal (401): {combined_output[:200]}"
            )

        if exit_code != 0 and (not clean_stdout.strip() or "error" in clean_stdout.lower()):
            return ProviderResponse(
                provider_name=self.name,
                model_name=model_to_use,
                content="",
                is_refusal=True,
                refusal_reason=f"Codex CLI exited with code {exit_code}: {combined_output[:200]}"
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
