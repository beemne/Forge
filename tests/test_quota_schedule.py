import sys, os, asyncio
sys.path.insert(0, os.path.abspath('.'))

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.providers.quota_manager import quota_manager

def test_quota_manager_schedule():
    # 1. Verify model classifications
    assert quota_manager.is_quota_limited_model("claude-opus-5") is True
    assert quota_manager.is_quota_limited_model("claude-opus-4-8") is True
    assert quota_manager.is_quota_limited_model("gpt-5.6") is True
    assert quota_manager.is_quota_limited_model("deepseek-v4-flash") is False
    assert quota_manager.is_quota_limited_model("glm-5.3") is False

    # 2. Verify error detection
    error_msg = "402 Budget pool quota has been exhausted."
    assert quota_manager.detect_quota_error(error_msg) is True
    
    # 3. Verify record and check
    quota_manager.record_quota_exhaustion("claude-opus-5", error_msg)
    assert quota_manager.is_model_exhausted("claude-opus-5") is True
    assert quota_manager.should_skip_quota_limited_models() is True

    # 4. Fallback mapping
    fb = quota_manager.get_fallback_model("claude-opus-5")
    assert fb == "deepseek/deepseek-chat"

    next_batch_str = quota_manager.get_next_batch_time_str()
    print(f"Next batch string: {next_batch_str}")
    print("[OK] Quota manager batch schedule tests passed!")

if __name__ == "__main__":
    test_quota_manager_schedule()
