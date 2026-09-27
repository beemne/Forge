import sys, os
sys.path.insert(0, os.path.abspath("."))

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.providers.router import model_router
from backend.providers.snippet_parser import SnippetParser

def test_dynamic_model_registration():
    snippet = """
import requests

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True

headers = {
    "Authorization": "Bearer nvapi-EXAMPLEKEY0000000000000000000000000000000000",
    "Accept": "text/event-stream" if stream else "application/json",
}

payload = {
  "messages": [
    {
      "role": "user",
      "content": "ping"
    }
  ],
  "model": "moonshotai/kimi-k3"
}
"""
    parsed = SnippetParser.parse_snippet(snippet)
    assert parsed["success"] is True
    
    # Register in router
    prov = model_router.register_custom_model(
        provider_name=parsed["provider_name"],
        api_key=parsed["api_key"],
        model_id=parsed["model"],
        base_url=parsed["base_url"]
    )
    
    assert "nvidia" in model_router.providers
    assert model_router.MODEL_PROVIDER_MAP["moonshotai/kimi-k3"] == ("nvidia", "moonshotai/kimi-k3")
    print("[OK] Dynamic model registration test passed!")

if __name__ == "__main__":
    test_dynamic_model_registration()
