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

from backend.providers.snippet_parser import SnippetParser

def test_nvidia_python_snippet():
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
      "content": "What is in this image?"
    }
  ],
  "model": "moonshotai/kimi-k3",
  "max_tokens": 16384,
  "temperature": 0.7
}
"""
    res = SnippetParser.parse_snippet(snippet)
    print("Parsed:", res)
    assert res["success"] is True
    assert res["api_key"] == "nvapi-EXAMPLEKEY0000000000000000000000000000000000"
    assert res["model"] == "moonshotai/kimi-k3"
    assert res["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert res["provider_name"] == "nvidia"
    assert res["parameters"]["temperature"] == 0.7
    assert res["parameters"]["max_tokens"] == 16384
    print("[OK] Python snippet test passed!")

def test_curl_snippet():
    curl = """
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \\
  -H "Authorization: Bearer nvapi-test1234567890abcdef1234567890" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "meta/llama-3.3-70b-instruct",
    "temperature": 0.2
  }'
"""
    res = SnippetParser.parse_snippet(curl)
    print("Parsed cURL:", res)
    assert res["success"] is True
    assert res["api_key"] == "nvapi-test1234567890abcdef1234567890"
    assert res["model"] == "meta/llama-3.3-70b-instruct"
    assert res["base_url"] == "https://integrate.api.nvidia.com/v1"
    print("[OK] cURL snippet test passed!")

def test_rapidapi_snippet():
    curl = r'''
curl --request POST \
	--url https://gpt-5-5.p.rapidapi.com/chat/completions \
	--header 'Content-Type: application/json' \
	--header 'x-rapidapi-host: gpt-5-5.p.rapidapi.com' \
	--header 'x-rapidapi-key: FAKErapidapikey0000000000000000' \
	--data '{"model":"GPT-5.5","messages":[{"role":"user","content":"Hi"}]}'
'''
    res = SnippetParser.parse_snippet(curl)
    print("Parsed RapidAPI:", res)
    assert res["success"] is True
    assert res["provider_name"] == "rapidapi"
    assert res["model"] == "GPT-5.5"
    assert res["api_key"] == "FAKErapidapikey0000000000000000"
    assert res["extra_headers"]["x-rapidapi-host"] == "gpt-5-5.p.rapidapi.com"
    print("[OK] RapidAPI snippet test passed!")

if __name__ == "__main__":
    test_nvidia_python_snippet()
    test_curl_snippet()
    test_rapidapi_snippet()
    print("ALL SNIPPET TESTS PASSED!")
