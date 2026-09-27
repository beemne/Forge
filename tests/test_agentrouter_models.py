import unittest
import asyncio
import os

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.config import settings
from backend.providers.router import model_router

class TestAgentRouterModels(unittest.TestCase):

    def test_agentrouter_models(self):
        if not settings.AGENTROUTER_API_KEY:
            self.skipTest("No AGENTROUTER_API_KEY configured")

        models_to_test = [
            "claude-opus-4-8",
            "claude-opus-5",
            "deepseek-v4-flash",
            "glm-5.3",
            "gpt-5.6-sol"
        ]

        async def run_tests():
            results = {}
            for model in models_to_test:
                try:
                    res = await model_router.route_request("Say hello", capability="general_reasoning", target_model=model)
                    if res.is_refusal:
                        results[model] = f"REFUSAL: {res.refusal_reason}"
                    else:
                        results[model] = f"SUCCESS ({res.provider_name}): {res.content[:50]}"
                except Exception as e:
                    results[model] = f"ERROR: {str(e)}"
            return results

        res_dict = asyncio.run(run_tests())
        print("\n--- AGENTROUTER SPECIFIC MODEL TEST RESULTS ---")
        for m, status in res_dict.items():
            line = f"Model [{m}]: {status}"
            print(line.encode('ascii', errors='ignore').decode('ascii'))
        print("-----------------------------------------------\n")

if __name__ == "__main__":
    unittest.main()
