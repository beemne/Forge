import unittest
import os
import tempfile
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.knowledge.ingest_repo import ingest_repository
from backend.knowledge.playbook_vault import playbook_vault

class TestIngestRepo(unittest.TestCase):

    def test_ingest_local_repo_structure(self):
        with tempfile.TemporaryDirectory(prefix="forge_test_repo_") as tmpdir:
            web_dir = os.path.join(tmpdir, "Server-Side Template Injection")
            os.makedirs(web_dir, exist_ok=True)
            
            md_path = os.path.join(web_dir, "README.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("""# SSTI Payload Collection

## Jinja2 Payload
```python
{{ self._TemplateReference__context.namespace.__init__.__globals__.os.popen('id').read() }}
```
""")

            count = ingest_repository(tmpdir, max_files=10)
            self.assertEqual(count, 1)

            # Verify searchable in vault
            results = playbook_vault.search_playbooks("SSTI Jinja2", category="web", candidate_tags=["ssti", "jinja2"], top_k=20)
            self.assertTrue(len(results) > 0)
            self.assertTrue(any("ssti" in r.tags for r in results))

            # Cleanup playbook generated in test
            for r in results:
                if "ssti" in r.tags:
                    pb_file = os.path.join(playbook_vault.base_dir, "web", f"{r.id}.yaml")
                    if os.path.exists(pb_file):
                        os.remove(pb_file)

if __name__ == "__main__":
    unittest.main()
