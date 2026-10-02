"""
Bulk Repository Ingestion Engine for FORGE Playbook Vault
=========================================================
Clones or processes repositories (e.g., PayloadsAllTheThings, HackTricks),
extracts markdown files, payload blocks, and trigger signatures,
and ingests them into FORGE's SQLite FTS5 search index.

Usage:
  py -m backend.knowledge.ingest_repo https://github.com/swisskyrepo/PayloadsAllTheThings.git
  py -m backend.knowledge.ingest_repo https://github.com/HackTricks-wiki/hacktricks.git
"""

import sys
import os
import re
import shutil
import tempfile
import subprocess
import argparse
import logging
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.knowledge.ingest_writeup import parse_writeup_content
from backend.knowledge.playbook_vault import playbook_vault

logger = logging.getLogger("forge.ingest_repo")
logging.basicConfig(level=logging.INFO, format="[IngestRepo] %(message)s")


def infer_category_and_tags(relative_path: str, content: str) -> Tuple[str, List[str]]:
    path_lower = relative_path.lower().replace("\\", "/")
    content_lower = content.lower()

    category = "web"
    tags = ["cheatsheet"]

    if "pwn" in path_lower or "exploit" in path_lower or "bof" in path_lower or "privilege-escalation" in path_lower:
        category = "pwn"
    elif "crypto" in path_lower or "rsa" in path_lower or "cipher" in path_lower:
        category = "crypto"
    elif "forensics" in path_lower or "stego" in path_lower or "memory" in path_lower or "pcap" in path_lower:
        category = "forensics"
    elif "reverse" in path_lower or "reversing" in path_lower or "ghidra" in path_lower or "binary" in path_lower:
        category = "reverse"
    elif "osint" in path_lower or "recon" in path_lower:
        category = "osint"

    # Category path tags
    parts = path_lower.split("/")
    for part in parts:
        clean_part = re.sub(r"[^a-z0-9]", "", part)
        if clean_part and len(clean_part) > 2 and clean_part not in ["readme", "md", "docs"]:
            tags.append(clean_part)

    # Common security keyword tags
    common_vuln_keywords = [
        "ssti", "sqli", "sql", "xss", "rce", "lfi", "rfi", "ssrf", "csrf", "jwt",
        "idor", "cors", "nosql", "xxe", "deserialization", "prototype-pollution",
        "jinja2", "flask", "django", "express", "bof", "rop", "heap", "format-string",
        "rsa", "aes", "ecc", "stego", "ghidra", "pcap", "wireshark"
    ]
    for kw in common_vuln_keywords:
        if kw in path_lower or kw in content_lower:
            tags.append(kw)

    return category, list(dict.fromkeys(tags))


def process_markdown_file(file_path: str, repo_name: str, base_dir: str) -> int:
    rel_path = os.path.relpath(file_path, base_dir)
    filename = os.path.basename(file_path)

    if filename.lower() in ["license", "contributing.md", "code_of_conduct.md"]:
        return 0

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if len(content.strip()) < 50:
            return 0

        # Skip files with no code blocks or technical headers
        if "```" not in content and "#" not in content:
            return 0

        category, tags = infer_category_and_tags(rel_path, content)
        title = f"{repo_name}: {rel_path.replace('.md', '').replace('/', ' > ')}"

        playbook = parse_writeup_content(
            content=content,
            category=category,
            title=title,
            source_type="repo",
            source_url=f"repo://{repo_name}/{rel_path}"
        )

        playbook.tags = list(dict.fromkeys(playbook.tags + tags))
        playbook_vault.save_playbook(playbook)
        return 1
    except Exception as e:
        logger.warning(f"Error processing {rel_path}: {e}")
        return 0


def ingest_repository(repo_url_or_path: str, max_files: int = 500) -> int:
    temp_dir = None
    if repo_url_or_path.startswith("http://") or repo_url_or_path.startswith("https://") or repo_url_or_path.endswith(".git"):
        repo_name = os.path.splitext(os.path.basename(repo_url_or_path))[0]
        temp_dir = tempfile.mkdtemp(prefix=f"forge_repo_{repo_name}_")
        logger.info(f"Cloning {repo_url_or_path} (sparse checkout, markdown only)...")
        try:
            # Step 1: Init bare repo with long path support
            subprocess.run(
                ["git", "-c", "core.longpaths=true", "init", temp_dir],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            # Step 2: Add remote
            subprocess.run(
                ["git", "-C", temp_dir, "remote", "add", "origin", repo_url_or_path],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            # Step 3: Enable sparse checkout with cone mode disabled (pattern mode)
            subprocess.run(
                ["git", "-C", temp_dir, "config", "core.sparseCheckout", "true"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            subprocess.run(
                ["git", "-C", temp_dir, "config", "core.longpaths", "true"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            # Step 4: Write sparse checkout patterns — only .md files
            sparse_dir = os.path.join(temp_dir, ".git", "info")
            os.makedirs(sparse_dir, exist_ok=True)
            with open(os.path.join(sparse_dir, "sparse-checkout"), "w") as sc:
                sc.write("*.md\n")
                sc.write("**/*.md\n")
            # Step 5: Shallow fetch + checkout
            subprocess.run(
                ["git", "-C", temp_dir, "fetch", "--depth", "1", "origin", "master"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            # Try 'master' first, fall back to 'main'
            result = subprocess.run(
                ["git", "-C", temp_dir, "checkout", "FETCH_HEAD"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                logger.info("master branch failed, trying 'main'...")
                subprocess.run(
                    ["git", "-C", temp_dir, "fetch", "--depth", "1", "origin", "main"],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                subprocess.run(
                    ["git", "-C", temp_dir, "checkout", "FETCH_HEAD"],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            logger.info(f"Sparse checkout complete for {repo_name} (markdown files only).")
        except Exception as e:
            logger.error(f"Failed to clone repository {repo_url_or_path}: {e}")
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            return 0
        target_dir = temp_dir
    else:
        target_dir = os.path.abspath(repo_url_or_path)
        repo_name = os.path.basename(target_dir)

    logger.info(f"Scanning and indexing Markdown files from '{repo_name}'...")
    count = 0
    try:
        for root, _, files in os.walk(target_dir):
            if ".git" in root:
                continue
            for file in files:
                if file.endswith(".md") or file.endswith(".markdown"):
                    full_path = os.path.join(root, file)
                    count += process_markdown_file(full_path, repo_name, target_dir)
                    if count >= max_files:
                        logger.info(f"Reached max file limit threshold of {max_files} playbooks for {repo_name}.")
                        break
            if count >= max_files:
                break
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info(f"Successfully ingested {count} playbooks into FORGE Playbook Vault from {repo_name}.")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest entire CTF cheat-sheet repositories into FORGE memory")
    parser.add_argument("repo", help="Git clone URL or local directory path of repository")
    parser.add_argument("--max-files", "-m", type=int, default=500, help="Maximum markdown files to ingest per repo")

    args = parser.parse_args()
    ingest_repository(args.repo, args.max_files)
