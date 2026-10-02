"""
Writeup Ingestion Engine for FORGE Playbook Vault
==================================================
Converts raw CTF writeups (markdown / text files) into structured FORGE Playbooks (.yaml).
Indexes them immediately into the Playbook Vault FTS search index with sanitization.

Usage:
  py -m backend.knowledge.ingest_writeup path/to/writeup.md --category web --title "SSTI via Jinja"
"""

import sys
import os
import re
import time
import hashlib
import argparse
import logging
from typing import List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.knowledge.playbook_vault import playbook_vault, PlaybookSchema, CATEGORIES

logger = logging.getLogger("forge.ingest_writeup")
logging.basicConfig(level=logging.INFO, format="[IngestWriteup] %(message)s")


PROMPT_INJECTION_PATTERNS = [
    r"(?i)ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
    r"(?i)system\s*directive",
    r"(?i)you\s+are\s+now\s+(?:in|operating|a)",
    r"(?i)<\|im_start\|>",
    r"(?i)<SYS>",
    r"(?i)override\s+safety\s+guidelines",
    r"(?i)do\s+not\s+follow\s+(?:any\s+)?rules"
]


def sanitize_playbook_content(text: str) -> str:
    """Strips meta-instructions and adversarial prompt injection vectors from writeup content."""
    if not text:
        return ""
    sanitized = text
    for pattern in PROMPT_INJECTION_PATTERNS:
        sanitized = re.sub(pattern, "[FILTERED_DIRECTIVE]", sanitized)
    return sanitized


def infer_expected_outcome_signatures(content: str, category: str) -> List[str]:
    """Extract intermediate outcome signatures indicating step success from writeup patterns."""
    outcomes = []
    content_lower = content.lower()

    if "sqli" in content_lower or "sql injection" in content_lower:
        outcomes.append(r"(?:SQL syntax|mysql_fetch|sqlite3\.OperationalError|PG::SyntaxError)")
    if "ssti" in content_lower or "template" in content_lower:
        outcomes.append(r"(?:uid=\d+|root:x:0:0|\{\{.*\}\})")
    if "jwt" in content_lower or "bearer" in content_lower:
        outcomes.append(r"(?:HTTP/1\.[01] 200|Welcome, admin|flag|authenticated)")
    if "lfi" in content_lower or "file inclusion" in content_lower:
        outcomes.append(r"root:x:0:0")
    if "pwn" in content_lower or "bof" in content_lower or "buffer overflow" in content_lower:
        outcomes.append(r"(?:\[\+\] Opening connection|\[\*\] Switching to interactive mode|Segmentation fault|Core dumped)")

    # Look for explicit success strings in markdown quotes or output blocks
    for match in re.findall(r"(?:Output|Response|Result):\s*`([^`]+)`", content):
        clean_m = match.strip()
        if 3 < len(clean_m) < 40 and not clean_m.startswith("http"):
            outcomes.append(re.escape(clean_m))

    return list(dict.fromkeys(outcomes))[:4]


def parse_writeup_content(content: str, category: str = "web", title: Optional[str] = None, source_type: str = "human", source_url: Optional[str] = None) -> PlaybookSchema:
    # Extract title if not provided
    if not title:
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title = f"Writeup-{int(time.time())}"

    # Generate clean ID
    clean_title = re.sub(r'[^a-zA-Z0-9]+', '-', title.lower()).strip('-')
    content_hash = hashlib.md5(content[:500].encode("utf-8", errors="ignore")).hexdigest()[:6]
    clean_id = f"{category.lower()}-{clean_title[:40]}-{content_hash}"

    # Extract code blocks
    code_blocks = re.findall(r"```(?:python|bash|sh|c|cpp|sql|html)?\n(.*?)```", content, re.DOTALL)
    exploit_template = ""
    if code_blocks:
        exploit_template = max(code_blocks, key=len).strip()

    # Parameterize target URLs and flags
    exploit_template = re.sub(r"https?://[a-zA-Z0-9\.\-]+(?::\d+)?", "{TARGET_URL}", exploit_template)
    exploit_template = re.sub(r"(?:picoCTF|FLAG|CTF|HTB|THM)\{[^\}\s]+\}", "{FLAG}", exploit_template, flags=re.IGNORECASE)

    # Sanitize prompt injection directives from template and notes
    sanitized_template = sanitize_playbook_content(exploit_template or content[:1000])

    # Extract trigger signatures and keywords
    signatures = []
    tags = [category.lower(), "writeup"]

    keyword_map = {
        "sql injection": "sqli",
        "sqli": "sqli",
        "command injection": "rce",
        "remote code execution": "rce",
        "server-side template injection": "ssti",
        "ssti": "ssti",
        "jinja": "jinja2",
        "buffer overflow": "pwn",
        "bof": "pwn",
        "format string": "fmtstr",
        "heap": "heap",
        "xss": "xss",
        "csrf": "csrf",
        "lfi": "lfi",
        "rfi": "rfi",
        "jwt": "jwt",
        "deserialization": "deserialization",
        "pickle": "pickle",
        "pwntools": "pwntools",
        "sqlmap": "sqlmap",
        "nmap": "nmap",
        "gdb": "gdb"
    }

    content_lower = content.lower()
    for phrase, tag in keyword_map.items():
        if phrase in content_lower:
            tags.append(tag)

    tags = list(dict.fromkeys(tags))

    # Extract headers as signatures
    headers = re.findall(r"^#{1,3}\s+(.+)$", content, re.MULTILINE)
    signatures.extend(headers[:3])

    notes = content[:600].strip()
    if source_url:
        notes = f"Source URL: {source_url}\n\n" + notes
    sanitized_notes = sanitize_playbook_content(notes)

    expected_outcomes = infer_expected_outcome_signatures(content, category)

    playbook = PlaybookSchema(
        id=clean_id,
        category=category.lower() if category.lower() in CATEGORIES else "web",
        tags=tags,
        trigger_signatures=signatures,
        notes=sanitized_notes,
        exploit_template=sanitized_template,
        expected_outcome_signatures=expected_outcomes,
        source=source_type,
        confidence_score=1.0,
        times_used=1,
        success_rate=1.0,
        is_promoted=True,
        is_sanitized=True
    )

    return playbook


import httpx
from bs4 import BeautifulSoup


def ingest_url(url: str, category: str = "web", title: Optional[str] = None) -> PlaybookSchema:
    """Fetch CTF writeup from web URL (Medium, blog, HackMD), extract text/code/images, and ingest as playbook."""
    logger.info(f"Fetching writeup URL: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    with httpx.Client(timeout=15.0, follow_redirects=True, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    if not title:
        page_title = soup.find("title")
        if page_title and page_title.string:
            title = page_title.string.strip()
        else:
            h1 = soup.find("h1")
            title = h1.text.strip() if h1 else "URL Writeup"

    article = soup.find("article") or soup.find("main") or soup.body

    image_refs = []
    if article:
        for img in article.find_all("img"):
            src = img.get("src") or img.get("data-src")
            alt = img.get("alt", "Writeup Diagram")
            if src and not src.startswith("data:"):
                image_refs.append(f"![{alt}]({src})")

    code_snippets = []
    if article:
        for pre in article.find_all(["pre", "code"]):
            snippet = pre.text.strip()
            if len(snippet) > 15 and "\n" in snippet:
                code_snippets.append(f"```\n{snippet}\n```")

    text_content = ""
    if article:
        for elem in article.find_all(["nav", "header", "footer", "script", "style"]):
            elem.decompose()
        text_content = article.get_text(separator="\n", strip=True)

    combined_parts = [f"# {title}"]
    if image_refs:
        combined_parts.append("\n### Attached Images / Figures\n" + "\n".join(image_refs[:5]))
    if code_snippets:
        combined_parts.append("\n### Extracted Code Snippets\n" + "\n\n".join(code_snippets[:5]))
    combined_parts.append("\n### Article Body\n" + text_content[:2000])

    full_markdown = "\n\n".join(combined_parts)

    playbook = parse_writeup_content(
        content=full_markdown,
        category=category,
        title=title,
        source_type="url",
        source_url=url
    )

    saved_path = playbook_vault.save_playbook(playbook)
    logger.info(f"Successfully ingested URL '{url}' -> Playbook `{playbook.id}` at {saved_path}")
    return playbook


def ingest_raw_text(text: str, category: str = "web", title: Optional[str] = None) -> PlaybookSchema:
    """Ingest raw text/markdown writeup directly into playbook vault."""
    playbook = parse_writeup_content(content=text, category=category, title=title, source_type="human")
    saved_path = playbook_vault.save_playbook(playbook)
    logger.info(f"Successfully ingested raw text -> Playbook `{playbook.id}` at {saved_path}")
    return playbook


def ingest_file(file_path: str, category: str = "web", title: Optional[str] = None) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Writeup file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if not title:
        title = os.path.splitext(os.path.basename(file_path))[0]

    playbook = parse_writeup_content(content, category=category, title=title, source_type="human")
    saved_path = playbook_vault.save_playbook(playbook)
    logger.info(f"Successfully ingested writeup '{file_path}' -> Playbook `{playbook.id}` at {saved_path}")
    return saved_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest raw CTF writeups into FORGE Playbook Vault")
    parser.add_argument("file", help="Path to raw markdown/text writeup file or URL")
    parser.add_argument("--category", "-c", default="web", choices=CATEGORIES, help="Challenge category")
    parser.add_argument("--title", "-t", default=None, help="Custom title for the playbook")
    parser.add_argument("--url", action="store_true", help="Treat input as URL")

    args = parser.parse_args()
    if args.url or args.file.startswith("http://") or args.file.startswith("https://"):
        ingest_url(args.file, args.category, args.title)
    else:
        ingest_file(args.file, args.category, args.title)
