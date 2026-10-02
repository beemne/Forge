import os
import re
import asyncio
import logging
import httpx
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from backend.agents.artifact_classifier import classify_http_response, save_artifact_binary

logger = logging.getLogger("forge.turbo_recon")

class TurboReconManager:
    """Pre-warmed parallel reconnaissance runner for immediate context caching."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get_cached_recon(self, challenge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve pre-warmed recon results for a challenge if available."""
        return self._cache.get(challenge_id)

    async def start_turbo_recon(self, challenge_id: str, target: str, category: str = "web", working_directory: Optional[str] = None) -> Dict[str, Any]:
        """Launch non-blocking parallel reconnaissance burst before agent turn #1."""
        logger.info(f"[TurboRecon] Initiating pre-warmed recon burst for challenge {challenge_id} on target: {target}")

        recon_result = {
            "challenge_id": challenge_id,
            "target": target,
            "category": category,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoints": [],
            "headers": {},
            "interesting_files": [],
            "technologies": [],
            "binary_info": {},
            "artifact": None,          # deterministic classifier verdict on the first fetch
            "raw_summary": ""
        }

        # Multi-target '+' splitting
        targets = [t.strip() for t in target.split("+") if t.strip()]

        tasks = []
        for t in targets:
            if t.startswith("http://") or t.startswith("https://"):
                tasks.append(self._probe_web_target(t, recon_result, working_directory))
            elif os.path.exists(t):
                tasks.append(self._probe_binary_target(t, recon_result))
            else:
                tasks.append(self._probe_host_target(t, recon_result))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Build distilled prompt context summary
        summary_lines = [f"=== PRE-WARMED TURBO RECON FOR {target} ==="]
        if recon_result["technologies"]:
            summary_lines.append(f"Detected Tech: {', '.join(recon_result['technologies'])}")
        if recon_result["interesting_files"]:
            summary_lines.append(f"Accessible Paths: {', '.join(recon_result['interesting_files'])}")
        if recon_result["headers"]:
            summary_lines.append(f"Server Headers: {dict(list(recon_result['headers'].items())[:5])}")
        if recon_result["binary_info"]:
            summary_lines.append(f"Binary Details: {recon_result['binary_info']}")

        recon_result["raw_summary"] = "\n".join(summary_lines)
        self._cache[challenge_id] = recon_result
        logger.info(f"[TurboRecon] Completed pre-warmed recon for {challenge_id}. Summary:\n{recon_result['raw_summary']}")
        return recon_result

    async def _probe_web_target(self, url: str, result: Dict[str, Any], working_directory: Optional[str] = None):
        """Parallel web probe for robots.txt, .git, headers, cookies, and tech stack."""
        base_url = url.rstrip("/")
        paths_to_check = ["/robots.txt", "/.git/HEAD", "/sitemap.xml", "/api", "/admin", "/login"]

        async with httpx.AsyncClient(timeout=4.0, verify=False, follow_redirects=True) as client:
            # 1. Probe Base Target Headers & Content
            try:
                r = await client.get(base_url)
                # Deterministic binary pre-classifier on the FIRST response — flags a
                # downloadable artifact (e.g. an S3 octet-stream) before any agent turn.
                try:
                    clf = classify_http_response(
                        url=base_url,
                        content_type=(r.headers.get("content-type") or "").lower(),
                        server_header=(r.headers.get("server") or "").lower(),
                        content_length=int(r.headers["content-length"]) if r.headers.get("content-length") else len(r.content or b""),
                        response_body_prefix=(r.content or b"")[:512],
                    )
                    if clf.is_binary:
                        result["artifact"] = {"is_binary": True, "type": clf.artifact_type, "reason": clf.reason}
                        result["binary_info"]["format"] = clf.artifact_type
                        if working_directory and r.content:
                            saved = save_artifact_binary(
                                r.content, working_directory,
                                suggested_name=os.path.basename(base_url.split("?")[0]) or "artifact.bin")
                            result["artifact"]["safe_file_path"] = saved
                            logger.info(f"[TurboRecon] First-fetch binary artifact saved byte-exact: {saved}")
                except Exception as clf_err:
                    logger.debug(f"[TurboRecon] Classifier skip for {base_url}: {clf_err}")
                for k, v in r.headers.items():
                    if k.lower() in ["server", "x-powered-by", "x-framework", "set-cookie", "content-type"]:
                        result["headers"][k] = v
                        if "flask" in v.lower() or "werkzeug" in v.lower():
                            result["technologies"].append("Flask/Werkzeug (Python)")
                        elif "express" in v.lower() or "node" in v.lower():
                            result["technologies"].append("Express/Node.js")
                        elif "php" in v.lower():
                            result["technologies"].append("PHP")
                        elif "apache" in v.lower() or "nginx" in v.lower():
                            result["technologies"].append(v)
            except Exception as e:
                logger.debug(f"[TurboRecon] Base URL probe failed for {base_url}: {e}")

            # 2. Parallel probe for common paths
            async def check_path(p: str):
                try:
                    resp = await client.get(f"{base_url}{p}")
                    if resp.status_code in [200, 301, 302, 403]:
                        result["interesting_files"].append(f"{p} ({resp.status_code})")
                        if p == "/robots.txt" and resp.status_code == 200:
                            for line in resp.text.splitlines()[:5]:
                                if "disallow" in line.lower():
                                    result["endpoints"].append(line.strip())
                except Exception:
                    pass

            await asyncio.gather(*[check_path(p) for p in paths_to_check], return_exceptions=True)

    async def _probe_binary_target(self, file_path: str, result: Dict[str, Any]):
        """Probes local file or binary for headers, magic bytes, and strings."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(16)
                if header.startswith(b"\x7fELF"):
                    result["binary_info"]["format"] = "ELF Executable (Linux)"
                    result["technologies"].append("ELF Binary (x86/x64)")
                elif header.startswith(b"MZ"):
                    result["binary_info"]["format"] = "PE Executable (Windows)"
                    result["technologies"].append("PE Binary (Windows)")
                elif header.startswith(b"PK\x03\x04"):
                    result["binary_info"]["format"] = "ZIP Archive"
                elif header.startswith(b"\x1f\x8b"):
                    result["binary_info"]["format"] = "GZIP Compressed"
                else:
                    result["binary_info"]["format"] = "Raw Artifact"

            # Quick strings check for flags or symbols
            with open(file_path, "rb") as f:
                content = f.read(50000)
                ascii_strings = re.findall(rb"[A-Za-z0-9_\-\.\{\}]{5,}", content)
                flag_candidates = [s.decode(errors="ignore") for s in ascii_strings if b"{" in s and b"}" in s]
                if flag_candidates:
                    result["binary_info"]["flag_candidates"] = flag_candidates[:3]
        except Exception as e:
            logger.debug(f"[TurboRecon] Binary probe failed for {file_path}: {e}")

    async def _probe_host_target(self, host: str, result: Dict[str, Any]):
        """Quick port connectivity check for non-HTTP raw host/IP targets."""
        clean_host = host.split(":")[0]
        port = int(host.split(":")[1]) if ":" in host else 80
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(clean_host, port), timeout=2.0)
            result["endpoints"].append(f"{clean_host}:{port} (OPEN)")
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

turbo_recon = TurboReconManager()
