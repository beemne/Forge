"""
OCR / vision capability (Phase 4.x §19).

OCR is modelled as a first-class *capability* rather than a hardcoded call to one
library.  The agent asks for text extraction; this service discovers whether any
OCR provider is available (``tesseract`` CLI, ``pytesseract``, or ``easyocr``) and
either performs it or returns a STRUCTURED "blocked" result so the swarm can
replan — instead of retrying a missing ``tesseract`` dozens of times (the Binary
Digits failure, §20).

The preferred provider is Gemini vision whenever a ``GEMINI_API_KEY`` is
configured: it needs no local binary and reads text straight from the image
bytes, so an environment without ``tesseract``/``pytesseract`` installed no
longer dead-ends. Local providers (tesseract, pytesseract, easyocr) remain as
offline fallbacks, and all providers are pluggable via the capability registry,
so callers never change when the provider order does.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger("forge.execution.ocr")

OCR_OK = "OK"
OCR_BLOCKED = "BLOCKED_CAPABILITY"
OCR_ERROR = "ERROR"
OCR_NO_INPUT = "NO_INPUT"

CAPABILITY = "ocr"


@dataclass
class OCRResult:
    status: str = OCR_OK
    text: str = ""
    provider: str = ""
    reason: str = ""
    recommended_action: str = "execute"
    providers_checked: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)
    acquisition_possible: bool = False

    @property
    def ok(self) -> bool:
        return self.status == OCR_OK

    @property
    def blocked(self) -> bool:
        return self.status == OCR_BLOCKED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status, "text": self.text, "provider": self.provider,
            "reason": self.reason, "recommended_action": self.recommended_action,
            "providers_checked": list(self.providers_checked),
            "alternatives": list(self.alternatives),
            "acquisition_possible": self.acquisition_possible,
        }

    def observability_line(self) -> str:
        if self.blocked:
            return (f"CAPABILITY_UNAVAILABLE capability=ocr "
                    f"providers_checked={self.providers_checked} "
                    f"alternatives={self.alternatives} "
                    f"acquisition_possible={self.acquisition_possible} "
                    f"recommended_action={self.recommended_action}")
        return f"OCR_{self.status} provider={self.provider} chars={len(self.text)}"


class OCRService:
    """Capability-aware OCR. Never hardcodes a provider; blocks cleanly when none exists."""

    def __init__(self, *, capability_service=None):
        self._cap_service = capability_service

    def _caps(self):
        if self._cap_service is None:
            from backend.execution.capabilities import capability_service
            self._cap_service = capability_service
        return self._cap_service

    def capability(self):
        return self._caps().discover(CAPABILITY)

    def available(self) -> bool:
        return self._caps().is_available(CAPABILITY)

    def _blocked_result(self, cap) -> OCRResult:
        return OCRResult(
            status=OCR_BLOCKED, provider="", reason=cap.reason,
            recommended_action=cap.recommended_action or "replan",
            providers_checked=list(cap.providers_checked),
            alternatives=list(cap.alternatives),
            acquisition_possible=cap.acquisition_possible)

    async def extract_text(self, image_path: str, *, lang: str = "eng",
                           timeout_seconds: int = 120) -> OCRResult:
        """Extract text from *image_path* using the best available OCR provider.

        Returns a STRUCTURED result:
          * OK           — text extracted (provider named)
          * BLOCKED      — no provider available; carries recommended_action=replan
          * ERROR        — a provider was available but failed on this input
        The caller must treat BLOCKED as a single, terminal signal — NOT something to
        retry — and route it to recovery/replan.
        """
        if not image_path or not os.path.exists(image_path):
            return OCRResult(status=OCR_NO_INPUT, reason=f"Image not found: {image_path}",
                             recommended_action="replan")

        cap = self.capability()
        if not cap.available:
            logger.info(f"[OCR] blocked: {cap.observability_line()}")
            return self._blocked_result(cap)

        provider = cap.provider
        try:
            if provider == "gemini":
                return await self._gemini(image_path)
            if provider == "tesseract":
                return await self._tesseract(image_path, lang, timeout_seconds)
            if provider == "pytesseract":
                return self._pytesseract(image_path, lang)
            if provider == "easyocr":
                return self._easyocr(image_path)
        except Exception as exc:
            logger.warning(f"[OCR] provider '{provider}' failed: {exc}")
            return OCRResult(status=OCR_ERROR, provider=provider,
                             reason=f"OCR provider '{provider}' failed: {exc}",
                             recommended_action="replan")
        # Unknown provider name (registry drift) — treat as blocked, not a crash.
        return self._blocked_result(cap)

    # -- providers -- #

    async def _gemini(self, image_path: str) -> OCRResult:
        """Extract text from an image with Gemini vision (preferred cloud OCR).

        Reuses the router's already-registered Gemini provider so the shared
        multi-key rotation pool and budget state are honoured; only builds a
        fresh provider from settings when the router has none registered.
        """
        provider = None
        try:
            from backend.providers.router import model_router
            provider = model_router.providers.get("gemini")
        except Exception:
            provider = None
        if provider is None:
            from backend.providers.real_providers import GeminiProvider
            from backend.config import settings
            keys = [k.strip() for k in (getattr(settings, "GEMINI_API_KEYS", "") or "").split(",") if k.strip()]
            provider = GeminiProvider(api_key=getattr(settings, "GEMINI_API_KEY", ""), api_keys=keys)

        resp = await provider.generate_response(
            prompt=(
                "You are an OCR engine. Transcribe EVERY character of readable text in "
                "this image exactly as it appears, preserving case, symbols and spacing. "
                "If a CTF flag is present, include it verbatim. Output only the raw text "
                "with no explanation."
            ),
            capability="ocr",
            image_path=image_path,
        )
        if getattr(resp, "is_refusal", False):
            reason = getattr(resp, "refusal_reason", "") or "Gemini vision refused the OCR request"
            return OCRResult(status=OCR_ERROR, provider="gemini",
                             reason=str(reason)[:300], recommended_action="replan")
        return OCRResult(status=OCR_OK, text=getattr(resp, "content", "") or "", provider="gemini")

    async def _tesseract(self, image_path: str, lang: str, timeout_seconds: int) -> OCRResult:
        from backend.execution.service import execution_service
        # `tesseract <img> stdout` writes recognized text to stdout.
        cmd = f'tesseract "{image_path}" stdout -l {lang}'
        res = await execution_service.run_command(cmd, timeout_seconds=timeout_seconds,
                                                  capability="ocr", tool_name="tesseract")
        if res.succeeded:
            return OCRResult(status=OCR_OK, text=res.stdout, provider="tesseract")
        return OCRResult(status=OCR_ERROR, provider="tesseract",
                         reason=(res.stderr or "tesseract returned no text")[:300],
                         recommended_action="replan")

    def _pytesseract(self, image_path: str, lang: str) -> OCRResult:
        import pytesseract          # provider presence already confirmed by discovery
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(image_path), lang=lang)
        return OCRResult(status=OCR_OK, text=text, provider="pytesseract")

    def _easyocr(self, image_path: str) -> OCRResult:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        parts = reader.readtext(image_path, detail=0)
        return OCRResult(status=OCR_OK, text="\n".join(parts), provider="easyocr")


# Module-level singleton.
ocr_service = OCRService()
