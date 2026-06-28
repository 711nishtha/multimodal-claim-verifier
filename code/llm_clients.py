"""LLM client module with caching, retry, and provider fallback.

Primary: Gemini 2.5 Flash via the Gemini API.
Fallback: Groq, model meta-llama/llama-4-scout-17b-16e-instruct (multimodal,
    supports JSON mode, max 5 images per request — safe because dataset never
    exceeds 3 images per claim).

All API keys are read from environment variables only.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

import requests

from usage_tracker import CallRecord, get_global_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loader — no python-dotenv dependency needed
# ---------------------------------------------------------------------------


def _load_dotenv(path: Path | None = None) -> None:
    """Parse a .env file and set each KEY=VALUE as an environment variable.

    Skips blank lines, comments (lines starting with #), and already-set
    variables (existing env vars take precedence).
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Support optional "export " prefix (bash convention)
            line = re.sub(r"^export\s+", "", line, flags=re.IGNORECASE)
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
_GROQ_API_KEY_ENV = "GROQ_API_KEY"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Rate limiting — 8 calls/min window, then 50-60s cool-off
_RATE_LIMIT_CALLS = 8
_RATE_LIMIT_WINDOW_SEC = 60
_RATE_COOLOFF_MIN = 50
_RATE_COOLOFF_MAX = 60
_call_timestamps: deque[float] = deque()


def _throttle_if_needed() -> None:
    """Block until we are allowed to make another API call.

    Enforces a sliding window: at most *_RATE_LIMIT_CALLS* requests per
    *_RATE_LIMIT_WINDOW_SEC* seconds.  When the window is full the caller
    is blocked for a randomised cool-off period before the window resets.
    """
    now = time.monotonic()
    # Purge timestamps older than the window
    cutoff = now - _RATE_LIMIT_WINDOW_SEC
    while _call_timestamps and _call_timestamps[0] < cutoff:
        _call_timestamps.popleft()

    if len(_call_timestamps) >= _RATE_LIMIT_CALLS:
        cool_off = random.uniform(_RATE_COOLOFF_MIN, _RATE_COOLOFF_MAX)
        logger.info(
            "Rate limit hit (%d calls in %ds) — cooling off for %.1fs",
            _RATE_LIMIT_CALLS,
            _RATE_LIMIT_WINDOW_SEC,
            cool_off,
        )
        time.sleep(cool_off)
        # Reset window after cool-off
        _call_timestamps.clear()
        _call_timestamps.append(time.monotonic())
        return

    _call_timestamps.append(now)


_MAX_RETRIES_PRIMARY = 3
_MAX_RETRIES_FALLBACK = 2
_BACKOFF_BASE_MS = 1000  # start at 1s

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    d = Path(os.environ.get("VLM_CACHE_DIR", _PROJECT_ROOT / ".vlm_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(cache_key: str) -> Path:
    return _cache_dir() / f"{cache_key}.json"


def _lookup_cache(cache_key: str) -> dict[str, Any] | None:
    path = _cache_path(cache_key)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


def _write_cache(cache_key: str, payload: dict[str, Any]) -> None:
    path = _cache_path(cache_key)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception as exc:
        logger.warning("Failed to write cache %s: %s", cache_key, exc)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _encode_image_base64(image_path: Path) -> tuple[str, str]:
    """Return (mime_type, base64_data) for the image at *image_path*."""
    mime, _ = mimetypes.guess_type(str(image_path))
    mime = mime or "image/jpeg"
    data = image_path.read_bytes()
    return mime, base64.b64encode(data).decode("utf-8")


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------


def _call_gemini(
    prompt: str,
    image_paths: list[Path],
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call Gemini 2.5 Flash with text + images, returning parsed JSON.

    Raises RuntimeError on failure (caller handles retry / fallback).
    """
    api_key = os.environ.get(_GEMINI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{_GEMINI_API_KEY_ENV} not set")

    url = _GEMINI_URL_TEMPLATE.format(model=_GEMINI_MODEL, key=api_key)

    parts: list[dict[str, Any]] = [{"text": prompt}]
    for img_path in image_paths:
        mime, b64 = _encode_image_base64(img_path)
        parts.append({"inline_data": {"mime_type": mime, "data": b64}})

    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.95,
        },
    }

    if response_schema is not None:
        body["generationConfig"]["responseMimeType"] = "application/json"
        body["generationConfig"]["responseSchema"] = response_schema

    resp = requests.post(url, json=body, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")

    content = candidates[0].get("content", {})
    content_parts = content.get("parts", [])
    if not content_parts:
        raise RuntimeError("Gemini returned empty content parts")

    # Gemini may return JSON inline in the text part
    text = content_parts[0].get("text", "")
    if text:
        # Clean markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            parsed = json.loads(cleaned)
            return {
                "parsed": parsed,
                "raw": text,
                "usage": data.get("usageMetadata", {}),
            }
        except json.JSONDecodeError:
            # Return raw text if JSON parsing fails
            return {"parsed": None, "raw": text, "usage": data.get("usageMetadata", {})}

    # Fallback: return the whole candidate structure
    return {"parsed": None, "raw": data, "usage": data.get("usageMetadata", {})}


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------


def _call_groq(
    prompt: str,
    image_paths: list[Path],
) -> dict[str, Any]:
    """Call Groq (OpenAI-compatible) with text + images, returning parsed JSON.

    Raises RuntimeError on failure.
    """
    api_key = os.environ.get(_GROQ_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{_GROQ_API_KEY_ENV} not set")

    groq_prompt = prompt
    if "json" not in groq_prompt:
        groq_prompt += "\nReturn output in json format."

    messages_content: list[dict[str, Any]] = [{"type": "text", "text": groq_prompt}]
    for img_path in image_paths:
        mime, b64 = _encode_image_base64(img_path)
        messages_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )

    body: dict[str, Any] = {
        "model": _GROQ_MODEL,
        "messages": [{"role": "user", "content": messages_content}],
        "temperature": 0.1,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(_GROQ_URL, json=body, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Groq returned no choices")

    message = choices[0].get("message", {})
    raw_content = message.get("content", "")
    try:
        parsed = json.loads(raw_content) if raw_content else None
    except json.JSONDecodeError:
        parsed = None

    usage = data.get("usage", {})
    return {
        "parsed": parsed,
        "raw": raw_content,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Unified client
# ---------------------------------------------------------------------------


def call_vlm(
    cache_key: str,
    prompt: str,
    image_paths: list[Path],
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call the VLM with caching, retry, and fallback.

    Args:
        cache_key: stable hash of (image content + prompt).  Used for the
            on-disk cache so identical (image, prompt) pairs never re-call.
        prompt: the text prompt sent to the model.
        image_paths: list of image file paths to include in the request.
        response_schema: optional JSON schema for Gemini structured output.

    Returns:
        A dict with at minimum a "parsed" key containing the JSON-decoded
        response.  Additional keys: "raw", "usage", "provider", "from_cache".
    """
    # 1. Check cache
    cached = _lookup_cache(cache_key)
    if cached is not None:
        logger.debug("Cache hit for key %s", cache_key[:16])
        cached["from_cache"] = True
        cached.setdefault("provider", "cached")
        return cached

    stats = get_global_stats()
    fallback = False
    last_error: str | None = None

    # 2. Try primary (Gemini) with retries
    for attempt in range(1, _MAX_RETRIES_PRIMARY + 1):
        _throttle_if_needed()
        start = time.time()
        try:
            result = _call_gemini(prompt, image_paths, response_schema=response_schema)
            latency = (time.time() - start) * 1000
            usage_meta = result.get("usage", {})
            # Gemini usageMetadata has promptTokenCount / candidatesTokenCount
            input_tokens = usage_meta.get("promptTokenCount", 0) or usage_meta.get(
                "prompt_tokens", 0
            )
            output_tokens = usage_meta.get("candidatesTokenCount", 0) or usage_meta.get(
                "completion_tokens", 0
            )

            stats.record(
                CallRecord(
                    provider="gemini",
                    model=_GEMINI_MODEL,
                    success=True,
                    fallback=False,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    image_count=len(image_paths),
                    latency_ms=latency,
                )
            )

            result["provider"] = "gemini"
            result["from_cache"] = False
            _write_cache(cache_key, result)
            return result

        except Exception as exc:
            last_error = str(exc)
            latency = (time.time() - start) * 1000
            logger.warning(
                "Gemini attempt %d/%d failed: %s",
                attempt,
                _MAX_RETRIES_PRIMARY,
                last_error[:300],
            )
            stats.record(
                CallRecord(
                    provider="gemini",
                    model=_GEMINI_MODEL,
                    success=False,
                    fallback=False,
                    input_tokens=0,
                    output_tokens=0,
                    image_count=len(image_paths),
                    latency_ms=latency,
                    error=last_error[:300],
                )
            )
            if attempt < _MAX_RETRIES_PRIMARY:
                backoff = _BACKOFF_BASE_MS * (2 ** (attempt - 1))
                logger.info("Backing off %d ms before retry", backoff)
                time.sleep(backoff / 1000.0)

    # 3. Fallback to Groq with retries
    fallback = True
    logger.info("Falling back to Groq after Gemini failed")
    for attempt in range(1, _MAX_RETRIES_FALLBACK + 1):
        _throttle_if_needed()
        start = time.time()
        try:
            result = _call_groq(prompt, image_paths)
            latency = (time.time() - start) * 1000
            usage_data = result.get("usage", {})
            input_tokens = usage_data.get("prompt_tokens", 0)
            output_tokens = usage_data.get("completion_tokens", 0)

            stats.record(
                CallRecord(
                    provider="groq",
                    model=_GROQ_MODEL,
                    success=True,
                    fallback=True,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    image_count=len(image_paths),
                    latency_ms=latency,
                )
            )

            result["provider"] = "groq"
            result["from_cache"] = False
            _write_cache(cache_key, result)
            return result

        except Exception as exc:
            last_error = str(exc)
            latency = (time.time() - start) * 1000
            logger.warning(
                "Groq attempt %d/%d failed: %s",
                attempt,
                _MAX_RETRIES_FALLBACK,
                last_error[:300],
            )
            stats.record(
                CallRecord(
                    provider="groq",
                    model=_GROQ_MODEL,
                    success=False,
                    fallback=True,
                    input_tokens=0,
                    output_tokens=0,
                    image_count=len(image_paths),
                    latency_ms=latency,
                    error=last_error[:300],
                )
            )
            if attempt < _MAX_RETRIES_FALLBACK:
                backoff = _BACKOFF_BASE_MS * (2 ** (attempt - 1))
                time.sleep(backoff / 1000.0)

    # 4. All retries exhausted — return a structured error so the pipeline can continue
    logger.error("All VLM retries exhausted for cache_key %s", cache_key[:16])
    return {
        "parsed": None,
        "raw": "",
        "usage": {},
        "provider": "none",
        "from_cache": False,
        "error": f"All retries exhausted. Last error: {last_error}"
        if last_error
        else "Unknown error",
    }


def call_text_llm(
    cache_key: str,
    prompt: str,
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call the text-only LLM with caching, retry, and fallback.

    Same semantics as call_vlm but without images.  Used for claim parsing
    and verification steps.
    """
    # Reuse the VLM call with empty image list — the underlying APIs handle text-only
    return call_vlm(
        cache_key=cache_key,
        prompt=prompt,
        image_paths=[],
        response_schema=response_schema,
    )
