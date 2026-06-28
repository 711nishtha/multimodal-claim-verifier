"""Usage tracking for LLM API calls.

Counts every model call (provider, success/failure, fallback), approximate
input/output token counts, and images processed.  These counters feed into
evaluation_report.md.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CallRecord:
    """A single API call record."""

    provider: str
    model: str
    success: bool
    fallback: bool
    input_tokens: int
    output_tokens: int
    image_count: int
    latency_ms: float
    error: str | None = None


@dataclass
class UsageStats:
    """Aggregated usage statistics."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    fallback_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_images: int = 0
    total_latency_ms: float = 0.0
    by_provider: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, call: CallRecord) -> None:
        """Thread-safe record of a call."""
        with self._lock:
            self.total_calls += 1
            if call.success:
                self.successful_calls += 1
            else:
                self.failed_calls += 1
            if call.fallback:
                self.fallback_calls += 1
            self.total_input_tokens += call.input_tokens
            self.total_output_tokens += call.output_tokens
            self.total_images += call.image_count
            self.total_latency_ms += call.latency_ms

            provider = call.provider
            if provider not in self.by_provider:
                self.by_provider[provider] = {
                    "calls": 0,
                    "success": 0,
                    "failure": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "images": 0,
                }
            p = self.by_provider[provider]
            p["calls"] += 1
            if call.success:
                p["success"] += 1
            else:
                p["failure"] += 1
            p["input_tokens"] += call.input_tokens
            p["output_tokens"] += call.output_tokens
            p["images"] += call.image_count

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary dict."""
        with self._lock:
            avg_latency = (
                self.total_latency_ms / max(self.total_calls, 1)
            )
            return {
                "total_calls": self.total_calls,
                "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls,
                "fallback_calls": self.fallback_calls,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_images_processed": self.total_images,
                "avg_latency_ms": round(avg_latency, 2),
                "by_provider": dict(self.by_provider),
            }

    def write_json(self, path: Path) -> None:
        """Write the summary to a JSON file."""
        summary = self.summary()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Usage stats written to %s", path)

    def estimated_cost_usd(
        self,
        gemini_input_per_1m: float = 0.15,
        gemini_output_per_1m: float = 0.60,
        gemini_image_per_1k: float = 0.00025,
        groq_input_per_1m: float = 0.50,
        groq_output_per_1m: float = 0.75,
    ) -> float:
        """Return an approximate cost in USD based on token counts.

        Uses default Gemini 2.5 Flash Lite and Groq Llama-4 Scout pricing
        assumptions.  Override with actual pricing if available.
        """
        cost = 0.0
        for provider, stats in self.by_provider.items():
            p_lower = provider.lower()
            if "gemini" in p_lower:
                cost += (stats["input_tokens"] / 1_000_000) * gemini_input_per_1m
                cost += (stats["output_tokens"] / 1_000_000) * gemini_output_per_1m
                cost += (stats["images"] / 1_000) * gemini_image_per_1k
            elif "groq" in p_lower:
                cost += (stats["input_tokens"] / 1_000_000) * groq_input_per_1m
                cost += (stats["output_tokens"] / 1_000_000) * groq_output_per_1m
            else:
                # Generic fallback
                cost += (stats["input_tokens"] / 1_000_000) * 0.50
                cost += (stats["output_tokens"] / 1_000_000) * 0.75
        return round(cost, 4)


# Global singleton for convenience
_GLOBAL_STATS: UsageStats | None = None


def get_global_stats() -> UsageStats:
    """Return the global UsageStats singleton (created on first call)."""
    global _GLOBAL_STATS
    if _GLOBAL_STATS is None:
        _GLOBAL_STATS = UsageStats()
    return _GLOBAL_STATS


def reset_global_stats() -> None:
    """Reset the global stats (useful between evaluation runs)."""
    global _GLOBAL_STATS
    _GLOBAL_STATS = UsageStats()
