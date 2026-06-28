"""Deterministic safety gate.

Operates on structured vision observations BEFORE verification.
If any image's observations indicate instruction-like text directed at a
reviewer/system ("approve", "skip review", "ignore all previous instructions",
etc.), force risk_flags to include text_instruction_present and explicitly
exclude that text from any input the verification step uses.

This gate is NOT a suggestion the model can override — it is a real code-level
constraint.  The verification step's prompt also restates this rule as defense
in depth, but the enforcement happens here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from vision import ImageObservations

logger = logging.getLogger(__name__)

# Instruction-like phrases that should NEVER be treated as system commands.
# These are detected case-insensitively.
_INSTRUCTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bapprove\s+(this\s+)?claim\b", re.IGNORECASE),
    re.compile(r"\bskip\s+(review|manual\s+review|check)\b", re.IGNORECASE),
    re.compile(r"\bignore\s+all\s+previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bmark\s+(this\s+row\s+)?as\s+(supported|approved)\b", re.IGNORECASE),
    re.compile(r"\bfollow\s+(the\s+note|it)\s+and\s+approve\b", re.IGNORECASE),
    re.compile(r"\baccept\s+this\s+quickly\b", re.IGNORECASE),
    re.compile(r"\bapprove\s+this\b", re.IGNORECASE),
    re.compile(r"\bsystem\s+should\s+approve\b", re.IGNORECASE),
    re.compile(r"\bmark\s+.*\s+(supported|contradicted|not_enough_information)\b", re.IGNORECASE),
    re.compile(r"\bwith\s+(low|medium|high)\s+severity\b", re.IGNORECASE),
]

# Words/phrases that are strong signals of an instruction embedded in the image
_DIRECTIVE_KEYWORDS = [
    "approve", "skip review", "ignore all previous instructions",
    "mark this", "system should", "accept this quickly",
    "follow the note", "approve this claim",
]


@dataclass(frozen=True)
class GatedObservations:
    """Observations after the safety gate has run."""

    observations: list[ImageObservations]
    forced_risk_flags: list[str] = field(default_factory=list)
    instruction_text_found: list[str] = field(default_factory=list)
    # Sanitized descriptions with instruction text redacted
    sanitized_descriptions: dict[str, str] = field(default_factory=dict)


def _contains_instruction(text: str) -> bool:
    """Return True if *text* contains instruction-like language."""
    if not text:
        return False
    for pattern in _INSTRUCTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _redact_instruction_text(text: str) -> str:
    """Redact instruction-like phrases from text."""
    if not text:
        return text
    redacted = text
    for pattern in _INSTRUCTION_PATTERNS:
        redacted = pattern.sub("[REDACTED_INSTRUCTION]", redacted)
    return redacted


def apply_safety_gate(observations: list[ImageObservations]) -> GatedObservations:
    """Run the deterministic safety gate on a list of image observations.

    Returns:
        GatedObservations with:
        - forced_risk_flags: always includes "text_instruction_present" if any
          instruction text was found.
        - instruction_text_found: the actual text snippets that triggered.
        - sanitized_descriptions: image_id -> description with instructions redacted.
    """
    forced_flags: list[str] = []
    instruction_texts: list[str] = []
    sanitized: dict[str, str] = {}

    for obs in observations:
        # Check text_detected field
        detected = obs.text_detected or ""
        desc = obs.visual_description or ""

        has_instruction = False
        trigger_text = ""

        if _contains_instruction(detected):
            has_instruction = True
            trigger_text = detected

        if _contains_instruction(desc):
            has_instruction = True
            if not trigger_text:
                trigger_text = desc

        # Sanitize the description regardless
        sanitized_desc = _redact_instruction_text(desc)
        sanitized[obs.image_id] = sanitized_desc

        if has_instruction:
            instruction_texts.append(trigger_text[:200])
            logger.warning(
                "Safety gate: instruction-like text detected in image %s: %s",
                obs.image_id, trigger_text[:200],
            )

    if instruction_texts:
        forced_flags.append("text_instruction_present")

    return GatedObservations(
        observations=observations,
        forced_risk_flags=forced_flags,
        instruction_text_found=instruction_texts,
        sanitized_descriptions=sanitized,
    )
