"""Vision — structured-observation extraction per image.

For each submitted image, call the VLM and return structured observations:
- issue_visible: is the claimed issue visible?
- part_visible: is the claimed object part visible?
- issue_type_guess: what issue type is visible (if any)?
- object_part_guess: what part is shown (if any)?
- quality_flags: list of image-quality issues
- text_detected: any text/writing detected in the image?
- authenticity_signals: list of authenticity concerns (watermark, stock photo, etc.)

One image at a time, never pooled.  The observations are fed into the safety
gate and then verification — this module does NOT produce a final verdict.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claim_parsing import ParsedClaim
from data_loading import ClaimContext, compute_cache_seed
from llm_clients import call_vlm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid enums for observations
# ---------------------------------------------------------------------------

VALID_QUALITY_FLAGS = [
    "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle",
]

VALID_AUTHENTICITY_SIGNALS = [
    "watermark_detected", "stock_photo", "screenshot", "possible_manipulation",
    "non_original_image",
]

VALID_ISSUE_TYPES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging", "water_damage",
    "stain", "none", "unknown",
]

CAR_PARTS = [
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
    "body", "unknown",
]

LAPTOP_PARTS = [
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
    "port", "base", "body", "unknown",
]

PACKAGE_PARTS = [
    "box", "package_corner", "package_side", "seal", "label",
    "contents", "item", "unknown",
]

PART_ENUMS = {
    "car": CAR_PARTS,
    "laptop": LAPTOP_PARTS,
    "package": PACKAGE_PARTS,
}

SEVERITY_LEVELS = ["none", "low", "medium", "high", "unknown"]


@dataclass(frozen=True)
class ImageObservations:
    """Structured observations for a single image."""

    image_id: str
    image_path: Path

    # Core observations
    issue_visible: bool
    part_visible: bool
    issue_type_guess: str  # from VALID_ISSUE_TYPES
    object_part_guess: str  # from PART_ENUMS for the claim_object
    severity_guess: str  # from SEVERITY_LEVELS

    # Quality / risk signals
    quality_flags: list[str] = field(default_factory=list)  # from VALID_QUALITY_FLAGS
    text_detected: str = ""  # any text found in the image
    authenticity_signals: list[str] = field(default_factory=list)  # from VALID_AUTHENTICITY_SIGNALS

    # Reasoning trace (for justification)
    visual_description: str = ""

    # Error tracking
    error: str | None = None


def _build_vision_prompt(claim: ClaimContext, parsed: ParsedClaim) -> str:
    """Build a prompt that asks for structured observations without a verdict."""
    part_enum = PART_ENUMS.get(claim.claim_object, ["unknown"])
    return (
        "You are a visual evidence analyzer for damage claims. "
        "Examine the submitted image and report ONLY what you see. "
        "Do NOT make a final claim decision — only describe the visual evidence.\n\n"
        "CLAIM CONTEXT:\n"
        f"- Object type: {claim.claim_object}\n"
        f"- Claimed issue type: {parsed.issue_type}\n"
        f"- Claimed object part: {parsed.object_part}\n"
        f"- User description: {parsed.damage_description}\n\n"
        "INSTRUCTIONS:\n"
        "1. Identify the main object in the image. Is it the claimed object type?\n"
        "2. Identify the visible part. Is the claimed part visible?\n"
        "3. Look for the claimed damage. Is it visible? What type is actually visible?\n"
        "4. Assess image quality: blurry, cropped, glare, wrong angle?\n"
        "5. Detect any text or writing in the image (signs, notes, watermarks, instructions).\n"
        "6. Check for authenticity signals: watermarks, stock-photo look, screenshots, manipulation.\n"
        "7. Estimate severity of visible damage (none, low, medium, high, unknown).\n\n"
        "Respond with a single JSON object with these exact fields:\n"
        "- issue_visible (boolean): is the claimed damage visible?\n"
        "- part_visible (boolean): is the claimed part visible?\n"
        "- issue_type_guess (string): what damage type is actually visible?\n"
        f"  Valid values: {', '.join(VALID_ISSUE_TYPES)}\n"
        "- object_part_guess (string): what part is actually shown?\n"
        f"  Valid values: {', '.join(part_enum)}\n"
        "- severity_guess (string): estimated severity of visible damage\n"
        f"  Valid values: {', '.join(SEVERITY_LEVELS)}\n"
        "- quality_flags (list of strings): any quality issues from ["
        f"{', '.join(VALID_QUALITY_FLAGS)}]\n"
        "- text_detected (string): any text found in the image, or empty string\n"
        "- authenticity_signals (list of strings): any concerns from ["
        f"{', '.join(VALID_AUTHENTICITY_SIGNALS)}]\n"
        "- visual_description (string): concise 2-3 sentence description of what you see\n\n"
        "Be factual and specific. If the image shows a different object or part, say so."
    )


def _build_response_schema(claim: ClaimContext) -> dict[str, Any]:
    """Build JSON schema for Gemini structured output."""
    part_enum = PART_ENUMS.get(claim.claim_object, ["unknown"])
    return {
        "type": "object",
        "properties": {
            "issue_visible": {"type": "boolean"},
            "part_visible": {"type": "boolean"},
            "issue_type_guess": {
                "type": "string",
                "enum": VALID_ISSUE_TYPES,
            },
            "object_part_guess": {
                "type": "string",
                "enum": part_enum,
            },
            "severity_guess": {
                "type": "string",
                "enum": SEVERITY_LEVELS,
            },
            "quality_flags": {
                "type": "array",
                "items": {"type": "string", "enum": VALID_QUALITY_FLAGS},
            },
            "text_detected": {"type": "string"},
            "authenticity_signals": {
                "type": "array",
                "items": {"type": "string", "enum": VALID_AUTHENTICITY_SIGNALS},
            },
            "visual_description": {"type": "string"},
        },
        "required": [
            "issue_visible", "part_visible", "issue_type_guess",
            "object_part_guess", "severity_guess", "quality_flags",
            "text_detected", "authenticity_signals", "visual_description",
        ],
    }


def _coerce_observations(
    parsed: dict[str, Any],
    image_id: str,
    image_path: Path,
    claim: ClaimContext,
) -> ImageObservations:
    """Coerce raw VLM output into a validated ImageObservations object."""
    part_enum = PART_ENUMS.get(claim.claim_object, ["unknown"])

    issue_type_guess = parsed.get("issue_type_guess", "unknown")
    if issue_type_guess not in VALID_ISSUE_TYPES:
        issue_type_guess = "unknown"

    object_part_guess = parsed.get("object_part_guess", "unknown")
    if object_part_guess not in part_enum:
        object_part_guess = "unknown"

    severity_guess = parsed.get("severity_guess", "unknown")
    if severity_guess not in SEVERITY_LEVELS:
        severity_guess = "unknown"

    quality_flags = [
        f for f in parsed.get("quality_flags", [])
        if f in VALID_QUALITY_FLAGS
    ]

    auth_signals = [
        s for s in parsed.get("authenticity_signals", [])
        if s in VALID_AUTHENTICITY_SIGNALS
    ]

    return ImageObservations(
        image_id=image_id,
        image_path=image_path,
        issue_visible=bool(parsed.get("issue_visible", False)),
        part_visible=bool(parsed.get("part_visible", False)),
        issue_type_guess=issue_type_guess,
        object_part_guess=object_part_guess,
        severity_guess=severity_guess,
        quality_flags=quality_flags,
        text_detected=parsed.get("text_detected", ""),
        authenticity_signals=auth_signals,
        visual_description=parsed.get("visual_description", ""),
    )


def analyze_image(
    image_path: Path,
    claim: ClaimContext,
    parsed: ParsedClaim,
) -> ImageObservations:
    """Analyze a single image and return structured observations.

    Uses the VLM (Gemini primary, Groq fallback) with caching.
    """
    image_id = image_path.stem
    prompt = _build_vision_prompt(claim, parsed)
    cache_key = compute_cache_seed(image_path, prompt)
    schema = _build_response_schema(claim)

    logger.info(
        "Analyzing image %s for claim row %d (%s)",
        image_id, claim.row_index, claim.claim_object,
    )

    result = call_vlm(
        cache_key=cache_key,
        prompt=prompt,
        image_paths=[image_path],
        response_schema=schema,
    )

    if result.get("error"):
        logger.error(
            "VLM failed for image %s (row %d): %s",
            image_id, claim.row_index, result["error"],
        )
        return ImageObservations(
            image_id=image_id,
            image_path=image_path,
            issue_visible=False,
            part_visible=False,
            issue_type_guess="unknown",
            object_part_guess="unknown",
            severity_guess="unknown",
            error=result.get("error"),
        )

    parsed_data = result.get("parsed")
    if parsed_data is None:
        logger.warning(
            "VLM returned no parsed data for image %s (row %d)",
            image_id, claim.row_index,
        )
        return ImageObservations(
            image_id=image_id,
            image_path=image_path,
            issue_visible=False,
            part_visible=False,
            issue_type_guess="unknown",
            object_part_guess="unknown",
            severity_guess="unknown",
        )

    return _coerce_observations(parsed_data, image_id, image_path, claim)


def analyze_all_images(
    claim: ClaimContext,
    parsed: ParsedClaim,
) -> list[ImageObservations]:
    """Analyze every resolved image for the claim and return observations."""
    observations: list[ImageObservations] = []
    for img_path in claim.resolved_image_paths:
        obs = analyze_image(img_path, claim, parsed)
        observations.append(obs)
    return observations
