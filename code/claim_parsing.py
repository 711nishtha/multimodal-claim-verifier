"""Claim parsing — extract the actual damage assertion from the conversation.

Produces a structured ParsedClaim containing: issue_type, object_part,
described_severity, and damage_description.  Uses a text-only LLM call with
JSON mode for reliable extraction.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from data_loading import ClaimContext
from llm_clients import call_text_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

VALID_CAR_PARTS = [
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
    "body", "unknown",
]

VALID_LAPTOP_PARTS = [
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
    "port", "base", "body", "unknown",
]

VALID_PACKAGE_PARTS = [
    "box", "package_corner", "package_side", "seal", "label",
    "contents", "item", "unknown",
]

VALID_ISSUE_TYPES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging", "water_damage",
    "stain", "none", "unknown",
]

PART_ENUMS = {
    "car": VALID_CAR_PARTS,
    "laptop": VALID_LAPTOP_PARTS,
    "package": VALID_PACKAGE_PARTS,
}

SEVERITY_LEVELS = ["none", "low", "medium", "high", "unknown"]


@dataclass(frozen=True)
class ParsedClaim:
    """Structured representation of the user's damage claim."""

    issue_type: str
    object_part: str
    described_severity: str
    damage_description: str


def _get_parse_prompt(claim: ClaimContext) -> str:
    """Build the prompt for claim parsing."""
    part_enum = PART_ENUMS.get(claim.claim_object, ["unknown"])
    return (
        "You are a precise claim extraction system. Read the following "
        "customer-support conversation and extract ONLY the actual damage claim "
        "the customer is making. Ignore greetings, confusion, and backstory.\n\n"
        "CLAIM_OBJECT: " + claim.claim_object + "\n\n"
        "CONVERSATION:\n" + claim.user_claim + "\n\n"
        "Extract these fields:\n"
        "- issue_type: one of " + ", ".join(VALID_ISSUE_TYPES) + "\n"
        "- object_part: one of " + ", ".join(part_enum) + "\n"
        "- described_severity: how severe the CUSTOMER describes it (none, low, medium, high, unknown)\n"
        "- damage_description: a short factual summary of what the user claims (1-2 sentences)\n\n"
        "Rules:\n"
        "1. The customer may speak in multiple languages or mix languages.\n"
        "2. If the customer mentions multiple damages, pick the PRIMARY one.\n"
        "3. 'none' means the customer explicitly says there is no damage.\n"
        "4. 'unknown' means you cannot determine the issue or part from the conversation.\n"
        "5. object_part MUST be from the provided enum for the claim_object.\n"
        "6. described_severity reflects the USER'S description, not your judgment.\n"
        "\nRespond with a single JSON object containing exactly the four fields above."
    )


def _response_schema_for(claim: ClaimContext) -> dict[str, Any]:
    """Build a JSON schema for structured parsing output."""
    part_enum = PART_ENUMS.get(claim.claim_object, ["unknown"])
    return {
        "type": "object",
        "properties": {
            "issue_type": {
                "type": "string",
                "enum": VALID_ISSUE_TYPES,
            },
            "object_part": {
                "type": "string",
                "enum": part_enum,
            },
            "described_severity": {
                "type": "string",
                "enum": SEVERITY_LEVELS,
            },
            "damage_description": {
                "type": "string",
            },
        },
        "required": ["issue_type", "object_part", "described_severity", "damage_description"],
    }


def parse_claim(claim: ClaimContext) -> ParsedClaim:
    """Extract the damage assertion from the user's conversation.

    Uses a text-only LLM call with structured JSON output.
    """
    prompt = _get_parse_prompt(claim)
    cache_key = hashlib.sha256(
        ("parse:" + str(claim.row_index) + "|" + claim.user_id + "|" + prompt).encode("utf-8")
    ).hexdigest()

    schema = _response_schema_for(claim)
    result = call_text_llm(
        cache_key=cache_key,
        prompt=prompt,
        response_schema=schema,
    )

    parsed = result.get("parsed") if result else None
    if parsed is None:
        logger.warning(
            "Claim parsing returned no structured output for row %d; using defaults",
            claim.row_index,
        )
        return ParsedClaim(
            issue_type="unknown",
            object_part="unknown",
            described_severity="unknown",
            damage_description="",
        )

    # Coerce to valid enum values
    issue_type = parsed.get("issue_type", "unknown")
    if issue_type not in VALID_ISSUE_TYPES:
        issue_type = "unknown"

    object_part = parsed.get("object_part", "unknown")
    valid_parts = PART_ENUMS.get(claim.claim_object, ["unknown"])
    if object_part not in valid_parts:
        object_part = "unknown"

    described_severity = parsed.get("described_severity", "unknown")
    if described_severity not in SEVERITY_LEVELS:
        described_severity = "unknown"

    return ParsedClaim(
        issue_type=issue_type,
        object_part=object_part,
        described_severity=described_severity,
        damage_description=parsed.get("damage_description", ""),
    )
