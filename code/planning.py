"""Planning — resolve which evidence requirements apply to a parsed claim.

Given the parsed claim (issue_type, object_part) and the set of requirements
already filtered to the correct claim_object, this module returns the specific
requirement rows that should be checked.

This is a rule-based, deterministic module — no LLM calls.
"""

from __future__ import annotations

import logging
from typing import Any

from claim_parsing import ParsedClaim
from data_loading import ClaimContext

logger = logging.getLogger(__name__)

# Mapping from issue_type -> requirement applies_to keywords
_ISSUE_TO_KEYWORDS: dict[str, list[str]] = {
    "dent": ["dent", "scratch", "body panel", "surface marks", "deformation"],
    "scratch": ["dent", "scratch", "body panel", "surface marks", "deformation"],
    "crack": ["crack", "broken", "missing part", "glass", "light", "mirror"],
    "glass_shatter": ["crack", "broken", "missing part", "glass", "light", "mirror"],
    "broken_part": ["crack", "broken", "missing part", "glass", "light", "mirror", "hinge", "lid", "corner", "body", "port", "base"],
    "missing_part": ["crack", "broken", "missing part"],
    "torn_packaging": ["crushed", "torn", "seal", "exterior"],
    "crushed_packaging": ["crushed", "torn", "seal", "exterior"],
    "water_damage": ["water", "stain", "label", "surface"],
    "stain": ["water", "stain", "label", "surface"],
    "none": ["general claim review", "reviewability"],
    "unknown": ["general claim review", "reviewability"],
}

# Object-specific requirement IDs that depend on the claimed part
_PART_REQUIREMENT_IDS: dict[str, dict[str, list[str]]] = {
    "car": {
        "screen": [],
        "keyboard": [],
        "trackpad": [],
        "hinge": [],
        "lid": [],
        "corner": [],
        "port": [],
        "base": [],
        "body": ["REQ_CAR_BODY_PANEL", "REQ_REVIEW_TRUST"],
        "front_bumper": ["REQ_CAR_BODY_PANEL", "REQ_CAR_IDENTITY_OR_SIDE", "REQ_REVIEW_TRUST"],
        "rear_bumper": ["REQ_CAR_BODY_PANEL", "REQ_CAR_IDENTITY_OR_SIDE", "REQ_REVIEW_TRUST"],
        "door": ["REQ_CAR_BODY_PANEL", "REQ_CAR_IDENTITY_OR_SIDE", "REQ_REVIEW_TRUST"],
        "hood": ["REQ_CAR_BODY_PANEL", "REQ_REVIEW_TRUST"],
        "windshield": ["REQ_CAR_GLASS_LIGHT_MIRROR", "REQ_REVIEW_TRUST"],
        "side_mirror": ["REQ_CAR_GLASS_LIGHT_MIRROR", "REQ_REVIEW_TRUST"],
        "headlight": ["REQ_CAR_GLASS_LIGHT_MIRROR", "REQ_REVIEW_TRUST"],
        "taillight": ["REQ_CAR_GLASS_LIGHT_MIRROR", "REQ_REVIEW_TRUST"],
        "fender": ["REQ_CAR_BODY_PANEL", "REQ_REVIEW_TRUST"],
        "quarter_panel": ["REQ_CAR_BODY_PANEL", "REQ_REVIEW_TRUST"],
        "unknown": ["REQ_GENERAL_OBJECT_PART", "REQ_REVIEW_TRUST"],
    },
    "laptop": {
        "screen": ["REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD", "REQ_REVIEW_TRUST"],
        "keyboard": ["REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD", "REQ_REVIEW_TRUST"],
        "trackpad": ["REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD", "REQ_REVIEW_TRUST"],
        "hinge": ["REQ_LAPTOP_BODY_HINGE_PORT", "REQ_REVIEW_TRUST"],
        "lid": ["REQ_LAPTOP_BODY_HINGE_PORT", "REQ_REVIEW_TRUST"],
        "corner": ["REQ_LAPTOP_BODY_HINGE_PORT", "REQ_REVIEW_TRUST"],
        "port": ["REQ_LAPTOP_BODY_HINGE_PORT", "REQ_REVIEW_TRUST"],
        "base": ["REQ_LAPTOP_BODY_HINGE_PORT", "REQ_REVIEW_TRUST"],
        "body": ["REQ_LAPTOP_BODY_HINGE_PORT", "REQ_REVIEW_TRUST"],
        "unknown": ["REQ_GENERAL_OBJECT_PART", "REQ_REVIEW_TRUST"],
    },
    "package": {
        "box": ["REQ_PACKAGE_EXTERIOR", "REQ_REVIEW_TRUST"],
        "package_corner": ["REQ_PACKAGE_EXTERIOR", "REQ_REVIEW_TRUST"],
        "package_side": ["REQ_PACKAGE_EXTERIOR", "REQ_PACKAGE_LABEL_OR_STAIN", "REQ_REVIEW_TRUST"],
        "seal": ["REQ_PACKAGE_EXTERIOR", "REQ_REVIEW_TRUST"],
        "label": ["REQ_PACKAGE_LABEL_OR_STAIN", "REQ_REVIEW_TRUST"],
        "contents": ["REQ_PACKAGE_CONTENTS", "REQ_REVIEW_TRUST"],
        "item": ["REQ_PACKAGE_CONTENTS", "REQ_REVIEW_TRUST"],
        "unknown": ["REQ_GENERAL_OBJECT_PART", "REQ_REVIEW_TRUST"],
    },
}


def resolve_requirements(
    parsed: ParsedClaim,
    claim: ClaimContext,
) -> list[dict[str, str]]:
    """Return the subset of *claim.applicable_requirements* that apply to this claim.

    Uses the parsed issue_type and object_part to match against requirement
    keywords and requirement IDs.  Always includes general requirements.
    """
    if not claim.applicable_requirements:
        return []

    # Collect requirement IDs that match by part
    part_map = _PART_REQUIREMENT_IDS.get(claim.claim_object, {})
    matched_ids = set(part_map.get(parsed.object_part, []))

    # Also match by issue_type keywords against requirement applies_to field
    issue_keywords = _ISSUE_TO_KEYWORDS.get(parsed.issue_type, [])
    for req in claim.applicable_requirements:
        req_id = req.get("requirement_id", "")
        applies_to = req.get("applies_to", "").lower()
        # Always include general requirements
        if req_id.startswith("REQ_GENERAL") or req_id == "REQ_REVIEW_TRUST":
            matched_ids.add(req_id)
            continue
        # Check keyword match
        for kw in issue_keywords:
            if kw.lower() in applies_to:
                matched_ids.add(req_id)
                break

    # Filter the applicable_requirements list
    result = [
        req for req in claim.applicable_requirements
        if req.get("requirement_id", "") in matched_ids
    ]

    # If we somehow got nothing, return all applicable as fallback
    if not result:
        result = list(claim.applicable_requirements)

    return result
