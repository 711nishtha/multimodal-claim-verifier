"""Output writer — assemble the exact 14-column CSV.

Columns, in exact order:
user_id, image_paths, user_claim, claim_object, evidence_standard_met,
evidence_standard_met_reason, risk_flags, issue_type, object_part,
claim_status, claim_status_justification, supporting_image_ids, valid_image,
severity
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from data_loading import ClaimContext
from verification import VerificationResult

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def write_output(
    results: list[tuple[ClaimContext, VerificationResult]],
    output_path: Path,
) -> None:
    """Write results to output.csv with the exact required schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for ctx, result in results:
            row: dict[str, Any] = {
                "user_id": ctx.user_id,
                "image_paths": ctx.image_paths,
                "user_claim": ctx.user_claim,
                "claim_object": ctx.claim_object,
                "evidence_standard_met": "true" if result.evidence_standard_met else "false",
                "evidence_standard_met_reason": result.evidence_standard_met_reason,
                "risk_flags": result.risk_flags,
                "issue_type": result.issue_type,
                "object_part": result.object_part,
                "claim_status": result.claim_status,
                "claim_status_justification": result.claim_status_justification,
                "supporting_image_ids": result.supporting_image_ids,
                "valid_image": "true" if result.valid_image else "false",
                "severity": result.severity,
            }
            writer.writerow(row)

    logger.info("Wrote %d rows to %s", len(results), output_path)
