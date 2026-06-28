"""Main orchestrator — run the full pipeline end to end.

Reads dataset/claims.csv, processes each claim through the full pipeline,
and writes output.csv.

Usage:
    cd code && python main.py

Environment variables:
    GEMINI_API_KEY  - primary VLM API key
    GROQ_API_KEY    - fallback VLM API key
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Ensure our code directory is importable
sys.path.insert(0, str(Path(__file__).parent))

from claim_parsing import parse_claim
from data_loading import load_all
from output_writer import write_output
from planning import resolve_requirements
from safety_gate import apply_safety_gate
from usage_tracker import get_global_stats, reset_global_stats
from verification import verify_claim
from vision import analyze_all_images

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(config: dict) -> list:
    """Run the full claims processing pipeline.

    Returns list of (ClaimContext, VerificationResult) tuples.
    """
    # Reset stats for this run
    reset_global_stats()
    stats = get_global_stats()

    # Module 1: Data Loading
    logger.info("=== Module 1: Loading data ===")
    claims = load_all(config)
    logger.info("Loaded %d claims", len(claims))

    results = []

    for idx, claim in enumerate(claims):
        logger.info(
            "--- Processing claim %d/%d (row %d, user %s, %s) ---",
            idx + 1, len(claims), claim.row_index, claim.user_id, claim.claim_object,
        )

        if claim.load_errors:
            logger.warning("Load errors for row %d: %s", claim.row_index, claim.load_errors)

        # Step 1: Claim Parsing
        logger.info("Step 1: Parsing claim...")
        parsed = parse_claim(claim)
        logger.info(
            "Parsed: issue=%s, part=%s, severity=%s",
            parsed.issue_type, parsed.object_part, parsed.described_severity,
        )

        # Step 2: Planning — resolve applicable evidence requirements
        logger.info("Step 2: Resolving requirements...")
        resolved_reqs = resolve_requirements(parsed, claim)
        logger.info("Applicable requirements: %d", len(resolved_reqs))

        # Step 3: Vision — per-image structured observations
        logger.info("Step 3: Analyzing %d image(s)...", len(claim.resolved_image_paths))
        observations = analyze_all_images(claim, parsed)

        # Step 4: Safety Gate — deterministic instruction detection
        logger.info("Step 4: Running safety gate...")
        gated = apply_safety_gate(observations)
        if gated.forced_risk_flags:
            logger.warning(
                "Safety gate forced flags: %s", gated.forced_risk_flags
            )

        # Step 5: Verification — final decision
        logger.info("Step 5: Verifying claim...")
        result = verify_claim(claim, parsed, gated, resolved_reqs)
        logger.info(
            "Result: status=%s, evidence_met=%s, valid_image=%s, severity=%s, risks=%s",
            result.claim_status,
            result.evidence_standard_met,
            result.valid_image,
            result.severity,
            result.risk_flags,
        )

        results.append((claim, result))

    return results


def main() -> None:
    """Entry point."""
    # Determine project paths relative to this file, independent of cwd.
    code_dir = Path(__file__).parent.resolve()
    project_root = code_dir.parent

    # Check API keys
    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY not set — will rely on Groq fallback")
    if not os.environ.get("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not set — no fallback available")

    config = {
        "dataset_root": str(project_root),
        "claims_csv": "dataset/claims.csv",
        "user_history_csv": "dataset/user_history.csv",
        "evidence_requirements_csv": "dataset/evidence_requirements.csv",
    }

    start_time = time.time()
    results = run_pipeline(config)
    elapsed = time.time() - start_time

    # Write output
    output_path = project_root / "output.csv"
    write_output(results, output_path)

    # Log stats summary
    stats = get_global_stats()
    summary = stats.summary()
    logger.info("=== Pipeline complete ===")
    logger.info("Runtime: %.1f seconds", elapsed)
    logger.info("Total API calls: %d", summary["total_calls"])
    logger.info("Successful calls: %d", summary["successful_calls"])
    logger.info("Failed calls: %d", summary["failed_calls"])
    logger.info("Fallback calls: %d", summary["fallback_calls"])
    logger.info("Input tokens: %d", summary["total_input_tokens"])
    logger.info("Output tokens: %d", summary["total_output_tokens"])
    logger.info("Images processed: %d", summary["total_images_processed"])
    logger.info("Approximate cost: $%.4f", stats.estimated_cost_usd())

    # Write stats JSON alongside output
    stats_path = code_dir / ".vlm_cache" / "usage_stats.json"
    stats.write_json(stats_path)


if __name__ == "__main__":
    main()
