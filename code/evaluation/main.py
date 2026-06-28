"""Evaluation — run the pipeline against sample_claims.csv and compare to expected outputs.

Usage:
    cd code && python -m evaluation.main

Computes per-field accuracy and supports comparing multiple configurations.
Writes evaluation_report.md to code/evaluation/.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure parent code/ directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from claim_parsing import parse_claim
from data_loading import load_all
from output_writer import OUTPUT_COLUMNS
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


def load_expected_outputs(sample_csv: Path) -> list[dict[str, str]]:
    """Load the expected outputs from sample_claims.csv."""
    rows: list[dict[str, str]] = []
    with sample_csv.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cleaned = {k.strip(): (v.strip() if v else "") for k, v in row.items()}
            rows.append(cleaned)
    return rows


def run_pipeline_for_evaluation(config: dict) -> list[dict[str, str]]:
    """Run the pipeline and return predictions as raw dict rows."""
    reset_global_stats()
    claims = load_all(config)
    predictions: list[dict[str, str]] = []

    for claim in claims:
        parsed = parse_claim(claim)
        resolved_reqs = resolve_requirements(parsed, claim)
        observations = analyze_all_images(claim, parsed)
        gated = apply_safety_gate(observations)
        result = verify_claim(claim, parsed, gated, resolved_reqs)

        row = {
            "user_id": claim.user_id,
            "image_paths": claim.image_paths,
            "user_claim": claim.user_claim,
            "claim_object": claim.claim_object,
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
        predictions.append(row)

    return predictions


def compute_accuracy(
    predictions: list[dict[str, str]],
    expected: list[dict[str, str]],
) -> dict[str, Any]:
    """Compute per-field accuracy and overall metrics."""
    fields = [
        "evidence_standard_met",
        "risk_flags",
        "issue_type",
        "object_part",
        "claim_status",
        "supporting_image_ids",
        "valid_image",
        "severity",
    ]

    if len(predictions) != len(expected):
        logger.warning(
            "Prediction count (%d) != expected count (%d)",
            len(predictions), len(expected),
        )

    n = min(len(predictions), len(expected))
    field_correct: dict[str, int] = {f: 0 for f in fields}
    total_rows_correct = 0

    mismatches: list[dict[str, Any]] = []

    for i in range(n):
        pred = predictions[i]
        exp = expected[i]
        row_correct = True
        row_mismatches: list[str] = []

        for field in fields:
            pred_val = pred.get(field, "").strip().lower()
            exp_val = exp.get(field, "").strip().lower()
            if pred_val == exp_val:
                field_correct[field] += 1
            else:
                row_correct = False
                row_mismatches.append(f"{field}: got '{pred_val}' expected '{exp_val}'")

        if row_correct:
            total_rows_correct += 1
        else:
            mismatches.append({
                "row_index": i,
                "user_id": pred.get("user_id", ""),
                "claim_object": pred.get("claim_object", ""),
                "differences": row_mismatches,
            })

    accuracies = {f: round(field_correct[f] / n, 4) if n > 0 else 0.0 for f in fields}
    overall_row_accuracy = round(total_rows_correct / n, 4) if n > 0 else 0.0
    macro_avg = round(sum(accuracies.values()) / len(accuracies), 4) if accuracies else 0.0

    return {
        "total_rows": n,
        "total_rows_correct": total_rows_correct,
        "overall_row_accuracy": overall_row_accuracy,
        "macro_avg_field_accuracy": macro_avg,
        "per_field_accuracy": accuracies,
        "mismatches": mismatches[:20],  # cap for readability
    }


def generate_report(
    config_name: str,
    accuracy: dict[str, Any],
    stats_summary: dict[str, Any],
    runtime: float,
    output_path: Path,
    comparison: dict[str, Any] | None = None,
) -> None:
    """Write evaluation_report.md."""
    lines = [
        "# Evaluation Report",
        "",
        f"**Configuration:** {config_name}",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Accuracy Metrics (sample_claims.csv)",
        "",
        f"- Total rows evaluated: {accuracy['total_rows']}",
        f"- Fully correct rows: {accuracy['total_rows_correct']}",
        f"- Overall row accuracy: {accuracy['overall_row_accuracy']:.2%}",
        f"- Macro-average field accuracy: {accuracy['macro_avg_field_accuracy']:.2%}",
        "",
        "### Per-Field Accuracy",
        "",
        "| Field | Accuracy |",
        "|---|---|",
    ]
    for field, acc in accuracy["per_field_accuracy"].items():
        lines.append(f"| {field} | {acc:.2%} |")

    lines.extend([
        "",
        "### Mismatches (first 20)",
        "",
    ])
    if accuracy["mismatches"]:
        for mm in accuracy["mismatches"]:
            lines.append(f"**Row {mm['row_index']}** (user={mm['user_id']}, {mm['claim_object']})")
            for diff in mm["differences"]:
                lines.append(f"- {diff}")
            lines.append("")
    else:
        lines.append("No mismatches — all rows match!")
        lines.append("")

    lines.extend([
        "## Operational Metrics",
        "",
        f"- Total API calls: {stats_summary['total_calls']}",
        f"- Successful calls: {stats_summary['successful_calls']}",
        f"- Failed calls: {stats_summary['failed_calls']}",
        f"- Fallback calls (Gemini -> Groq): {stats_summary['fallback_calls']}",
        f"- Total input tokens: {stats_summary['total_input_tokens']:,}",
        f"- Total output tokens: {stats_summary['total_output_tokens']:,}",
        f"- Total images processed: {stats_summary['total_images_processed']}",
        f"- Average latency per call: {stats_summary['avg_latency_ms']:.0f} ms",
        f"- Pipeline runtime: {runtime:.1f} s",
        "",
    ])

    # Provider breakdown
    lines.append("### Provider Breakdown")
    lines.append("")
    lines.append("| Provider | Calls | Success | Failure | Input Tokens | Output Tokens | Images |")
    lines.append("|---|---|---|---|---|---|---|")
    for provider, pstats in stats_summary.get("by_provider", {}).items():
        lines.append(
            f"| {provider} | {pstats['calls']} | {pstats['success']} | "
            f"{pstats['failure']} | {pstats['input_tokens']:,} | "
            f"{pstats['output_tokens']:,} | {pstats['images']} |"
        )
    lines.append("")

    # Cost estimate for full test set
    # The sample has ~20 rows; the full test has ~56 rows
    sample_rows = accuracy["total_rows"]
    full_rows = 56  # approximate, based on claims.csv
    scale_factor = full_rows / max(sample_rows, 1)
    est_full_cost = stats_summary.get("estimated_cost", 0.0) * scale_factor

    lines.extend([
        "## Cost Projection (Full Test Set)",
        "",
        f"- Sample set size: {sample_rows} rows",
        f"- Estimated full test set: {full_rows} rows",
        f"- Scale factor: {scale_factor:.1f}x",
        f"- Estimated cost to process full test set: **${est_full_cost:.4f}**",
        "",
        "Assumptions:",
        "- Gemini 2.5 Flash Lite: $0.15/1M input tokens, $0.60/1M output tokens",
        "- Groq Llama-4 Scout: $0.50/1M input tokens, $0.75/1M output tokens",
        "- Image costs estimated at $0.00025 per image",
        "",
    ])

    # Batching and caching strategy
    lines.extend([
        "## Implementation Strategy",
        "",
        "### Caching",
        "- File-based cache in `.vlm_cache/` keyed by SHA-256 of image content + prompt.",
        "- Re-running the pipeline never re-calls the API for an already-seen image+prompt pair.",
        "",
        "### Retry & Fallback",
        f"- Primary: Gemini 2.5 Flash with {_MAX_RETRIES_PRIMARY} retries and exponential backoff.",
        f"- Fallback: Groq Llama-4 Scout with {_MAX_RETRIES_FALLBACK} retries.",
        "- Backoff starts at 1s and doubles each retry.",
        "",
        "### Rate Limiting",
        "- No explicit batching (dataset is small enough for sequential processing).",
        "- Sequential per-image calls to avoid hitting TPM/RPM limits.",
        "- Cache minimizes redundant calls across evaluation iterations.",
        "",
    ])

    # Comparison section
    if comparison:
        lines.extend([
            "## Configuration Comparison",
            "",
            f"| Metric | {config_name} | {comparison.get('config_name', 'baseline')} |",
            "|---|---|---|",
        ])
        comp_acc = comparison.get("accuracy", {})
        lines.append(
            f"| Overall row accuracy | {accuracy['overall_row_accuracy']:.2%} | "
            f"{comp_acc.get('overall_row_accuracy', 0):.2%} |"
        )
        lines.append(
            f"| Macro field accuracy | {accuracy['macro_avg_field_accuracy']:.2%} | "
            f"{comp_acc.get('macro_avg_field_accuracy', 0):.2%} |"
        )
        lines.append("")

    lines.append("---")
    lines.append("*Report generated automatically by evaluation/main.py*")

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info("Evaluation report written to %s", output_path)


# Constants for retry counts (duplicated here for report)
_MAX_RETRIES_PRIMARY = 3
_MAX_RETRIES_FALLBACK = 2


def main() -> None:
    """Entry point for evaluation."""
    # Load environment variables from .env file if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    code_dir = Path(__file__).parent.parent.resolve()
    project_root = code_dir.parent

    config = {
        "dataset_root": str(project_root),
        "claims_csv": "dataset/sample_claims.csv",
        "user_history_csv": "dataset/user_history.csv",
        "evidence_requirements_csv": "dataset/evidence_requirements.csv",
    }

    logger.info("=== Running evaluation pipeline on sample_claims.csv ===")
    start_time = time.time()
    predictions = run_pipeline_for_evaluation(config)
    runtime = time.time() - start_time

    # Load expected
    sample_csv = project_root / "dataset" / "sample_claims.csv"
    expected = load_expected_outputs(sample_csv)

    # Compute accuracy
    accuracy = compute_accuracy(predictions, expected)
    logger.info("Overall row accuracy: %.2f%%", accuracy["overall_row_accuracy"] * 100)
    logger.info("Macro field accuracy: %.2f%%", accuracy["macro_avg_field_accuracy"] * 100)
    for field, acc in accuracy["per_field_accuracy"].items():
        logger.info("  %s: %.2f%%", field, acc * 100)

    # Get stats
    stats = get_global_stats()
    summary = stats.summary()
    summary["estimated_cost"] = stats.estimated_cost_usd()

    # Write predictions CSV for reference
    pred_path = code_dir / "evaluation" / "sample_predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)
    logger.info("Sample predictions written to %s", pred_path)

    # Generate report
    report_path = code_dir / "evaluation" / "evaluation_report.md"
    generate_report(
        config_name="default (Gemini primary + Groq fallback)",
        accuracy=accuracy,
        stats_summary=summary,
        runtime=runtime,
        output_path=report_path,
        comparison=None,
    )

    # Also write a JSON version of the accuracy for programmatic use
    json_path = code_dir / "evaluation" / "accuracy.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(accuracy, fh, indent=2)
    logger.info("Accuracy JSON written to %s", json_path)


if __name__ == "__main__":
    main()
