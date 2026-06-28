"""Tests for data_loading.py (Module 1).

Uses only stdlib + pytest as required.  No pandas, no pydantic.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loading import (
    ClaimContext,
    _build_user_history_map,
    _filter_applicable_requirements,
    _resolve_image_paths,
    compute_cache_seed,
    load_all,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    """Write a CSV file."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Unit tests for private helpers
# ---------------------------------------------------------------------------


def test_resolve_image_paths_existing(tmp_path: Path) -> None:
    img = tmp_path / "img_1.jpg"
    img.write_text("fake")
    resolved = _resolve_image_paths("img_1.jpg", tmp_path)
    assert len(resolved) == 1
    assert resolved[0].name == "img_1.jpg"


def test_resolve_image_paths_missing(tmp_path: Path) -> None:
    resolved = _resolve_image_paths("missing.jpg", tmp_path)
    assert len(resolved) == 0


def test_resolve_image_paths_multiple(tmp_path: Path) -> None:
    img1 = tmp_path / "img_1.jpg"
    img2 = tmp_path / "img_2.jpg"
    img1.write_text("fake1")
    img2.write_text("fake2")
    resolved = _resolve_image_paths("img_1.jpg;img_2.jpg", tmp_path)
    assert len(resolved) == 2


def test_resolve_image_paths_with_extra_search_dirs(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images" / "test" / "case_001"
    images.mkdir(parents=True)
    img = images / "img_1.jpg"
    img.write_text("fake")

    resolved = _resolve_image_paths(
        "test/case_001/img_1.jpg",
        dataset,
        extra_search_dirs=[dataset / "images"],
    )

    assert resolved == [img.resolve()]


def test_resolve_image_paths_ignores_directories(tmp_path: Path) -> None:
    img_dir = tmp_path / "img_1.jpg"
    img_dir.mkdir()

    resolved = _resolve_image_paths("img_1.jpg", tmp_path)

    assert resolved == []


def test_build_user_history_map() -> None:
    rows = [
        {"user_id": "user_001", "past_claim_count": "2"},
        {"user_id": "user_002", "past_claim_count": "5"},
    ]
    hmap = _build_user_history_map(rows)
    assert len(hmap) == 2
    assert hmap["user_001"]["past_claim_count"] == "2"


def test_filter_applicable_requirements() -> None:
    reqs = [
        {"requirement_id": "REQ_GENERAL", "claim_object": "all", "applies_to": "general"},
        {"requirement_id": "REQ_CAR", "claim_object": "car", "applies_to": "car"},
        {"requirement_id": "REQ_LAPTOP", "claim_object": "laptop", "applies_to": "laptop"},
    ]
    car_reqs = _filter_applicable_requirements(reqs, "car")
    assert len(car_reqs) == 2
    ids = {r["requirement_id"] for r in car_reqs}
    assert "REQ_GENERAL" in ids
    assert "REQ_CAR" in ids

    laptop_reqs = _filter_applicable_requirements(reqs, "laptop")
    assert len(laptop_reqs) == 2


def test_compute_cache_seed(tmp_path: Path) -> None:
    img = tmp_path / "test.jpg"
    img.write_bytes(b"fake_image_data")
    key1 = compute_cache_seed(img, "prompt1")
    key2 = compute_cache_seed(img, "prompt2")
    key3 = compute_cache_seed(img, "prompt1")
    assert key1 != key2
    assert key1 == key3


# ---------------------------------------------------------------------------
# Integration test for load_all
# ---------------------------------------------------------------------------


def test_load_all_full(tmp_path: Path) -> None:
    """End-to-end test of load_all with all three auxiliary CSVs."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    images = dataset / "images" / "test" / "case_001"
    images.mkdir(parents=True)
    img = images / "img_1.jpg"
    img.write_text("fake")

    _make_csv(
        dataset / "claims.csv",
        ["user_id", "image_paths", "user_claim", "claim_object"],
        [
            {
                "user_id": "user_001",
                "image_paths": "images/test/case_001/img_1.jpg",
                "user_claim": "My car has a dent.",
                "claim_object": "car",
            }
        ],
    )
    _make_csv(
        dataset / "user_history.csv",
        ["user_id", "past_claim_count", "accept_claim", "manualReview_claim",
         "rejected_claim", "last_90_days_claim_count", "history_flags", "history_summary"],
        [
            {
                "user_id": "user_001",
                "past_claim_count": "2",
                "accept_claim": "2",
                "manualReview_claim": "0",
                "rejected_claim": "0",
                "last_90_days_claim_count": "1",
                "history_flags": "none",
                "history_summary": "Low risk",
            }
        ],
    )
    _make_csv(
        dataset / "evidence_requirements.csv",
        ["requirement_id", "claim_object", "applies_to", "minimum_image_evidence"],
        [
            {
                "requirement_id": "REQ_GENERAL",
                "claim_object": "all",
                "applies_to": "general",
                "minimum_image_evidence": "Object should be visible.",
            }
        ],
    )

    config = {
        "dataset_root": str(tmp_path),
        "claims_csv": "dataset/claims.csv",
        "user_history_csv": "dataset/user_history.csv",
        "evidence_requirements_csv": "dataset/evidence_requirements.csv",
    }

    contexts = load_all(config)
    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx.row_index == 0
    assert ctx.user_id == "user_001"
    assert ctx.claim_object == "car"
    assert len(ctx.resolved_image_paths) == 1
    assert ctx.user_history is not None
    assert ctx.user_history["past_claim_count"] == "2"
    assert len(ctx.applicable_requirements) == 1
    assert ctx.applicable_requirements[0]["requirement_id"] == "REQ_GENERAL"


def test_load_all_missing_image(tmp_path: Path) -> None:
    """load_all should log a warning and record load_errors for missing images."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    _make_csv(
        dataset / "claims.csv",
        ["user_id", "image_paths", "user_claim", "claim_object"],
        [
            {
                "user_id": "user_001",
                "image_paths": "dataset/images/test/case_999/img_1.jpg",
                "user_claim": "Claim.",
                "claim_object": "car",
            }
        ],
    )
    _make_csv(
        dataset / "user_history.csv",
        ["user_id", "past_claim_count", "accept_claim", "manualReview_claim",
         "rejected_claim", "last_90_days_claim_count", "history_flags", "history_summary"],
        [],
    )
    _make_csv(
        dataset / "evidence_requirements.csv",
        ["requirement_id", "claim_object", "applies_to", "minimum_image_evidence"],
        [],
    )

    config = {
        "dataset_root": str(tmp_path),
        "claims_csv": "dataset/claims.csv",
        "user_history_csv": "dataset/user_history.csv",
        "evidence_requirements_csv": "dataset/evidence_requirements.csv",
    }

    contexts = load_all(config)
    assert len(contexts) == 1
    assert len(contexts[0].resolved_image_paths) == 0
    assert any("not found" in err for err in contexts[0].load_errors)


def test_load_all_invalid_claim_object(tmp_path: Path) -> None:
    """load_all should record load_errors for invalid claim_object values."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    _make_csv(
        dataset / "claims.csv",
        ["user_id", "image_paths", "user_claim", "claim_object"],
        [
            {
                "user_id": "user_001",
                "image_paths": "",
                "user_claim": "Claim.",
                "claim_object": "invalid_object",
            }
        ],
    )
    _make_csv(
        dataset / "user_history.csv",
        ["user_id", "past_claim_count", "accept_claim", "manualReview_claim",
         "rejected_claim", "last_90_days_claim_count", "history_flags", "history_summary"],
        [],
    )
    _make_csv(
        dataset / "evidence_requirements.csv",
        ["requirement_id", "claim_object", "applies_to", "minimum_image_evidence"],
        [],
    )

    config = {
        "dataset_root": str(tmp_path),
        "claims_csv": "dataset/claims.csv",
        "user_history_csv": "dataset/user_history.csv",
        "evidence_requirements_csv": "dataset/evidence_requirements.csv",
    }

    contexts = load_all(config)
    assert any("invalid claim_object" in err for err in contexts[0].load_errors)


def test_claim_context_image_ids(tmp_path: Path) -> None:
    ctx = ClaimContext(
        row_index=0,
        user_id="u1",
        image_paths="a/img_1.jpg;a/img_2.jpg",
        user_claim="claim",
        claim_object="car",
        resolved_image_paths=[tmp_path / "img_1.jpg", tmp_path / "img_2.jpg"],
    )
    assert ctx.image_ids == ["img_1", "img_2"]
