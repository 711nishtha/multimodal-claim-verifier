"""Module 1 — Data Loading.

Stateless functions that load CSV files, resolve image paths, validate rows,
and produce frozen ClaimContext objects.  No pandas, no pydantic, no heavy
frameworks — stdlib + csv module only.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaimContext:
    """Immutable context for a single damage claim.

    Fields:
        row_index: the 0-based index of the row in the source CSV.
        user_id: the submitting user.
        image_paths: raw semicolon-separated image path string.
        user_claim: chat transcript describing the issue.
        claim_object: one of car, laptop, package.
        resolved_image_paths: list of Path objects pointing to existing images.
        user_history: the user's history row (dict), or None.
        applicable_requirements: evidence-requirement rows for this claim's
            object+issue family.
        load_errors: non-fatal issues encountered while loading this row.
    """

    row_index: int
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    resolved_image_paths: list[Path] = field(default_factory=list)
    user_history: dict[str, str] | None = None
    applicable_requirements: list[dict[str, str]] = field(default_factory=list)
    load_errors: list[str] = field(default_factory=list)

    @property
    def image_ids(self) -> list[str]:
        """Return the image IDs (stem of each filename) from resolved_image_paths."""
        return [p.stem for p in self.resolved_image_paths]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _load_csv_dicts(path: Path) -> list[dict[str, str]]:
    """Read a CSV file and return a list of row dicts."""
    rows: list[dict[str, str]] = []
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Strip whitespace from keys and values
            cleaned = {k.strip(): (v.strip() if v else "") for k, v in row.items()}
            rows.append(cleaned)
    return rows


def _resolve_image_paths(
    image_paths_str: str,
    base_dir: Path,
    extra_search_dirs: list[Path] | None = None,
) -> list[Path]:
    """Convert a semicolon-separated string into a list of existing Path objects.

    Each component may already be absolute, relative to *base_dir* (the
    directory containing the claims CSV), relative to the project root, or
    relative to dataset/images. Missing files are silently dropped (caller logs
    them).
    """
    resolved: list[Path] = []
    if not image_paths_str:
        return resolved
    search_dirs = [base_dir]
    if extra_search_dirs:
        search_dirs.extend(extra_search_dirs)

    for part in image_paths_str.split(";"):
        part = part.strip()
        if not part:
            continue
        candidate = Path(part)
        candidates = [candidate] if candidate.is_absolute() else [
            search_dir / candidate for search_dir in search_dirs
        ]
        for candidate_path in candidates:
            if candidate_path.exists() and candidate_path.is_file():
                resolved.append(candidate_path.resolve())
                break
        # Caller decides what to do with missing files
    return resolved


def _build_user_history_map(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Index user_history rows by user_id."""
    history_map: dict[str, dict[str, str]] = {}
    for row in rows:
        uid = row.get("user_id", "").strip()
        if uid:
            history_map[uid] = row
    return history_map


def _filter_applicable_requirements(
    all_requirements: list[dict[str, str]],
    claim_object: str,
) -> list[dict[str, str]]:
    """Return only requirements that apply to *claim_object* or to 'all'."""
    applicable: list[dict[str, str]] = []
    for req in all_requirements:
        req_object = req.get("claim_object", "").strip().lower()
        if req_object in ("all", claim_object.lower()):
            applicable.append(req)
    return applicable


def _compute_row_hash(ctx: ClaimContext) -> str:
    """Compute a deterministic hash of the input fields for cache/tracking."""
    payload = f"{ctx.user_id}|{ctx.image_paths}|{ctx.user_claim}|{ctx.claim_object}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_all(config: dict[str, Any]) -> list[ClaimContext]:
    """Load and validate every row from claims.csv, building a ClaimContext for each.

    Args:
        config: dictionary with keys:
            - dataset_root (str or Path): root directory containing dataset/
            - claims_csv (str, optional): relative path to claims.csv
            - user_history_csv (str, optional): relative path to user_history.csv
            - evidence_requirements_csv (str, optional): path to evidence_requirements.csv

    Returns:
        A list of ClaimContext objects, one per row in claims.csv.
        Per-row load issues are stored in *load_errors* and logged via
        logging.warning().
    """
    dataset_root = Path(config.get("dataset_root", ".")).expanduser().resolve()

    claims_path = dataset_root / config.get("claims_csv", "dataset/claims.csv")
    history_path = dataset_root / config.get("user_history_csv", "dataset/user_history.csv")
    requirements_path = dataset_root / config.get(
        "evidence_requirements_csv", "dataset/evidence_requirements.csv"
        ,)

    # Load auxiliary data
    user_history_rows = _load_csv_dicts(history_path) if history_path.exists() else []
    history_map = _build_user_history_map(user_history_rows)

    requirements_rows = (
        _load_csv_dicts(requirements_path) if requirements_path.exists() else []
    )

    claim_rows = _load_csv_dicts(claims_path)

    contexts: list[ClaimContext] = []
    for idx, row in enumerate(claim_rows):
        user_id = row.get("user_id", "").strip()
        image_paths_str = row.get("image_paths", "").strip()
        user_claim = row.get("user_claim", "").strip()
        claim_object = row.get("claim_object", "").strip().lower()

        load_errors: list[str] = []

        # Validate required fields
        if not user_id:
            load_errors.append(f"Row {idx}: missing user_id")
        if not image_paths_str:
            load_errors.append(f"Row {idx}: missing image_paths")
        if not user_claim:
            load_errors.append(f"Row {idx}: missing user_claim")
        if not claim_object:
            load_errors.append(f"Row {idx}: missing claim_object")
        elif claim_object not in ("car", "laptop", "package"):
            load_errors.append(f"Row {idx}: invalid claim_object '{claim_object}'")

        # Resolve image paths relative to common dataset layouts.
        claims_dir = claims_path.parent
        project_root = dataset_root
        image_root = claims_dir / "images"
        resolved = _resolve_image_paths(
            image_paths_str,
            claims_dir,
            extra_search_dirs=[project_root, image_root],
        )
        raw_parts = [p.strip() for p in image_paths_str.split(";") if p.strip()]
        if len(resolved) != len(raw_parts):
            resolved_names = {str(p) for p in resolved}
            missing_parts = []
            for raw_part in raw_parts:
                raw_path = Path(raw_part)
                candidates = [raw_path] if raw_path.is_absolute() else [
                    claims_dir / raw_path,
                    project_root / raw_path,
                    image_root / raw_path,
                ]
                if not any(str(c.resolve()) in resolved_names for c in candidates if c.exists()):
                    missing_parts.append(raw_part)
            load_errors.append(
                f"Row {idx}: {len(missing_parts)} of {len(raw_parts)} image(s) not found: "
                + ";".join(missing_parts)
            )

        # Look up user history
        user_history = history_map.get(user_id)
        if user_history is None:
            load_errors.append(f"Row {idx}: no history found for user_id '{user_id}'")

        # Determine applicable requirements
        applicable = _filter_applicable_requirements(requirements_rows, claim_object)

        ctx = ClaimContext(
            row_index=idx,
            user_id=user_id,
            image_paths=image_paths_str,
            user_claim=user_claim,
            claim_object=claim_object,
            resolved_image_paths=resolved,
            user_history=user_history,
            applicable_requirements=applicable,
            load_errors=load_errors,
        )

        for err in load_errors:
            logging.warning("[data_loading] %s", err)

        contexts.append(ctx)

    logger.info("Loaded %d claim(s) from %s", len(contexts), claims_path)
    return contexts


# ---------------------------------------------------------------------------
# Expose hash helper for downstream cache keys
# ---------------------------------------------------------------------------


def compute_cache_seed(image_path: Path, prompt: str) -> str:
    """Return a stable cache key string for an image + prompt combination.

    Uses SHA-256 over (file content + prompt text) so that re-running the
    pipeline never re-calls the API for an already-seen image+prompt pair.
    """
    hasher = hashlib.sha256()
    if image_path.exists():
        with image_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                hasher.update(chunk)
    hasher.update(prompt.encode("utf-8"))
    return hasher.hexdigest()
