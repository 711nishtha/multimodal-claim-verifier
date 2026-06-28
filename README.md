# multimodal-claim-verifier

A modular pipeline for automated insurance/damage claim verification using vision-language models (VLMs). Given a customer conversation and submitted images, the system determines whether visual evidence supports, contradicts, or is insufficient for claims on cars, laptops, and packages.

---

## Overview

The pipeline ingests CSV-based claim data alongside product images, extracts structured assertions from customer chat transcripts, and reasons over per-image VLM observations using rule-based logic to produce a final verdict for each claim. A deterministic safety gate prevents prompt injection attacks embedded in submitted images.

Primary model: `gemini-3.1-flash-lite`. Fallback: Groq Llama-4 Scout. All VLM responses are locally cached by default, and API calls are rate-limited to avoid quota exhaustion.

---

## Architecture

The pipeline follows a strict sequential, module-by-module design with no cross-stage coupling:

| Step | Module | Description |
|------|--------|-------------|
| 1 | `data_loading.py` | Loads CSVs, resolves image paths, builds frozen `ClaimContext` objects |
| 2 | `claim_parsing.py` | Extracts damage assertion (issue type, part, severity) from chat transcripts via structured LLM prompt |
| 3 | `planning.py` | Maps parsed claims to applicable evidence requirements (rule-based, no LLM) |
| 4 | `vision.py` | Calls VLM per image to extract structured observations: issue visibility, part visibility, quality flags, detected text, authenticity signals |
| 5 | `safety_gate.py` | Deterministic, non-overridable code-level constraint — detects instruction-like text in images and redacts it before verification |
| 6 | `verification.py` | Rule-based reasoning over gated observations, parsed claim, and user history to produce all output fields; text-only LLM call for natural-language justification grounded in specific image IDs |
| 7 | `output_writer.py` | Writes the exact 14-column `output.csv` schema |
| 8 | `llm_clients.py` | `gemini-3.1-flash-lite` primary, Groq Llama-4 Scout fallback; file-based SHA-256 cache; sliding-window rate limiter (8 calls/60s, 50-60s cool-off); exponential backoff retry |
| 9 | `usage_tracker.py` | Tracks every API call, token counts, images processed, latency, and estimated cost |

---

## Repository Structure

```
.
├── code/
│   ├── main.py                  # Pipeline orchestrator
│   ├── data_loading.py
│   ├── claim_parsing.py
│   ├── planning.py
│   ├── vision.py
│   ├── safety_gate.py
│   ├── verification.py
│   ├── output_writer.py
│   ├── llm_clients.py
│   ├── usage_tracker.py
│   ├── trace_obs.py
│   ├── dump_cache.py
│   ├── evaluation/
│   │   ├── main.py              # Evaluation harness
│   │   └── __init__.py
│   └── tests/
│       ├── test_data_loading.py
│       └── test_safety_gate.py
├── dataset/
│   ├── claims.csv               # Full claim dataset
│   ├── sample_claims.csv        # Labeled subset for evaluation
│   ├── evidence_requirements.csv
│   ├── user_history.csv
│   └── images/
├── requirements.txt
└── README.md
```

Not included in this repository: `venv/`, `.env`, `deliverables/`, `output.csv`, and any generated evaluation artifacts or VLM cache files.

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Primary VLM API key (`gemini-3.1-flash-lite`) |
| `GROQ_API_KEY` | Fallback VLM API key (Groq Llama-4 Scout) |

Both are read from the environment only and must never be hardcoded. A `.env` file at the project root is supported via `python-dotenv` but is not committed to the repository.

---

## Setup

### Requirements

Python 3.9 or later. Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:
- `requests` — HTTP client for LLM API calls
- `pytest` — test runner
- `python-dotenv` — loads `.env` file into environment

No other third-party packages are required. The rest of the pipeline uses the Python standard library.

### API Keys

Create a `.env` file in the project root (this file is gitignored):

```
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

---

## How to Run

### Full pipeline

```bash
cd code
python main.py
```

Reads `dataset/claims.csv`, processes every row, and writes `output.csv` to the project root.

### Evaluation

```bash
cd code
python -m evaluation.main
```

Runs the pipeline against `dataset/sample_claims.csv`, compares predictions to expected labels, and writes:
- `code/evaluation/evaluation_report.md`
- `code/evaluation/sample_predictions.csv`
- `code/evaluation/accuracy.json`
- `code/.vlm_cache/usage_stats.json`

### Tests

```bash
cd code
python -m pytest tests/ -v
```

Tests cover `data_loading.py` and `safety_gate.py`.

---

## Output Schema

`output.csv` contains these 14 columns in order:

| # | Column | Description |
|---|--------|-------------|
| 1 | `user_id` | Claimant identifier |
| 2 | `image_paths` | Submitted image paths |
| 3 | `user_claim` | Raw claim text |
| 4 | `claim_object` | car, laptop, or package |
| 5 | `evidence_standard_met` | true/false |
| 6 | `evidence_standard_met_reason` | Human-readable explanation |
| 7 | `risk_flags` | Semicolon-separated risk flags |
| 8 | `issue_type` | dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown |
| 9 | `object_part` | Object-specific part enum |
| 10 | `claim_status` | supported, contradicted, not_enough_information |
| 11 | `claim_status_justification` | Natural-language reasoning grounded in image IDs |
| 12 | `supporting_image_ids` | Semicolon-separated (or "none") |
| 13 | `valid_image` | true/false — authenticity assessment |
| 14 | `severity` | none, low, medium, high, unknown |

---

## VLM Cache

VLM responses are cached in `code/.vlm_cache/` and keyed by SHA-256 of (image file content + prompt text). Re-running the pipeline never re-calls the API for an already-seen (image, prompt) pair. The cache directory is gitignored.

---

## Key Design Decisions

- **No hardcoded labels**: All reasoning generalizes across claim types and objects. No row content, filename, or case ID is matched against sample data.
- **History is additive only**: `user_history.csv` may contribute risk flags and appear in justifications, but never flips a `claim_status` that clear visual evidence has already determined.
- **Evidence standard and image validity are independent**: An image can be clear enough to assess (`evidence_standard_met=true`) but inauthentic (`valid_image=false`), for example a watermarked stock photo.
- **Text instructions in images are ignored and redacted**: The safety gate forces `text_instruction_present` and explicitly strips such text before it reaches the verification step. This is enforced at the code level, not by model instruction alone.
- **Wrong object equals contradiction**: Visible damage on the wrong object type does not support the claim, even if real damage is present.
- **Severity exaggeration triggers contradiction**: If the user describes severe damage but submitted images show only minor or no damage, the claim is contradicted.
- **One image at a time**: Vision analysis is performed per image and never pooled across images in a single prompt, ensuring each image observation is independent.

---

## Known Issues

- **VLM hallucination on ambiguous images**: Gemini 2.5 Flash occasionally misidentifies object parts or issue types when images are low resolution, heavily cropped, or shot at unusual angles. The severity cap in `verification.py` partially compensates for over-prediction of "high" severity, but false negatives on subtle damage are still possible.

- **Safety gate relies on text extraction quality**: The safety gate can only redact instruction text that the VLM accurately transcribes via the `text_detected` field. If the VLM fails to recognize text in an image (e.g., handwritten or stylized fonts), the instruction text may pass through undetected.

- **Fallback model inconsistency**: The Groq Llama-4 Scout fallback does not always produce output in the same structured format as Gemini 2.5 Flash. The parsing layer handles common deviations, but edge cases may produce malformed fields that default to "unknown".

- **Severity voting is noisy**: When multiple images are submitted and show different severity levels, the voting mechanism resolves ties by picking the lowest severity. This is conservative but may under-report damage in legitimate multi-image claims.

- **No streaming or async execution**: The pipeline processes claims sequentially and makes synchronous API calls. On large datasets (hundreds of claims with multiple images each), runtime can be significant. The file-based cache mitigates repeated runs but does not help on first-pass processing.

- **CSV-only input format**: The data loading module expects a specific CSV schema. There is no validation layer that raises human-readable errors for missing or malformed columns; incorrect input will typically result in a Python exception mid-run.

---

## Future Improvements

- **Async/concurrent image processing**: Refactor `vision.py` and `llm_clients.py` to use `asyncio` or a thread pool so images for a single claim (and multiple claims) can be processed concurrently, significantly reducing wall-clock time on larger datasets.

- **Structured output enforcement**: Switch Gemini API calls to use the `response_schema` parameter to guarantee JSON conformance at the model level, eliminating the need for fragile regex-based parsing fallbacks in `llm_clients.py`.

- **Expanded object support**: The current system is limited to cars, laptops, and packages. Extending `evidence_requirements.csv` and the part enums in `verification.py` to cover more product categories (electronics, furniture, appliances) would broaden applicability.

- **Confidence scores**: The current output is categorical. Adding a confidence or uncertainty score per decision field would allow downstream consumers to apply their own thresholds and flag borderline cases for human review more precisely.

- **Database or API backend**: Replace the CSV-based I/O with a database backend (e.g., PostgreSQL) and expose the pipeline as a REST API, enabling integration with existing claims management systems without manual file handling.

- **Human-in-the-loop interface**: Build a lightweight review UI that surfaces claims flagged with `manual_review_required`, displays the submitted images alongside the system's justification, and allows a human reviewer to confirm, override, or escalate the decision.

- **Fine-tuned vision model**: The current approach relies entirely on general-purpose VLMs via prompt engineering. Fine-tuning a smaller, specialized model on labeled damage images could improve accuracy on the specific damage categories (dent, scratch, crack, etc.) and reduce API cost.

- **Evaluation dataset expansion**: The labeled sample in `dataset/sample_claims.csv` is small. A larger, more diverse labeled set covering edge cases (multiple objects in frame, partial damage, staged images) would make the evaluation metrics more meaningful.

- **Safety gate pattern coverage**: The current instruction-detection patterns in `safety_gate.py` cover a limited set of phrases. A more robust approach would use a dedicated text classification model or embedding similarity to detect adversarial text regardless of phrasing.
