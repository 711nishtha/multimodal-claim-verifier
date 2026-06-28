# Evaluation Report

**Configuration:** default (Gemini primary + Groq fallback)
**Date:** 2026-06-20 10:28:36

## Accuracy Metrics (sample_claims.csv)

- Total rows evaluated: 20
- Fully correct rows: 3
- Overall row accuracy: 15.00%
- Macro-average field accuracy: 64.38%

### Per-Field Accuracy

| Field | Accuracy |
|---|---|
| evidence_standard_met | 90.00% |
| risk_flags | 35.00% |
| issue_type | 45.00% |
| object_part | 80.00% |
| claim_status | 60.00% |
| supporting_image_ids | 80.00% |
| valid_image | 80.00% |
| severity | 45.00% |

### Mismatches (first 20)

**Row 1** (user=user_002, car)
- risk_flags: got 'claim_mismatch;manual_review_required' expected 'none'
- issue_type: got 'dent' expected 'scratch'
- claim_status: got 'contradicted' expected 'supported'
- severity: got 'high' expected 'low'

**Row 2** (user=user_004, car)
- severity: got 'low' expected 'medium'

**Row 3** (user=user_007, car)
- risk_flags: got 'claim_mismatch;manual_review_required' expected 'none'
- issue_type: got 'glass_shatter' expected 'broken_part'
- claim_status: got 'contradicted' expected 'supported'
- severity: got 'high' expected 'medium'

**Row 4** (user=user_005, car)
- risk_flags: got 'user_history_risk' expected 'claim_mismatch;user_history_risk;manual_review_required'
- issue_type: got 'dent' expected 'scratch'
- claim_status: got 'supported' expected 'contradicted'
- severity: got 'medium' expected 'low'

**Row 5** (user=user_006, car)
- risk_flags: got 'non_original_image' expected 'wrong_angle;damage_not_visible'
- valid_image: got 'false' expected 'true'

**Row 7** (user=user_008, car)
- risk_flags: got 'damage_not_visible;possible_manipulation;non_original_image;user_history_risk' expected 'claim_mismatch;non_original_image;user_history_risk;manual_review_required'
- issue_type: got 'unknown' expected 'broken_part'
- object_part: got 'hood' expected 'front_bumper'
- claim_status: got 'not_enough_information' expected 'contradicted'
- supporting_image_ids: got 'none' expected 'img_1'
- severity: got 'unknown' expected 'high'

**Row 8** (user=user_009, laptop)
- risk_flags: got 'claim_mismatch;manual_review_required' expected 'none'
- issue_type: got 'glass_shatter' expected 'crack'
- claim_status: got 'contradicted' expected 'supported'
- severity: got 'high' expected 'medium'

**Row 10** (user=user_011, laptop)
- issue_type: got 'water_damage' expected 'stain'

**Row 11** (user=user_012, laptop)
- risk_flags: got 'low_light_or_glare;wrong_object_part' expected 'none'
- object_part: got 'lid' expected 'corner'
- claim_status: got 'contradicted' expected 'supported'

**Row 12** (user=user_018, laptop)
- issue_type: got 'glass_shatter' expected 'crack'

**Row 13** (user=user_020, laptop)
- risk_flags: got 'claim_mismatch;user_history_risk;manual_review_required' expected 'damage_not_visible;user_history_risk;manual_review_required'
- issue_type: got 'scratch' expected 'none'
- severity: got 'low' expected 'none'

**Row 14** (user=user_015, package)
- severity: got 'low' expected 'medium'

**Row 15** (user=user_030, package)
- risk_flags: got 'non_original_image' expected 'none'
- valid_image: got 'false' expected 'true'

**Row 16** (user=user_031, package)
- risk_flags: got 'user_history_risk' expected 'user_history_risk;manual_review_required'

**Row 17** (user=user_032, package)
- evidence_standard_met: got 'true' expected 'false'
- risk_flags: got 'manual_review_required' expected 'cropped_or_obstructed;damage_not_visible;manual_review_required'
- issue_type: got 'missing_part' expected 'unknown'
- claim_status: got 'supported' expected 'not_enough_information'
- supporting_image_ids: got 'img_1' expected 'none'
- valid_image: got 'true' expected 'false'
- severity: got 'medium' expected 'unknown'

**Row 18** (user=user_033, package)
- evidence_standard_met: got 'false' expected 'true'
- risk_flags: got 'user_history_risk' expected 'wrong_object;claim_mismatch;user_history_risk;manual_review_required'
- issue_type: got 'dent' expected 'unknown'
- object_part: got 'box' expected 'unknown'
- claim_status: got 'not_enough_information' expected 'contradicted'
- supporting_image_ids: got 'none' expected 'img_1'
- severity: got 'unknown' expected 'low'

**Row 19** (user=user_034, package)
- risk_flags: got 'wrong_object_part;possible_manipulation;text_instruction_present;user_history_risk' expected 'damage_not_visible;text_instruction_present;user_history_risk;manual_review_required'
- issue_type: got 'torn_packaging' expected 'none'
- object_part: got 'package_corner' expected 'seal'
- supporting_image_ids: got 'img_1' expected 'img_1;img_2'
- valid_image: got 'false' expected 'true'
- severity: got 'medium' expected 'none'

## Operational Metrics

- Total API calls: 0
- Successful calls: 0
- Failed calls: 0
- Fallback calls (Gemini -> Groq): 0
- Total input tokens: 0
- Total output tokens: 0
- Total images processed: 0
- Average latency per call: 0 ms
- Pipeline runtime: 0.3 s

### Provider Breakdown

| Provider | Calls | Success | Failure | Input Tokens | Output Tokens | Images |
|---|---|---|---|---|---|---|

## Cost Projection (Full Test Set)

- Sample set size: 20 rows
- Estimated full test set: 56 rows
- Scale factor: 2.8x
- Estimated cost to process full test set: **$0.0000**

Assumptions:
- Gemini 2.5 Flash Lite: $0.15/1M input tokens, $0.60/1M output tokens
- Groq Llama-4 Scout: $0.50/1M input tokens, $0.75/1M output tokens
- Image costs estimated at $0.00025 per image

## Implementation Strategy

### Caching
- File-based cache in `.vlm_cache/` keyed by SHA-256 of image content + prompt.
- Re-running the pipeline never re-calls the API for an already-seen image+prompt pair.

### Retry & Fallback
- Primary: Gemini 2.5 Flash with 3 retries and exponential backoff.
- Fallback: Groq Llama-4 Scout with 2 retries.
- Backoff starts at 1s and doubles each retry.

### Rate Limiting
- No explicit batching (dataset is small enough for sequential processing).
- Sequential per-image calls to avoid hitting TPM/RPM limits.
- Cache minimizes redundant calls across evaluation iterations.

---
*Report generated automatically by evaluation/main.py*