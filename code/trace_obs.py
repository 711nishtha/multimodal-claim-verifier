"""Trace the full pipeline for sample_claims rows to see VLM observations."""
import sys, json
sys.path.insert(0, '.')

from data_loading import load_all
from claim_parsing import parse_claim  
from vision import analyze_all_images
from safety_gate import apply_safety_gate

config = {
    "dataset_root": "..",
    "claims_csv": "dataset/sample_claims.csv",
    "user_history_csv": "dataset/user_history.csv",
    "evidence_requirements_csv": "dataset/evidence_requirements.csv",
}

claims = load_all(config)
for i, claim in enumerate(claims):
    parsed = parse_claim(claim)
    obs_list = analyze_all_images(claim, parsed)
    print(f"\n=== ROW {i} user={claim.user_id} obj={claim.claim_object} ===")
    print(f"  Parsed: issue={parsed.issue_type} part={parsed.object_part} sev={parsed.described_severity}")
    for obs in obs_list:
        print(f"  IMG {obs.image_id}: vis={obs.issue_visible} part_vis={obs.part_visible} "
              f"issue={obs.issue_type_guess} part={obs.object_part_guess} sev={obs.severity_guess} "
              f"qual={obs.quality_flags} auth={obs.authenticity_signals}")
