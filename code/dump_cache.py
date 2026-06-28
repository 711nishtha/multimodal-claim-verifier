"""Dump all VLM cache entries for analysis."""
import json, glob, os

cache_dir = os.path.join(os.path.dirname(__file__), '.vlm_cache')
for f in sorted(glob.glob(os.path.join(cache_dir, '*.json'))):
    if 'usage' in f:
        continue
    with open(f) as fh:
        data = json.load(fh)
    parsed = data.get('parsed', {})
    if isinstance(parsed, dict):
        vis = parsed.get("issue_visible")
        part = parsed.get("part_visible")
        issue = parsed.get("issue_type_guess")
        obj_part = parsed.get("object_part_guess")
        sev = parsed.get("severity_guess")
        qual = parsed.get("quality_flags")
        auth = parsed.get("authenticity_signals")
        desc = (parsed.get("visual_description") or "")[:80]
        print(f"FILE={os.path.basename(f)[:16]}")
        print(f"  vis={vis} part={part} issue={issue} obj_part={obj_part} sev={sev}")
        print(f"  qual={qual} auth={auth}")
        print(f"  desc={desc}")
        print()
    else:
        print(f"FILE={os.path.basename(f)[:16]}  TEXT: {str(parsed)[:150]}")
        print()
