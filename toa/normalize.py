
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re

MODALITY_MAP = {
    "ct": "ct", "computed tomography": "ct",
    "mri": "mri", "magnetic resonance": "mri",
    "xray": "xray", "chest x-ray": "xray", "x-ray": "xray",
    "pet": "pet", "pet-ct": "pet-ct", "pet/ct": "pet-ct",
    "ultrasound": "ultrasound", "us": "ultrasound",
}

SITE_MAP = {
    "chest":"chest","thorax":"chest",
    "head":"head","brain":"head",
    "neck":"neck",
    "abdomen":"abdomen",
    "pelvis":"pelvis",
    "spine":"spine",
    "bone":"bone",
    "whole body":"whole_body","whole-body":"whole_body","skull base to mid thigh":"whole_body",
}

def normalize_imaging(modality: str|None, site: str|None, description: str) -> Tuple[str|None, str|None, str]:
    m = (modality or "").lower().strip()
    s = (site or "").lower().strip()
    # heuristics from description
    if not m:
        if "pet" in description.lower() and "ct" in description.lower():
            m = "pet-ct"
        elif "pet" in description.lower():
            m = "pet"
        elif "ct" in description.lower():
            m = "ct"
        elif "mri" in description.lower():
            m = "mri"
        elif "x-ray" in description.lower() or "xray" in description.lower():
            m = "xray"
        elif "ultrasound" in description.lower():
            m = "ultrasound"
    m = MODALITY_MAP.get(m, m if m in MODALITY_MAP.values() else None)

    if not s:
        desc = description.lower()
        for k in ["chest","thorax","head","brain","neck","abdomen","pelvis","spine","bone"]:
            if k in desc: s = k; break
        if not s and ("skull base" in desc and "thigh" in desc or "whole body" in desc):
            s = "whole body"
    s = SITE_MAP.get(s, s if s in SITE_MAP.values() else None)

    # strip noise
    clean_desc = re.sub(r"\b(limited evaluation|limited exam|without contrast|with contrast)\b", "", description, flags=re.I).strip()
    return m, s, clean_desc

def aggregate_labs(day_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse multiple lab entries into a single decision-informing entry per day.
    Keeps only abnormal analytes (↑ or ↓ markers in 'values').
    """
    labs = [e for e in day_events if e.get("type") == "lab"]
    if not labs: return []
    abnormal = {}
    for e in labs:
        vals = e.get("values") or {}
        for k,v in vals.items():
            if any(mark in str(v) for mark in ["↑","↓","high","low","elevated","decreased"]):
                abnormal[k] = v
    if not abnormal: return []
    # use first lab as base
    base = {k:v for k,v in labs[0].items() if k!="values"}
    base["description"] = ", ".join(f"{k} {abnormal[k]}" for k in sorted(abnormal))
    base["values"] = abnormal
    base["priority"] = "MINOR"  # labs usually minor unless specified otherwise
    return [base]

def background_portrait(day_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Produces a single critical_information event if portrait elements are present.
    """
    portrait_keys = ["smoking","occupation","comorbidities","family_history","allergies","medications"]
    snippets = []
    for e in day_events:
        if e.get("type") == "critical_information":
            d = e.get("description","").strip()
            if d: snippets.append(d)
    if not snippets: return []
    return [{
        **{k:v for k,v in day_events[0].items() if k not in ("type","description","subtype","values")},
        "type":"critical_information",
        "description":"; ".join(snippets),
        "priority":"MINOR",
    }]
