
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import hashlib
from .normalize import normalize_imaging, aggregate_labs, background_portrait

PRECEDENCE = ["surgery","treatment","adverse_effect","infection","diagnostic","imaging","procedure","examination","symptom","lab","critical_information"]

def event_key(e: Dict[str, Any]) -> Tuple:
    vt = e.get("valid_time_start") or ""
    typ = e.get("type") or ""
    desc = (e.get("description") or "").lower().strip()
    modality, site, clean_desc = normalize_imaging(e.get("modality"), e.get("site"), desc) if typ=="imaging" else (e.get("modality"), e.get("site"), desc)
    values_sig = ""
    if e.get("type")=="lab" and isinstance(e.get("values"), dict):
        values_sig = "|".join(sorted(f"{k}:{v}" for k,v in e["values"].items()))
    return (vt, typ, modality or "", site or "", clean_desc, values_sig)

def stable_event_id(pid: str, e: Dict[str, Any]) -> str:
    k = "|".join(map(str, event_key(e) + (pid,)))
    return hashlib.blake2b(k.encode("utf-8"), digest_size=12).hexdigest()

def rollup_same_day(pid: str, day_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    - One event per (date,type) after specialized rollups
    - Specialized rollups: imaging merge by (modality,site), labs collapse to abnormal-only, background portrait
    - Apply precedence to choose representative when duplicates remain
    """
    # Specialized aggregations
    imaging_groups = {}
    other = []
    for e in day_events:
        if e.get("type") == "imaging":
            m,s,_ = normalize_imaging(e.get("modality"), e.get("site"), e.get("description",""))
            key = (m or "", s or "")
            imaging_groups.setdefault(key, []).append(e)
        else:
            other.append(e)
    merged_imaging = []
    for (m,s), group in imaging_groups.items():
        base = min(group, key=lambda x: (PRECEDENCE.index("imaging"), len((x.get("description") or "")) ))
        findings = [g.get("description","") for g in group if g.get("description")]
        base = dict(base)
        base["modality"], base["site"] = m, s
        if findings:
            base["description"] = ", ".join(sorted(set(findings)))
        merged_imaging.append(base)

    # Labs
    labs = aggregate_labs(day_events)

    # Background portrait
    portrait = background_portrait(day_events)

    # Combine and collapse by (type) - EXCEPT imaging which keeps per (modality, site)
    pool = [e for e in other if e.get("type")!="lab" and e.get("type")!="critical_information"] + merged_imaging + labs + portrait
    # Choose one per type using precedence order (earlier in list = higher precedence)
    # For imaging: preserve multiple studies per day by using (type, modality, site) as key
    by_key = {}
    for e in pool:
        t = e.get("type")
        # Use composite key for imaging to preserve different modality/site combinations
        if t == "imaging":
            key = (t, e.get("modality") or "", e.get("site") or "")
        else:
            key = (t,)

        if key not in by_key:
            by_key[key] = e
        else:
            # keep the one with longer informative description, tie-break by priority (MAJOR wins)
            cur = by_key[key]
            score = (1 if (e.get("priority")=="MAJOR") else 0, len(e.get("description","")))
            cur_score = (1 if (cur.get("priority")=="MAJOR") else 0, len(cur.get("description","")))
            if score > cur_score:
                by_key[key] = e

    # Output in precedence order, preserving all imaging studies
    out = []
    for t in PRECEDENCE:
        # Collect all events of this type (handles multiple imaging studies)
        type_events = [e for key, e in by_key.items() if key[0] == t]
        for e in type_events:
            e_copy = dict(e)
            e_copy["event_id"] = stable_event_id(pid, e_copy)
            out.append(e_copy)
    return out

def sort_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(e):
        v = e.get("valid_time_start") or ""
        r0 = sorted(e.get("recorded_times") or ["9999-12-31"])[0]
        p = PRECEDENCE.index(e.get("type")) if e.get("type") in PRECEDENCE else len(PRECEDENCE)
        return (v, r0, p)
    return sorted(events, key=key)
