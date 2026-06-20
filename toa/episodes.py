
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import hashlib
from .dedupe_rollup import PRECEDENCE

def _eid(seed: str) -> str:
    return hashlib.blake2b(seed.encode("utf-8"), digest_size=12).hexdigest()

def _date(e: Dict[str, Any]) -> str:
    return e.get("valid_time_start") or ""

def derive_episodes(pid: str, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deterministic, rule-based episode construction.
    """
    # Sort by date then precedence
    events = sorted(events, key=lambda e: (_date(e), PRECEDENCE.index(e.get("type")) if e.get("type") in PRECEDENCE else 999))
    episodes: List[Dict[str, Any]] = []
    current_line: Optional[Dict[str, Any]] = None
    line_number = 0

    for e in events:
        t = e.get("type")
        d = _date(e)

        # Diagnosis
        if t in ("diagnostic",) and "malignan" in (e.get("description","").lower()):
            ep = {
                "episode_id": _eid(f"{pid}|diagnosis|{d}|{e.get('event_id','')}"),
                "kind": "diagnosis",
                "start_date": d,
                "end_date": d,
                "anchor_event_id": e.get("event_id"),
                "event_ids": [e.get("event_id")],
            }
            episodes.append(ep)

        # Systemic therapy lines
        if t == "treatment" and any(k in (e.get("description","").lower()) for k in ["chemotherapy","immunotherapy","targeted","bevacizumab","carboplatin","cisplatin","pembrolizumab","nivolumab","erlotinib","osimertinib"]):
            # Start or switch line if description suggests new regimen
            if (current_line is None) or (e.get("description") not in (current_line.get("regimen_desc") or "")):
                line_number += 1
                current_line = {
                    "episode_id": _eid(f"{pid}|systemic_tx_line|{line_number}|{d}|{e.get('event_id','')}"),
                    "kind": "systemic_tx_line",
                    "start_date": d,
                    "end_date": None,
                    "anchor_event_id": e.get("event_id"),
                    "event_ids": [e.get("event_id")],
                    "line_number": line_number,
                    "response_status": None,
                    "regimen_desc": e.get("description","")
                }
                episodes.append({k:v for k,v in current_line.items() if k!="regimen_desc"})
            else:
                # continue same line
                for ep in episodes[::-1]:
                    if ep["kind"]=="systemic_tx_line" and ep.get("end_date") is None:
                        ep["event_ids"].append(e.get("event_id"))
                        break

        # Radiation course
        if t in ("treatment","procedure") and "radiation" in (e.get("description","").lower()):
            ep = {
                "episode_id": _eid(f"{pid}|rt_course|{d}|{e.get('event_id','')}"),
                "kind": "rt_course",
                "start_date": d,
                "end_date": d,
                "anchor_event_id": e.get("event_id"),
                "event_ids": [e.get("event_id")]
            }
            episodes.append(ep)

        # Surgery course
        if t == "surgery":
            ep = {
                "episode_id": _eid(f"{pid}|surgery_course|{d}|{e.get('event_id','')}"),
                "kind": "surgery_course",
                "start_date": d,
                "end_date": d,
                "anchor_event_id": e.get("event_id"),
                "event_ids": [e.get("event_id")]
            }
            episodes.append(ep)

        # Response / Progression markers
        if t in ("imaging","diagnostic"):
            desc = (e.get("description","").lower())
            if any(k in desc for k in ["progression","progressed","pd","worsening","increase in size"]):
                # close current line if open
                for ep in episodes[::-1]:
                    if ep["kind"]=="systemic_tx_line" and ep.get("end_date") is None:
                        ep["end_date"] = d
                        ep["response_status"] = "Progression"
                        break
                ep = {
                    "episode_id": _eid(f"{pid}|progression|{d}|{e.get('event_id','')}"),
                    "kind": "progression",
                    "start_date": d, "end_date": d,
                    "anchor_event_id": e.get("event_id"),
                    "event_ids": [e.get("event_id")]
                }
                episodes.append(ep)
            elif any(k in desc for k in ["partial response","pr ","complete response","cr ","stable disease","sd "]):
                for ep in episodes[::-1]:
                    if ep["kind"]=="systemic_tx_line" and ep.get("end_date") is None:
                        ep["response_status"] = "Response/SD"
                        break

        # Major AE
        if t == "adverse_effect" and any(k in (e.get("description","").lower()) for k in ["grade 3","grade 4","hospital","held","dose reduction","stopped"]):
            ep = {
                "episode_id": _eid(f"{pid}|major_ae|{d}|{e.get('event_id','')}"),
                "kind": "major_ae",
                "start_date": d, "end_date": d,
                "anchor_event_id": e.get("event_id"),
                "event_ids": [e.get("event_id")]
            }
            episodes.append(ep)

    # Surveillance blocks between systemic lines (posthoc)
    # naive: any gap > 45 days without treatment events -> surveillance episode
    dates = sorted(set([_date(e) for e in events if _date(e)]))
    # We skip explicit implementation for brevity; can add later.
    return episodes
