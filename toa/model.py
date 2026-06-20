
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class Event:
    patient_id: str
    event_id: str
    type: str                      # imaging|diagnostic|treatment|surgery|lab|symptom|examination|procedure|adverse_effect|critical_information
    subtype: Optional[str] = None  # e.g. ct_chest, pet-ct_whole_body
    description: str = ""
    priority: str = "MINOR"        # MAJOR|MINOR
    valid_time_start: str = ""     # YYYY-MM-DD
    valid_time_end: Optional[str] = None
    recorded_times: List[str] = field(default_factory=list)  # evidence dates
    modality: Optional[str] = None
    site: Optional[str] = None
    values: Optional[Dict[str, str]] = None                  # for labs: {Analyte: "value+direction"}
    sources: List[Dict[str, Any]] = field(default_factory=list)  # provenance snippets, doc ids, etc.

@dataclass
class Episode:
    episode_id: str
    kind: str               # diagnosis|systemic_tx_line|rt_course|surgery_course|progression|major_ae|hospitalization|surveillance
    start_date: str
    end_date: Optional[str]
    anchor_event_id: Optional[str]
    event_ids: List[str] = field(default_factory=list)
    response_status: Optional[str] = None
    line_number: Optional[int] = None
    intent: Optional[str] = None
