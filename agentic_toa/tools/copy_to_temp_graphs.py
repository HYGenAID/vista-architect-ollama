#!/usr/bin/env python3
"""Sandbox a patient's Lumia graph + extract Person demographics from the TB-truncated XML.

  graph_store/graphs/{pid}.graphml  ->  agentic_toa/temp_graphs/{pid}.graphml
  patient_records/cohorts/*_{pid}.xml  ->  agentic_toa/runs/{pid}/demographics.json

The graph itself has no Person node (the existing pipeline reads demographics
from the source XML's <person> element). To keep the agentic pipeline working
without re-reading the full XML at every step, we extract demographics ONCE
during sandbox setup and write them to runs/{pid}/demographics.json. The
display step reads from there.

Idempotent. The agent operates only on the sandboxed copy of the graph; the
original graph_store/ is never modified.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # vista_architect/
SRC = ROOT / "graph_store" / "graphs"
DST = ROOT / "agentic_toa" / "temp_graphs"
RUNS_ROOT = ROOT / "agentic_toa" / "runs"


def copy_one(pid: str, force: bool = False) -> Path:
    src = SRC / f"{pid}.graphml"
    dst = DST / f"{pid}.graphml"
    if not src.exists():
        raise FileNotFoundError(f"Source graph not found: {src}")
    if dst.exists() and not force:
        return dst
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def extract_demographics(pid: str) -> dict | None:
    """Pull Person demographics from the TB-truncated XML (or full XML as fallback)."""
    cohorts_dir = ROOT / "patient_records" / "cohorts"
    # Prefer any TB-truncated XML for this PID
    candidates = list(cohorts_dir.glob(f"*_{pid}.xml")) if cohorts_dir.exists() else []
    # Fall back to the untruncated XML
    untrunc = ROOT / "patient_records" / "thoracic_xmls" / f"{pid}.xml"
    if untrunc.exists():
        candidates.append(untrunc)
    if not candidates:
        return None
    xml_path = candidates[0]
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    person = root.find(".//person")
    if person is None:
        return None

    def child_text(parent, tag):
        el = parent.find(tag)
        return el.text.strip() if (el is not None and el.text) else None

    demographics = person.find("demographics")
    out = {
        "name": child_text(person, "name") or person.get("name"),  # XMLs vary; keep both shapes
        "date_of_birth": child_text(person, "birthdate") or person.get("birthdate") or person.get("birth_datetime"),
        "sex": (child_text(demographics, "gender") if demographics is not None else None) or person.get("gender"),
        "race": (child_text(demographics, "race") if demographics is not None else None) or person.get("race"),
        "ethnicity": (child_text(demographics, "ethnicity") if demographics is not None else None) or person.get("ethnicity"),
        "_source_xml": str(xml_path.relative_to(ROOT)),
    }
    # Trim DOB to YYYY-MM-DD if it includes a time
    if out["date_of_birth"]:
        out["date_of_birth"] = out["date_of_birth"][:10]
    # Title-case sex for downstream consistency ("MALE" -> "Male")
    if out["sex"]:
        out["sex"] = out["sex"].title()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", required=True, help="Patient ID")
    ap.add_argument("--force", action="store_true", help="Overwrite if destination exists")
    args = ap.parse_args()

    dst = copy_one(args.pid, force=args.force)

    # Extract demographics and stash in the run dir
    demo = extract_demographics(args.pid)
    run_dir = RUNS_ROOT / args.pid
    run_dir.mkdir(parents=True, exist_ok=True)
    demo_path = run_dir / "demographics.json"
    demo_path.write_text(json.dumps(demo or {}, indent=2))

    sz = dst.stat().st_size
    summary = {
        "graph": str(dst),
        "graph_kb": round(sz / 1024),
        "demographics_path": str(demo_path),
        "demographics": demo,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
