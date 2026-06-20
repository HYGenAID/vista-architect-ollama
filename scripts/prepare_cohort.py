#!/usr/bin/env python3
"""
Prepare a cohort of patients for TOA timeline extraction by creating XML files
truncated to data available before their first tumor board encounter.

Usage:
    python prepare_cohort.py --cohort cohort1 --n 10 --start 2020

This will:
1. Read patient tumor board dates from tb_rt_eval_dataset_1.1.csv
2. Filter patients with TB encounters >= start year
3. Randomly sample N patients
4. For each patient, create a copy of their XML with all data on or after
   the TB date removed (keep only data from TB_date - 1 day or earlier)
5. Save cohort XMLs as: cohort_name_patient_id.xml
6. Generate a manifest JSON with patient IDs and TB dates
"""

import argparse
import csv
import json
import random
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Set
import glob

# Constants
CSV_PATH = Path("patient_records/tb_rt_eval_dataset_1.1.csv")
XML_DIR = Path("patient_records/thoracic_xmls")
COHORT_OUTPUT_DIR = Path("patient_records/cohorts")  # Output directory for cohort files


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        # Try other common formats
        for fmt in ["%m/%d/%Y", "%Y/%m/%d", "%m-%d-%Y"]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unable to parse date: {date_str}")


def load_existing_cohort_patients(cohort_dir: Path = None) -> Set[str]:
    """
    Load all patient IDs from existing cohort manifests to ensure uniqueness.

    Args:
        cohort_dir: Directory to search for manifests (defaults to COHORT_OUTPUT_DIR)

    Returns:
        Set of patient IDs already used in existing cohorts
    """
    if cohort_dir is None:
        cohort_dir = COHORT_OUTPUT_DIR

    used_patients = set()

    # Find all manifest JSON files in cohort directory
    manifest_files = list(cohort_dir.glob("*manifest.json")) + list(cohort_dir.glob("*_manifest.json"))

    for manifest_file in manifest_files:
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)

            # Extract patient IDs from the manifest
            if 'patients' in manifest:
                for patient in manifest['patients']:
                    if 'patient_id' in patient:
                        used_patients.add(patient['patient_id'])

        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"⚠️  Warning: Could not load manifest {manifest_file}: {e}", file=sys.stderr)
            continue

    return used_patients


def load_tb_data(csv_path: Path, start_year: int = None, exclude_patients: Set[str] = None) -> List[Dict]:
    """
    Load tumor board data from CSV.

    Args:
        csv_path: Path to the CSV file
        start_year: Minimum year for TB encounter (optional)
        exclude_patients: Set of patient IDs to exclude (optional)

    Returns list of dicts with keys: patient_id, first_tb_encounter_date
    """
    if exclude_patients is None:
        exclude_patients = set()

    patients = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row['patient_id'].strip()
            tb_date_str = row['first_tb_encounter_date'].strip()

            if not patient_id or not tb_date_str:
                continue

            # Skip if patient is in exclusion list
            if patient_id in exclude_patients:
                continue

            try:
                tb_date = parse_date(tb_date_str)
            except ValueError as e:
                print(f"⚠️  Skipping patient {patient_id}: {e}", file=sys.stderr)
                continue

            # Filter by start year if specified
            if start_year and tb_date.year < start_year:
                continue

            patients.append({
                'patient_id': patient_id,
                'tb_date': tb_date,
                'tb_date_str': tb_date.strftime("%Y-%m-%d")
            })

    return patients


def truncate_xml(xml_path: Path, cutoff_date: datetime) -> str:
    """
    Truncate XML at the first <entry> with timestamp >= cutoff_date.

    Removes everything from that line onwards and adds proper closing tags.
    This prevents data leakage from later encounters/providers/caresites.

    Args:
        xml_path: Path to original patient XML
        cutoff_date: TB encounter date (keep only data BEFORE this date)

    Returns:
        Truncated XML as string with proper closing tags
    """
    import re

    # Read XML as text
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_lines = f.readlines()

    # Track statistics
    kept_entries = 0
    removed_entries = 0
    truncate_line_idx = None

    # Pattern to match <entry timestamp="YYYY-MM-DD HH:MM:SS">
    entry_pattern = re.compile(r'<entry\s+timestamp="([^"]+)"')

    for idx, line in enumerate(xml_lines):
        match = entry_pattern.search(line)
        if match:
            timestamp_str = match.group(1)
            try:
                # Extract date part (YYYY-MM-DD)
                date_str = timestamp_str.split(' ')[0] if ' ' in timestamp_str else timestamp_str
                entry_date = parse_date(date_str)

                if entry_date >= cutoff_date:
                    # Found first entry >= TB date - truncate here
                    truncate_line_idx = idx
                    # Count remaining <entry> tags
                    remaining_text = ''.join(xml_lines[idx:])
                    removed_entries = remaining_text.count('<entry')
                    break
                else:
                    kept_entries += 1

            except ValueError:
                # Could not parse - keep it
                kept_entries += 1

    if truncate_line_idx is None:
        # No entries >= cutoff date found - keep entire XML
        print(f"  No truncation needed: all {kept_entries} entries before TB date")
        with open(xml_path, 'r', encoding='utf-8') as f:
            return f.read()

    # Truncate at the identified line and add closing tags
    truncated_lines = xml_lines[:truncate_line_idx]

    # Add proper closing tags
    truncated_lines.append('    </events>\n')
    truncated_lines.append('  </encounter>\n')
    truncated_lines.append('</eventstream>\n')

    print(f"  Truncated: kept {kept_entries} entries, removed {removed_entries} entries")

    return ''.join(truncated_lines)


def prepare_cohort(cohort_name: str, n_patients: int, start_year: int = None, ensure_unique: bool = True, max_chars: int = None, seed: int = None):
    """
    Main function to prepare a patient cohort.

    Args:
        cohort_name: Name for the cohort (e.g., 'cohort1')
        n_patients: Number of patients to randomly sample
        start_year: Minimum year for TB encounter (optional)
        ensure_unique: If True, exclude patients already in existing cohorts (default: True)
        max_chars: Optional max characters - only select patients whose truncated XML is <= this size
                  (maintains realistic clinical context for TB meeting, avoids weird truncation artifacts)
    """
    # Create output directory if it doesn't exist
    COHORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*80}")
    print(f"Cohort Preparation - {cohort_name}")
    print(f"{'='*80}")
    print(f"CSV: {CSV_PATH}")
    print(f"XML directory: {XML_DIR}")
    print(f"Output directory: {COHORT_OUTPUT_DIR}")
    print(f"Patients to sample: {n_patients}")
    if start_year:
        print(f"Minimum TB year: {start_year}")
    if max_chars:
        print(f"Max chars filter: {max_chars:,} (only select patients with truncated XML <= this size)")
    if seed is not None:
        print(f"Random seed: {seed}")
    print(f"Ensure unique patients: {ensure_unique}")
    print()

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Load existing cohort patients if uniqueness is required
    exclude_patients = set()
    if ensure_unique:
        print("Checking existing cohorts for patient uniqueness...")
        exclude_patients = load_existing_cohort_patients()
        if exclude_patients:
            print(f"Found {len(exclude_patients)} patients in existing cohorts (will be excluded)")
            print(f"Existing patients: {', '.join(sorted(list(exclude_patients)[:5]))}{'...' if len(exclude_patients) > 5 else ''}")
        else:
            print("No existing cohorts found")
        print()

    # Load patient data
    print("Loading tumor board data...")
    patients = load_tb_data(CSV_PATH, start_year, exclude_patients)
    print(f"Found {len(patients)} eligible patients (after exclusions)")

    if len(patients) < n_patients:
        print(f"⚠️  Warning: Only {len(patients)} patients available, requested {n_patients}")
        n_patients = len(patients)

    # If max_chars is specified, pre-filter patients by truncated XML size
    if max_chars:
        print(f"Pre-filtering patients by truncated XML size (<= {max_chars:,} chars)...")
        eligible_patients = []

        for patient in patients:
            patient_id = patient['patient_id']
            tb_date = patient['tb_date']
            xml_path = XML_DIR / f"{patient_id}.xml"

            if not xml_path.exists():
                continue

            try:
                # Truncate and check size
                truncated_xml = truncate_xml(xml_path, tb_date)
                xml_size = len(truncated_xml)

                if xml_size <= max_chars:
                    patient['truncated_size'] = xml_size
                    eligible_patients.append(patient)
            except Exception as e:
                print(f"  ⚠️  Error checking {patient_id}: {e}")
                continue

        print(f"  Found {len(eligible_patients)} patients with XML <= {max_chars:,} chars (out of {len(patients)} total)")

        if len(eligible_patients) < n_patients:
            print(f"  ⚠️  Warning: Only {len(eligible_patients)} eligible patients found, requested {n_patients}")
            n_patients = len(eligible_patients)

        # Sample from eligible patients
        sampled = random.sample(eligible_patients, n_patients)
        print(f"  Randomly sampled {n_patients} from eligible patients")
    else:
        # No size filter - sample from all patients
        sampled = random.sample(patients, n_patients)
        print(f"Randomly sampled {n_patients} patients")

    print()

    # Process each patient
    manifest = {
        'cohort_name': cohort_name,
        'n_patients': n_patients,
        'start_year': start_year,
        'seed': seed,
        'created': datetime.now().isoformat(),
        'patients': []
    }

    success_count = 0

    for i, patient in enumerate(sampled, 1):
        patient_id = patient['patient_id']
        tb_date = patient['tb_date']

        print(f"[{i}/{n_patients}] Processing patient {patient_id} (TB date: {patient['tb_date_str']})")

        # Find original XML
        xml_path = XML_DIR / f"{patient_id}.xml"
        if not xml_path.exists():
            print(f"  ⚠️  XML not found: {xml_path}")
            continue

        # Truncate XML to data before TB date
        cutoff_date = tb_date  # Remove everything >= TB date (keep only < TB date)
        try:
            truncated_xml = truncate_xml(xml_path, cutoff_date)
        except Exception as e:
            print(f"  ❌ Error truncating XML: {e}")
            continue

        # Save cohort XML to cohorts directory
        cohort_xml_path = COHORT_OUTPUT_DIR / f"{cohort_name}_{patient_id}.xml"
        try:
            with open(cohort_xml_path, 'w', encoding='utf-8') as f:
                f.write(truncated_xml)

            # Show size info
            xml_size = len(truncated_xml)
            size_info = f"{xml_size:,} chars"
            if 'truncated_size' in patient:
                size_info += " (pre-checked)"

            print(f"  ✅ Saved: {cohort_xml_path.name} ({size_info})")
            success_count += 1
        except Exception as e:
            print(f"  ❌ Error saving XML: {e}")
            continue

        # Add to manifest
        manifest['patients'].append({
            'patient_id': patient_id,
            'tb_date': patient['tb_date_str'],
            'original_xml': str(xml_path.name),
            'cohort_xml': str(cohort_xml_path.name)
        })

        print()

    # Save manifest to cohorts directory
    manifest_path = COHORT_OUTPUT_DIR / f"{cohort_name}_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"{'='*80}")
    print(f"✅ Cohort preparation complete!")
    print(f"   Successfully processed: {success_count}/{n_patients} patients")
    print(f"   Manifest saved: {manifest_path}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare patient cohort XMLs truncated before tumor board encounters"
    )
    parser.add_argument('--cohort', required=True,
                        help='Cohort name (e.g., cohort1)')
    parser.add_argument('--n', type=int, required=True,
                        help='Number of patients to randomly sample')
    parser.add_argument('--start', type=int, default=None,
                        help='Earliest TB encounter year to include (optional)')
    parser.add_argument('--no-unique-check', action='store_true',
                        help='Disable uniqueness check (allow duplicate patients from existing cohorts)')
    parser.add_argument('--max-chars', type=int, default=None,
                        help='Max characters - only select patients whose truncated XML is <= this size (default: None = no limit)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible sampling (default: None = non-deterministic)')

    args = parser.parse_args()

    # Validate inputs
    if args.n <= 0:
        print("Error: --n must be positive", file=sys.stderr)
        sys.exit(1)

    if not CSV_PATH.exists():
        print(f"Error: CSV file not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    if not XML_DIR.exists():
        print(f"Error: XML directory not found: {XML_DIR}", file=sys.stderr)
        sys.exit(1)

    # Run cohort preparation
    ensure_unique = not args.no_unique_check
    prepare_cohort(args.cohort, args.n, args.start, ensure_unique, args.max_chars, args.seed)


if __name__ == '__main__':
    main()
