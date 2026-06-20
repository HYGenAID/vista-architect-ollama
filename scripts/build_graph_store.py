#!/usr/bin/env python3
"""
Build a cohort graph store from patient XMLs.

Reads the TB dataset CSV, filters to patients with TB dates >= start_year,
truncates each patient's XML at the TB date, builds a Lumia graph, and saves
to the graph store directory.

Usage:
    python build_graph_store.py --start-year 2020 --workers 8
    python build_graph_store.py --start-year 2020 --workers 1 --resume
    python build_graph_store.py --start-year 2020 --workers 8 --benchmark
"""

import argparse
import json
import sys
import time

from toa.graph_store import CohortGraphStore


def main():
    parser = argparse.ArgumentParser(
        description="Build cohort graph store from patient XMLs"
    )
    parser.add_argument(
        '--csv', default='patient_records/tb_rt_eval_dataset_1.1.csv',
        help='Path to TB dataset CSV (default: patient_records/tb_rt_eval_dataset_1.1.csv)'
    )
    parser.add_argument(
        '--xml-dir', default='patient_records/thoracic_xmls',
        help='Directory containing patient XML files'
    )
    parser.add_argument(
        '--store-dir', default='graph_store',
        help='Output graph store directory (default: graph_store)'
    )
    parser.add_argument(
        '--start-year', type=int, default=2020,
        help='Minimum TB encounter year (default: 2020)'
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Number of parallel workers (default: 8)'
    )
    parser.add_argument(
        '--resume', action='store_true', default=True,
        help='Skip already-built patients (default: True)'
    )
    parser.add_argument(
        '--no-resume', action='store_true',
        help='Rebuild all patients (ignore existing graphs)'
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Benchmark mode: clean build, detailed timing output'
    )
    parser.add_argument(
        '--no-namespace', action='store_true',
        help='Disable node ID namespacing (not recommended for cohort use)'
    )

    args = parser.parse_args()

    resume = not args.no_resume
    if args.benchmark:
        resume = False
        print("BENCHMARK MODE: clean build, no resume")

    print("=" * 70)
    print("COHORT GRAPH STORE BUILDER")
    print("=" * 70)
    print(f"CSV:        {args.csv}")
    print(f"XML dir:    {args.xml_dir}")
    print(f"Store dir:  {args.store_dir}")
    print(f"Start year: {args.start_year}")
    print(f"Workers:    {args.workers}")
    print(f"Resume:     {resume}")
    print(f"Namespace:  {not args.no_namespace}")
    print("=" * 70)
    print()

    store = CohortGraphStore(args.store_dir)

    t0 = time.time()
    report = store.build_cohort(
        csv_path=args.csv,
        xml_dir=args.xml_dir,
        start_year=args.start_year,
        n_workers=args.workers,
        resume=resume,
        namespace_ids=not args.no_namespace,
    )
    total_time = time.time() - t0

    # Print report
    print()
    print("=" * 70)
    print("BUILD REPORT")
    print("=" * 70)
    print(f"Success:      {report['success']}")
    print(f"Failed:       {report['failed']}")
    print(f"Skipped:      {report['skipped']}")
    print(f"Total stored: {report['total_in_store']}")
    print(f"Total time:   {total_time:.1f}s ({total_time/60:.1f}m)")

    timing = report.get("timing", {})
    if timing:
        print(f"\nPer-patient build time:")
        print(f"  Median: {timing['median']:.3f}s")
        print(f"  Mean:   {timing['mean']:.3f}s")
        print(f"  P90:    {timing['p90']:.3f}s")
        print(f"  P99:    {timing['p99']:.3f}s")
        print(f"  Min:    {timing['min']:.3f}s")
        print(f"  Max:    {timing['max']:.3f}s")

    # Print store summary
    summary = store.summary()
    if summary.get("total_patients"):
        print(f"\nStore summary:")
        print(f"  Patients:    {summary['total_patients']}")
        print(f"  Store size:  {summary['store_size_gb']:.2f} GB")
        nc = summary['node_counts']
        print(f"  Total nodes: {nc['total']:,}")
        print(f"  Nodes/patient: median={nc['median']:,}, min={nc['min']:,}, max={nc['max']:,}")
        xs = summary['xml_sizes']
        print(f"  XML input:   {xs['total_gb']:.2f} GB total, {xs['median_mb']:.1f} MB median")

    print("=" * 70)

    # Save report to store dir
    report_path = store.store_dir / "build_report.json"
    report["total_wall_time_seconds"] = round(total_time, 1)
    report["store_summary"] = summary
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

    if report['failed'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
