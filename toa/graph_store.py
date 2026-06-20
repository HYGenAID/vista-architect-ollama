"""
Cohort Graph Store — Persistent graph store for pre-built patient Lumia graphs.

Stores individual patient Lumia graphs (pre-truncated at tumor board date)
with a JSON index for fast metadata access. Supports batch building with
multiprocessing, feature extraction, and cohort sampling.

Directory structure:
    graph_store/
        index.json          # Cohort metadata + per-patient index
        graphs/
            {pid}.graphml   # Individual patient Lumia graphs

Usage:
    store = CohortGraphStore("graph_store")
    store.build_cohort("patient_records/tb_rt_eval_dataset_1.1.csv",
                       "patient_records/thoracic_xmls", start_year=2020, n_workers=8)
    graph = store.load_patient_graph("136056052")
"""

import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional

from toa.graph import TOAGraph
from toa.xml_to_graph_hierarchical import build_hierarchical_graph


def _compute_xml_hash(xml_text: str) -> str:
    """Compute SHA-256 hash of XML content."""
    return hashlib.sha256(xml_text.encode('utf-8')).hexdigest()[:16]


def _get_graph_stats(graph: TOAGraph) -> Dict[str, Any]:
    """Extract statistics from a built graph."""
    node_counts = defaultdict(int)
    event_type_counts = defaultdict(int)
    dates = []

    for _, data in graph.G.nodes(data=True):
        node_type = data.get('node_type', 'unknown')
        node_counts[node_type] += 1
        if node_type == 'Event':
            event_type_counts[data.get('type', 'unknown')] += 1
            date = data.get('date', '')
            if date:
                dates.append(date)

    date_range = {}
    if dates:
        dates.sort()
        date_range = {"start": dates[0], "end": dates[-1]}

    return {
        "node_count": graph.G.number_of_nodes(),
        "edge_count": graph.G.number_of_edges(),
        "event_count": node_counts.get('Event', 0),
        "visit_count": node_counts.get('Visit', 0),
        "xml_fragment_count": node_counts.get('XMLFragment', 0),
        "date_range": date_range,
        "event_type_distribution": dict(event_type_counts),
    }


def _build_single_patient(args: tuple) -> Dict[str, Any]:
    """
    Build graph for a single patient. Designed for multiprocessing.Pool.

    Args:
        args: (patient_id, xml_path, tb_date_str, store_dir, namespace_ids)

    Returns:
        Result dict with status, timing, and metadata.
    """
    # Import here to avoid pickling issues with multiprocessing
    import io
    import contextlib
    from prepare_cohort import truncate_xml, parse_date

    patient_id, xml_path, tb_date_str, store_dir, namespace_ids = args

    result = {
        "patient_id": patient_id,
        "status": "error",
        "error": None,
        "build_time_seconds": 0,
    }

    try:
        t0 = time.time()

        # Get XML size
        xml_size = os.path.getsize(xml_path)
        result["xml_size_bytes"] = xml_size

        # Truncate XML at TB date (suppress truncate_xml print output)
        tb_date = parse_date(tb_date_str)
        with contextlib.redirect_stdout(io.StringIO()):
            truncated_xml = truncate_xml(Path(xml_path), tb_date)

        # Build hierarchical Lumia graph
        graph = build_hierarchical_graph(
            patient_id,
            xml_source=truncated_xml,
            namespace_ids=namespace_ids,
        )

        # Save graph (ensure directory exists — needed for multiprocessing)
        graph_path = Path(store_dir) / "graphs" / f"{patient_id}.graphml"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph.save(str(graph_path))

        # Collect stats
        stats = _get_graph_stats(graph)
        build_time = time.time() - t0

        result.update({
            "status": "success",
            "tb_date": tb_date_str,
            "graph_path": f"graphs/{patient_id}.graphml",
            "xml_hash": _compute_xml_hash(truncated_xml),
            "build_time_seconds": round(build_time, 3),
            "graphml_size_bytes": os.path.getsize(graph_path),
            **stats,
        })

    except Exception as e:
        result["error"] = str(e)
        result["build_time_seconds"] = round(time.time() - t0, 3)

    return result


class CohortGraphStore:
    """Persistent store for pre-built patient Lumia graphs."""

    def __init__(self, store_dir: str = "graph_store"):
        self.store_dir = Path(store_dir)
        self.index_path = self.store_dir / "index.json"
        self.graphs_dir = self.store_dir / "graphs"
        self._index = None

    def _ensure_dirs(self):
        """Create store directories if needed."""
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> Dict[str, Any]:
        """Load or initialize the index."""
        if self._index is not None:
            return self._index

        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self._index = json.load(f)
        else:
            self._index = {
                "store_version": "1.0",
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat(),
                "total_patients": 0,
                "patients": {},
            }
        return self._index

    def _save_index(self):
        """Persist index to disk."""
        if self._index is None:
            return
        self._index["updated"] = datetime.now().isoformat()
        self._index["total_patients"] = len(self._index["patients"])
        with open(self.index_path, 'w') as f:
            json.dump(self._index, f, indent=2)

    # ========== Build Methods ==========

    def build_cohort(
        self,
        csv_path: str,
        xml_dir: str,
        start_year: int = 2020,
        n_workers: int = 8,
        resume: bool = True,
        namespace_ids: bool = True,
    ) -> Dict[str, Any]:
        """
        Batch build graph store for a cohort of patients.

        Args:
            csv_path: Path to TB dataset CSV (patient_id, first_tb_encounter_date)
            xml_dir: Directory containing patient XML files
            start_year: Minimum TB encounter year
            n_workers: Number of parallel workers
            resume: Skip patients whose graphs already exist
            namespace_ids: Prefix node IDs with patient_id

        Returns:
            Build report dict with success/fail counts and timing.
        """
        self._ensure_dirs()
        index = self._load_index()

        # Load patient list from CSV
        patients = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row['patient_id'].strip()
                tb_date = row['first_tb_encounter_date'].strip()
                if not pid or not tb_date:
                    continue
                if tb_date < f"{start_year}":
                    continue
                xml_path = Path(xml_dir) / f"{pid}.xml"
                if not xml_path.exists():
                    continue
                patients.append((pid, str(xml_path), tb_date))

        print(f"Found {len(patients)} eligible patients (TB date >= {start_year})")

        # Filter out already-built if resuming
        if resume:
            already_built = set(index["patients"].keys())
            # Also check if graphml file actually exists
            to_build = []
            for pid, xml_path, tb_date in patients:
                graph_file = self.graphs_dir / f"{pid}.graphml"
                if pid in already_built and graph_file.exists():
                    continue
                to_build.append((pid, xml_path, tb_date))
            skipped = len(patients) - len(to_build)
            if skipped > 0:
                print(f"Resuming: skipping {skipped} already-built patients")
            patients_to_process = to_build
        else:
            patients_to_process = patients

        if not patients_to_process:
            print("Nothing to build.")
            return {"success": 0, "failed": 0, "skipped": len(patients)}

        print(f"Building graphs for {len(patients_to_process)} patients with {n_workers} workers...")

        # Prepare args for multiprocessing
        args_list = [
            (pid, xml_path, tb_date, str(self.store_dir), namespace_ids)
            for pid, xml_path, tb_date in patients_to_process
        ]

        # Build with multiprocessing
        t_start = time.time()
        results = []

        if n_workers <= 1:
            # Single-threaded for debugging
            for i, args in enumerate(args_list):
                result = _build_single_patient(args)
                results.append(result)
                status = "OK" if result["status"] == "success" else f"FAIL: {result['error']}"
                print(f"  [{i+1}/{len(args_list)}] {result['patient_id']}: {status} ({result['build_time_seconds']:.2f}s)")
        else:
            with Pool(n_workers) as pool:
                for i, result in enumerate(pool.imap_unordered(_build_single_patient, args_list)):
                    results.append(result)
                    status = "OK" if result["status"] == "success" else f"FAIL: {result['error']}"
                    nodes = result.get('node_count', '?')
                    print(f"  [{i+1}/{len(args_list)}] {result['patient_id']}: {status} ({result['build_time_seconds']:.2f}s, {nodes} nodes)")

        total_time = time.time() - t_start

        # Update index with successful builds
        success_count = 0
        fail_count = 0
        for result in results:
            if result["status"] == "success":
                pid = result["patient_id"]
                # Store in index (exclude transient fields)
                entry = {k: v for k, v in result.items() if k not in ("status", "error")}
                entry["features"] = {}  # Populated later by feature extraction
                index["patients"][pid] = entry
                success_count += 1
            else:
                fail_count += 1

        self._save_index()

        # Build report
        build_times = [r["build_time_seconds"] for r in results if r["status"] == "success"]
        build_times.sort()

        report = {
            "success": success_count,
            "failed": fail_count,
            "skipped": len(patients) - len(patients_to_process),
            "total_in_store": len(index["patients"]),
            "total_time_seconds": round(total_time, 1),
            "timing": {},
        }

        if build_times:
            report["timing"] = {
                "median": round(build_times[len(build_times) // 2], 3),
                "mean": round(sum(build_times) / len(build_times), 3),
                "p90": round(build_times[int(len(build_times) * 0.9)], 3),
                "p99": round(build_times[int(len(build_times) * 0.99)], 3),
                "min": round(build_times[0], 3),
                "max": round(build_times[-1], 3),
            }

        # Print errors
        errors = [r for r in results if r["status"] == "error"]
        if errors:
            print(f"\nFailed patients ({len(errors)}):")
            for e in errors[:10]:
                print(f"  {e['patient_id']}: {e['error']}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")

        return report

    # ========== Load Methods ==========

    def load_patient_graph(self, patient_id: str) -> TOAGraph:
        """Load a patient's Lumia graph from the store."""
        graph_path = self.graphs_dir / f"{patient_id}.graphml"
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph not found in store: {graph_path}")
        return TOAGraph.load(str(graph_path))

    def get_patient_metadata(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient metadata from index without loading graph."""
        index = self._load_index()
        return index["patients"].get(patient_id)

    def list_patients(self, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        List patients in the store, optionally filtered.

        Args:
            filters: Optional dict with filter criteria:
                - min_nodes: Minimum node count
                - max_nodes: Maximum node count
                - min_year: Minimum TB date year
                - max_year: Maximum TB date year

        Returns:
            List of patient metadata dicts.
        """
        index = self._load_index()
        patients = list(index["patients"].values())

        if filters:
            if "min_nodes" in filters:
                patients = [p for p in patients if p.get("node_count", 0) >= filters["min_nodes"]]
            if "max_nodes" in filters:
                patients = [p for p in patients if p.get("node_count", 0) <= filters["max_nodes"]]
            if "min_year" in filters:
                patients = [p for p in patients if p.get("tb_date", "")[:4] >= str(filters["min_year"])]
            if "max_year" in filters:
                patients = [p for p in patients if p.get("tb_date", "")[:4] <= str(filters["max_year"])]

        return patients

    def has_patient(self, patient_id: str) -> bool:
        """Check if a patient exists in the store."""
        index = self._load_index()
        return patient_id in index["patients"]

    # ========== Feature Extraction ==========

    def extract_patient_features(self, patient_id: str) -> Dict[str, Any]:
        """
        Extract deterministic features from a patient's graph.

        Runs DeterministicRetriever on the Lumia graph to populate
        the features field in the index.
        """
        from toa.deterministic_retrieval import DeterministicRetriever

        graph = self.load_patient_graph(patient_id)
        retriever = DeterministicRetriever(graph)
        context = retriever.get_comprehensive_context()

        features = {
            "metastasis_mention_count": len(context.get("metastasis_mentions", {}).get("notes", []))
                                       + len(context.get("metastasis_mentions", {}).get("radiology", [])),
            "lymph_node_mentioned": len(context.get("lymph_node_mentions", [])) > 0,
            "driver_mutations_found": list(context.get("driver_mutations", {}).keys()),
            "smoking_mentioned": bool(context.get("smoking_status")),
            "ecog_mentioned": bool(context.get("ecog_status")),
            "radiation_mentioned": bool(context.get("radiation_therapy_mentions")),
            "drug_count": len(context.get("drug_exposures", [])),
            "condition_count": len(context.get("distinct_conditions", [])),
            "allergy_mentioned": bool(context.get("allergy_mentions")),
        }

        # Update index
        index = self._load_index()
        if patient_id in index["patients"]:
            index["patients"][patient_id]["features"] = features
            self._save_index()

        return features

    def build_feature_index(self, n_workers: int = 1) -> Dict[str, Any]:
        """
        Run feature extraction on all patients in the store.

        Returns:
            Report with success/fail counts.
        """
        index = self._load_index()
        patient_ids = list(index["patients"].keys())

        print(f"Extracting features for {len(patient_ids)} patients...")
        success = 0
        failed = 0

        for i, pid in enumerate(patient_ids):
            try:
                self.extract_patient_features(pid)
                success += 1
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(patient_ids)}] features extracted...")
            except Exception as e:
                print(f"  [{i+1}/{len(patient_ids)}] {pid}: FAIL - {e}")
                failed += 1

        print(f"Feature extraction complete: {success} success, {failed} failed")
        return {"success": success, "failed": failed}

    def export_feature_matrix(self) -> Any:
        """
        Export patient-by-feature matrix as pandas DataFrame.

        Returns:
            pandas DataFrame with patients as rows and features as columns.
        """
        import pandas as pd

        index = self._load_index()
        rows = []

        for pid, entry in index["patients"].items():
            row = {
                "patient_id": pid,
                "tb_date": entry.get("tb_date"),
                "node_count": entry.get("node_count", 0),
                "event_count": entry.get("event_count", 0),
            }
            # Add features
            features = entry.get("features", {})
            for k, v in features.items():
                if isinstance(v, list):
                    row[k] = ",".join(v) if v else ""
                    row[f"{k}_count"] = len(v)
                else:
                    row[k] = v
            rows.append(row)

        return pd.DataFrame(rows)

    # ========== Store Info ==========

    def summary(self) -> Dict[str, Any]:
        """Get store summary statistics."""
        index = self._load_index()
        patients = index["patients"]

        if not patients:
            return {"total_patients": 0}

        node_counts = [p.get("node_count", 0) for p in patients.values()]
        build_times = [p.get("build_time_seconds", 0) for p in patients.values()]
        xml_sizes = [p.get("xml_size_bytes", 0) for p in patients.values()]

        total_graphml = sum(p.get("graphml_size_bytes", 0) for p in patients.values())

        return {
            "total_patients": len(patients),
            "store_size_gb": round(total_graphml / 1024**3, 2),
            "node_counts": {
                "total": sum(node_counts),
                "median": sorted(node_counts)[len(node_counts) // 2],
                "min": min(node_counts),
                "max": max(node_counts),
            },
            "build_times": {
                "median": round(sorted(build_times)[len(build_times) // 2], 3),
                "total": round(sum(build_times), 1),
            },
            "xml_sizes": {
                "total_gb": round(sum(xml_sizes) / 1024**3, 2),
                "median_mb": round(sorted(xml_sizes)[len(xml_sizes) // 2] / 1024**2, 1),
            },
        }
