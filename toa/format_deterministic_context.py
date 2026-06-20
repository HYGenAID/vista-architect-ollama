"""
Format deterministic retrieval results into LLM context snippets.

Purpose: Surface "best hits" from graph search to guide LLM attention
         during extraction (like showing Google search results before the task).
"""

from typing import Dict, Any, Tuple
from toa.graph import TOAGraph
from toa.deterministic_retrieval import extract_deterministic_context


def format_context_for_llm(graph: TOAGraph, max_snippet_chars: int = 300) -> Tuple[str, Dict[str, str]]:
    """
    Format deterministic retrieval results as LLM context snippet with short references.

    Args:
        graph: Patient graph
        max_snippet_chars: Max chars per text snippet (default: 300)

    Returns:
        (formatted_context, reference_map)
        - formatted_context: Context string to prepend to LLM prompt
        - reference_map: Dict mapping short refs (E1, E2...) to event IDs

    Example output:
        DETERMINISTIC SEARCH HITS (to guide your attention):

        🔍 METASTASIS mentions (5 found):
          Latest note (2024-01-15): "...liver metastases confirmed..."
          Latest radiology (2024-01-10): "...new hepatic lesions..."

        🔍 LYMPH NODE mentions (3 found):
          Latest note (2023-12-20): "...mediastinal lymphadenopathy..."

        🔍 DRIVER MUTATIONS:
          EGFR (2 mentions, latest: 2023-11-05): "...exon 19 deletion..."
          KRAS (1 mention, latest: 2023-11-05): "...wild type..."
    """
    context = extract_deterministic_context(graph)

    # Initialize reference counter and map
    event_counter = 1
    reference_map = {}

    def _assign_ref(event_id: str) -> str:
        """Assign short reference to event and return it."""
        nonlocal event_counter
        ref_id = f"E{event_counter}"
        reference_map[ref_id] = event_id
        event_counter += 1
        return ref_id

    lines = []
    lines.append("=" * 70)
    lines.append("DETERMINISTIC SEARCH HITS (to guide your attention):")
    lines.append("=" * 70)
    lines.append("")

    # Metastasis
    mets = context['metastasis']
    if mets['count'] > 0:
        lines.append(f"🔍 METASTASIS mentions ({mets['count']} found):")
        if mets['latest_note']:
            event = mets['latest_note']
            ref_id = _assign_ref(event.get('event_id', ''))
            text = _truncate(event['text'], max_snippet_chars)
            date = event['date']
            timestamp = event.get('timestamp', '')
            event_type = event.get('type', '')
            meta = f"{date}"
            if timestamp:
                meta += f" {timestamp}"
            lines.append(f"  [{ref_id}] Latest note ({meta}, {event_type}):")
            lines.append(f"    \"{text}\"")
        if mets['latest_radiology']:
            event = mets['latest_radiology']
            ref_id = _assign_ref(event.get('event_id', ''))
            text = _truncate(event['text'], max_snippet_chars)
            date = event['date']
            timestamp = event.get('timestamp', '')
            event_type = event.get('type', '')
            meta = f"{date}"
            if timestamp:
                meta += f" {timestamp}"
            lines.append(f"  [{ref_id}] Latest radiology ({meta}, {event_type}):")
            lines.append(f"    \"{text}\"")
        lines.append("")

    # Lymph nodes
    lymph = context['lymph_nodes']
    if lymph['count'] > 0:
        lines.append(f"🔍 LYMPH NODE mentions ({lymph['count']} found):")
        if lymph['latest_note']:
            event = lymph['latest_note']
            ref_id = _assign_ref(event.get('event_id', ''))
            text = _truncate(event['text'], max_snippet_chars)
            date = event['date']
            timestamp = event.get('timestamp', '')
            event_type = event.get('type', '')
            meta = f"{date}"
            if timestamp:
                meta += f" {timestamp}"
            lines.append(f"  [{ref_id}] Latest note ({meta}, {event_type}):")
            lines.append(f"    \"{text}\"")
        if lymph['latest_radiology']:
            event = lymph['latest_radiology']
            ref_id = _assign_ref(event.get('event_id', ''))
            text = _truncate(event['text'], max_snippet_chars)
            date = event['date']
            timestamp = event.get('timestamp', '')
            event_type = event.get('type', '')
            meta = f"{date}"
            if timestamp:
                meta += f" {timestamp}"
            lines.append(f"  [{ref_id}] Latest radiology ({meta}, {event_type}):")
            lines.append(f"    \"{text}\"")
        lines.append("")

    # Driver mutations
    mutations = context['driver_mutations']
    if mutations:
        lines.append("🔍 DRIVER MUTATIONS:")

        # Sort by gene name
        genes = [k for k in mutations.keys() if k != '_driver_mutation_keyword']
        genes.sort()

        for gene in genes:
            info = mutations[gene]
            count = info['count']
            event = info['latest_event']
            ref_id = _assign_ref(event.get('event_id', ''))
            date = event['date']
            timestamp = event.get('timestamp', '')
            event_type = event.get('type', '')
            text = _truncate(event['text'], max_snippet_chars)
            meta = f"{date}"
            if timestamp:
                meta += f" {timestamp}"
            lines.append(f"  [{ref_id}] {gene} ({count} mention{'s' if count > 1 else ''}, latest: {meta}, {event_type}):")
            lines.append(f"    \"{text}\"")

        # Also show generic "driver mutation" keyword hits
        if '_driver_mutation_keyword' in mutations:
            info = mutations['_driver_mutation_keyword']
            count = info['count']
            event = info['latest_event']
            ref_id = _assign_ref(event.get('event_id', ''))
            date = event['date']
            timestamp = event.get('timestamp', '')
            event_type = event.get('type', '')
            text = _truncate(event['text'], max_snippet_chars)
            meta = f"{date}"
            if timestamp:
                meta += f" {timestamp}"
            lines.append(f"  [{ref_id}] [generic 'driver mutation' keyword] ({count} mention{'s' if count > 1 else ''}, latest: {meta}, {event_type}):")
            lines.append(f"    \"{text}\"")

        lines.append("")

    # Latest Oncology Note (FULL TEXT)
    oncology_note = context.get('latest_oncology_note')
    if oncology_note and oncology_note.get('note'):
        note = oncology_note['note']
        ref_id = _assign_ref(note.get('event_id', ''))
        count = oncology_note['count']
        date = note['date']
        timestamp = note.get('timestamp', '')
        name = note.get('name', '')
        visit_name = note.get('visit_name', '')
        full_text = note.get('full_text', '')

        meta = f"{date}"
        if timestamp:
            meta += f" {timestamp}"

        lines.append(f"🔍 LATEST ONCOLOGY NOTE [{ref_id}] ({count} total oncology notes found):")
        lines.append(f"  Visit: {visit_name}")
        lines.append(f"  Note: {name}")
        lines.append(f"  Date: {meta}")
        lines.append(f"  FULL NOTE:")
        lines.append(f"  {'-'*68}")
        # Show full text (no truncation)
        for line in full_text.split('\n'):
            if line.strip():
                lines.append(f"  {line}")
        lines.append(f"  {'-'*68}")
        lines.append("")

    # Latest Chest CT Report (FULL TEXT)
    chest_ct = context.get('latest_chest_ct')
    if chest_ct and chest_ct.get('report'):
        report = chest_ct['report']
        ref_id = _assign_ref(report.get('event_id', ''))
        count = chest_ct['count']
        date = report['date']
        timestamp = report.get('timestamp', '')
        name = report.get('name', '')
        full_text = report.get('full_text', '')

        meta = f"{date}"
        if timestamp:
            meta += f" {timestamp}"

        lines.append(f"🔍 LATEST CHEST CT REPORT [{ref_id}] ({count} total chest CTs found):")
        lines.append(f"  Study: {name}")
        lines.append(f"  Date: {meta}")
        lines.append(f"  FULL REPORT:")
        lines.append(f"  {'-'*68}")
        # Show full text (no truncation)
        for line in full_text.split('\n'):
            if line.strip():
                lines.append(f"  {line}")
        lines.append(f"  {'-'*68}")
        lines.append("")

    # Smoking status
    smoking = context['smoking']
    if smoking['count'] > 0:
        event = smoking['latest_mention']
        ref_id = _assign_ref(event.get('event_id', ''))
        date = event['date']
        timestamp = event.get('timestamp', '')
        event_type = event.get('type', '')
        text = _truncate(event['text'], max_snippet_chars)
        meta = f"{date}"
        if timestamp:
            meta += f" {timestamp}"
        lines.append(f"🔍 SMOKING status [{ref_id}] ({smoking['count']} mention{'s' if smoking['count'] > 1 else ''}, latest: {meta}, {event_type}):")
        lines.append(f"  \"{text}\"")
        lines.append("")

    # ECOG
    ecog = context['ecog']
    if ecog['count'] > 0:
        event = ecog['latest_mention']
        ref_id = _assign_ref(event.get('event_id', ''))
        date = event['date']
        timestamp = event.get('timestamp', '')
        event_type = event.get('type', '')
        text = _truncate(event['text'], max_snippet_chars)
        meta = f"{date}"
        if timestamp:
            meta += f" {timestamp}"
        lines.append(f"🔍 ECOG/Performance Status [{ref_id}] ({ecog['count']} mention{'s' if ecog['count'] > 1 else ''}, latest: {meta}, {event_type}):")
        lines.append(f"  \"{text}\"")
        lines.append("")

    # Allergies (show ALL, not just latest)
    allergies = context['allergies']
    if allergies['count'] > 0:
        lines.append(f"🔍 ALLERGIES ({allergies['count']} mention{'s' if allergies['count'] > 1 else ''}):")
        # Group by unique text to avoid duplicates
        unique_texts = {}
        for mention in allergies['all_mentions']:
            text = mention['text'][:200]  # Use shorter snippet for allergies
            if text not in unique_texts:
                unique_texts[text] = mention

        for text, mention in list(unique_texts.items())[:5]:  # Max 5 unique allergy mentions
            ref_id = _assign_ref(mention.get('event_id', ''))
            date = mention['date']
            timestamp = mention.get('timestamp', '')
            event_type = mention.get('type', '')
            meta = f"{date}"
            if timestamp:
                meta += f" {timestamp}"
            lines.append(f"  [{ref_id}] ({meta}, {event_type}): \"{text}\"")

        if len(unique_texts) > 5:
            lines.append(f"  ... and {len(unique_texts) - 5} more")

        lines.append("")

    # Conditions (show top 5 earliest)
    conditions = context['conditions']
    if conditions['count'] > 0:
        lines.append(f"🔍 DISTINCT CONDITIONS ({conditions['count']} total, showing earliest 5):")
        for cond in conditions['conditions'][:5]:
            name = cond['name']
            date = cond['earliest_date']
            code = cond.get('code', '')
            if code:
                lines.append(f"  ({date}): {name} [{code}]")
            else:
                lines.append(f"  ({date}): {name}")

        if conditions['count'] > 5:
            lines.append(f"  ... and {conditions['count'] - 5} more")

        lines.append("")

    # Drug exposures (show ALL with first → last date)
    drugs = context['drug_exposures']
    if drugs['count'] > 0:
        lines.append(f"🔍 DISTINCT DRUG EXPOSURES ({drugs['count']} total):")
        for drug in drugs['drugs']:
            name = drug['name']
            first = drug['first_date']
            last = drug['last_date']
            count = drug['mention_count']

            # Format date range
            if first == last:
                date_str = f"{first}"
            else:
                date_str = f"{first} → {last}"

            lines.append(f"  {name} ({date_str}, {count} mention{'s' if count > 1 else ''})")

        lines.append("")

    # Recent imaging (metadata only)
    imaging = context['recent_imaging']
    total_studies = imaging.get('total_studies', 0)
    if total_studies > 0:
        lines.append(f"🔍 RECENT IMAGING STUDIES ({total_studies} total):")

        # Show by modality
        for modality in ['CT', 'MRI', 'PET-CT', 'XR']:
            if modality in imaging:
                studies = imaging[modality]
                lines.append(f"  {modality} ({len(studies)} studies):")
                for study in studies[:3]:  # Max 3 per modality
                    date = study['date']
                    site = study['site']
                    lines.append(f"    - {date}: {site}")
                if len(studies) > 3:
                    lines.append(f"    ... and {len(studies) - 3} more")

        lines.append("")

    lines.append("=" * 70)
    lines.append("")
    lines.append("INSTRUCTIONS FOR TIMELINE EXTRACTION:")
    lines.append("When creating timeline events, include a 'source_event_refs' field")
    lines.append("containing the short IDs (E1, E2, etc.) of the base events you used.")
    lines.append("")
    lines.append("Example:")
    lines.append('  "source_event_refs": ["E3", "E4"]  <- Used oncology note and CT report')
    lines.append("")
    lines.append("This ensures deterministic provenance linking back to source documents.")
    lines.append("=" * 70)
    lines.append("")

    return '\n'.join(lines), reference_map


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def inject_context_into_prompt(
    base_prompt: str,
    graph: TOAGraph,
    injection_marker: str = "{xml_chunk}"
) -> Tuple[str, Dict[str, str]]:
    """
    Inject deterministic context into prompt BEFORE the xml_chunk marker.

    Args:
        base_prompt: Original prompt template with {xml_chunk} marker
        graph: Patient graph
        injection_marker: Where to inject context (default: "{xml_chunk}")

    Returns:
        (enhanced_prompt, reference_map)
        - enhanced_prompt: Prompt with context injected before xml_chunk
        - reference_map: Dict mapping short refs (E1, E2...) to event IDs

    Example:
        >>> prompt_template = "Extract events from:\\n{xml_chunk}"
        >>> enhanced_prompt, ref_map = inject_context_into_prompt(prompt_template, graph)
        >>> # Now enhanced_prompt has search hits before {xml_chunk}
    """
    context_snippet, reference_map = format_context_for_llm(graph)

    # Find the injection point
    if injection_marker not in base_prompt:
        # If no marker found, just prepend context
        return context_snippet + "\n\n" + base_prompt, reference_map

    # Inject context before the marker
    parts = base_prompt.split(injection_marker, 1)
    return parts[0] + context_snippet + "\n" + injection_marker + parts[1], reference_map


# ==========================================================================
# Quick usage example
# ==========================================================================

if __name__ == "__main__":
    """
    Quick test - shows context formatting on a test patient.
    """
    import sys
    from toa.xml_to_graph_hierarchical import build_hierarchical_graph

    if len(sys.argv) < 2:
        print("Usage: python format_deterministic_context.py <patient_id>")
        print("Example: python format_deterministic_context.py 112")
        sys.exit(1)

    patient_id = sys.argv[1]
    xml_path = f"patient_records/thoracic_xmls/{patient_id}.xml"

    print(f"Building graph for patient {patient_id}...")
    graph = build_hierarchical_graph(patient_id, xml_path)

    print(f"\nFormatting deterministic context...")
    context, reference_map = format_context_for_llm(graph)

    print("\n" + context)

    print("\n" + "=" * 70)
    print(f"REFERENCE MAP ({len(reference_map)} entries):")
    print("=" * 70)
    for ref_id, event_id in list(reference_map.items())[:10]:
        print(f"  {ref_id} → {event_id}")
    if len(reference_map) > 10:
        print(f"  ... and {len(reference_map) - 10} more")
