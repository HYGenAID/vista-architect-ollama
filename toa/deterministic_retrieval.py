"""
Deterministic Clinical Info Retrieval from Hierarchical Graph

GOAL: Surface relevant context snippets (like Google search best hits)
      to guide LLM attention, NOT to provide structured answers.

Strategy:
- Search for keywords (metasta*, lymph*, gene names, etc.)
- Find latest mentions (preferably latest note + latest radiology)
- Return event IDs + text snippets for LLM context
- Let LLM interpret - we just point it to the right data

Key Variables:
- Metastasis mentions (latest note + radiology)
- Lymph node mentions (latest note + radiology)
- Driver mutation gene mentions (EGFR, ALK, KRAS, etc.)
- TNM components (T, N, M separately - don't worry about exact staging)
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from toa.graph import TOAGraph


class DeterministicRetriever:
    """
    Deterministic retrieval of clinical facts from hierarchical graph.

    Like Google search: finds best matching events for keywords,
    returns text snippets to guide LLM attention.
    """

    # Gene list from prompts/timeline_compact.txt
    DRIVER_GENES = [
        'EGFR', 'ALK', 'KRAS', 'PD-L1', 'ROS1', 'BRAF',
        'MET', 'RET', 'NTRK', 'ERBB2', 'HER2', 'NRG1'
    ]

    def __init__(self, graph: TOAGraph):
        """
        Initialize retriever with patient graph.

        Args:
            graph: TOAGraph (hierarchical structure with Visit → Events)
        """
        self.graph = graph

    def _search_keyword(self, keyword: str, context_chars: int = 250,
                        case_sensitive: bool = False, word_boundary: bool = False) -> List[Dict[str, Any]]:
        """
        Search for keyword in all events, extracting ±context_chars around match.

        Args:
            keyword: Search term
            context_chars: Characters to show before/after keyword match (default: 250)
            case_sensitive: If True, match exact case (useful for gene names like EGFR vs eGFR)
            word_boundary: If True, require word boundaries around match (avoids MET matching 'Method')

        Returns:
            List of matching events with metadata and context snippet
        """
        import re
        matches = []

        # Build regex pattern
        pattern_str = re.escape(keyword)
        if word_boundary:
            pattern_str = r'\b' + pattern_str + r'\b'
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(pattern_str, flags)

        for node_id, data in self.graph.G.nodes(data=True):
            if data.get('node_type') != 'Event':
                continue

            # Combine text fields
            text_fields = [
                data.get('value', ''),
                data.get('name', ''),
                data.get('description', '')
            ]
            text = ' '.join(text_fields)

            match = pattern.search(text)
            if match:
                # Find keyword position and extract ±context_chars around it
                match_pos = match.start()
                start = max(0, match_pos - context_chars)
                end = min(len(text), match.end() + context_chars)

                snippet = text[start:end]

                # Add ellipsis if truncated
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet = snippet + "..."

                matches.append({
                    'event_id': node_id,
                    'date': data.get('date', ''),
                    'timestamp': data.get('timestamp', ''),
                    'type': data.get('type'),
                    'name': data.get('name', ''),
                    'visit_id': data.get('visit_id', ''),
                    'text': snippet,
                })

        return matches

    def get_latest_metastasis_mentions(self) -> Dict[str, Any]:
        """
        Find latest mentions of 'metasta*' in notes and radiology.

        Returns best hits to guide LLM attention (not trying to parse status).

        Returns:
            {
                'latest_note': {'event_id': '...', 'date': '...', 'text': '...'},
                'latest_radiology': {'event_id': '...', 'date': '...', 'text': '...'},
                'count': number of matches
            }
        """
        keyword = 'metasta'  # Catches metastasis, metastases, metastatic

        # Find all matching events
        matches = self._search_keyword(keyword)

        if not matches:
            return {
                'latest_note': None,
                'latest_radiology': None,
                'count': 0
            }

        # Separate by type
        notes = [m for m in matches if m['type'] == 'note']
        radiology_keywords = ['ct', 'mri', 'pet', 'xr', 'x-ray', 'ultrasound', 'imaging', 'radiology']
        radiology = [m for m in matches if any(kw in m.get('name', '').lower() for kw in radiology_keywords)]

        # Get latest of each
        latest_note = max(notes, key=lambda e: e['date']) if notes else None
        latest_rad = max(radiology, key=lambda e: e['date']) if radiology else None

        return {
            'latest_note': latest_note,
            'latest_radiology': latest_rad,
            'count': len(matches)
        }

    def get_latest_lymph_node_mentions(self) -> Dict[str, Any]:
        """
        Find latest mentions of 'lymph' in notes and radiology.

        Returns:
            {
                'latest_note': {...},
                'latest_radiology': {...},
                'count': ...
            }
        """
        # Search multiple keywords to catch all lymph node references
        keywords = ['lymph', 'adenopathy', 'nodal']

        matches = []
        seen = set()
        for kw in keywords:
            for m in self._search_keyword(kw):
                if m['event_id'] not in seen:
                    seen.add(m['event_id'])
                    matches.append(m)

        if not matches:
            return {
                'latest_note': None,
                'latest_radiology': None,
                'count': 0
            }

        # Separate by type
        notes = [m for m in matches if m['type'] == 'note']
        radiology_keywords = ['ct', 'mri', 'pet', 'xr', 'x-ray', 'ultrasound', 'imaging', 'radiology']
        radiology = [m for m in matches if any(kw in m.get('name', '').lower() for kw in radiology_keywords)]

        # Get latest
        latest_note = max(notes, key=lambda e: e['date']) if notes else None
        latest_rad = max(radiology, key=lambda e: e['date']) if radiology else None

        return {
            'latest_note': latest_note,
            'latest_radiology': latest_rad,
            'count': len(matches)
        }

    def get_driver_mutation_mentions(self) -> Dict[str, Any]:
        """
        Find latest mentions of driver genes + "driver mut*".

        Uses case-sensitive word-boundary matching to avoid false positives:
        - EGFR (not eGFR renal function)
        - ALK (not Alk Phos / alkaline phosphatase)
        - MET (not "method", "metformin", etc.)
        - RET (not "retired", "return", etc.)

        Returns:
            {
                'by_gene': {
                    'EGFR': {'latest_event': {...}, 'count': 3},
                    'ALK': {'latest_event': {...}, 'count': 1},
                    ...
                },
                'driver_mutation_keyword': {'latest_event': {...}, 'count': 2}
            }
        """
        results = {}

        # Search for each driver gene with case-sensitive word boundaries
        # This prevents: eGFR→EGFR, Alk Phos→ALK, method→MET, return→RET
        for gene in self.DRIVER_GENES:
            matches = self._search_keyword(gene, case_sensitive=True, word_boundary=True)
            if matches:
                # Get latest mention
                latest = max(matches, key=lambda e: e['date'])
                results[gene] = {
                    'latest_event': latest,
                    'count': len(matches)
                }

        # Also search for "driver mut" (case-insensitive is fine here)
        driver_mut_matches = self._search_keyword('driver mut')
        if driver_mut_matches:
            latest = max(driver_mut_matches, key=lambda e: e['date'])
            results['_driver_mutation_keyword'] = {
                'latest_event': latest,
                'count': len(driver_mut_matches)
            }

        return results

    def get_molecular_testing_mentions(self) -> Dict[str, Any]:
        """
        Find molecular/genomic testing mentions (NGS panels, liquid biopsy, etc.).

        Catches testing that may not mention specific gene names — e.g.,
        "UCSF500: no pathogenic variants" or "FoundationOne CDx: TMB low".

        Returns:
            {'mentions': [{'event_id': ..., 'date': ..., 'text': ...}], 'count': int}
        """
        keywords = [
            'UCSF500', 'FoundationOne', 'Guardant', 'Tempus',
            'NGS panel', 'next-generation sequencing', 'genomic profiling',
            'molecular testing', 'molecular profiling',
            'pathogenic variant', 'no pathogenic',
            'tumor mutational burden',
            'microsatellite', 'MSI',
        ]

        matches = []
        seen = set()
        for kw in keywords:
            for m in self._search_keyword(kw):
                if m['event_id'] not in seen:
                    seen.add(m['event_id'])
                    matches.append(m)

        # Sort by date descending
        matches.sort(key=lambda e: e['date'], reverse=True)

        return {
            'mentions': matches[:5],  # Top 5 most recent
            'count': len(matches)
        }

    def get_latest_oncology_note(self) -> Dict[str, Any]:
        """
        Get latest oncology clinic note (FULL TEXT).

        Uses proper XML metadata:
        1. Find Visit nodes with "ONCOLOGY" in care site name
        2. Get notes within those visits via VISIT_CONTAINS edges
        3. Return latest note from latest oncology visit

        Returns:
            {
                'note': {
                    'event_id': '...',
                    'date': '...',
                    'timestamp': '...',
                    'name': '...',
                    'visit_name': '...',
                    'full_text': '...' (complete note, not truncated)
                },
                'count': total number of oncology notes found
            }
        """
        oncology_notes = []

        # Find all oncology visits
        oncology_visit_ids = []
        for node_id, data in self.graph.G.nodes(data=True):
            if data.get('node_type') == 'Visit':
                visit_name = data.get('name', '').upper()
                # Match oncology visits or tumor board visits
                if 'ONCOLOGY' in visit_name or 'TUMOR BOARD' in visit_name:
                    oncology_visit_ids.append(node_id)

        # Get notes from oncology visits
        for visit_id in oncology_visit_ids:
            visit_data = self.graph.G.nodes[visit_id]
            visit_name = visit_data.get('name', '')

            # Find notes in this visit via VISIT_CONTAINS edges
            for u, v, edge_data in self.graph.G.edges(visit_id, data=True):
                if edge_data.get('edge_type') == 'VISIT_CONTAINS':
                    event_data = self.graph.G.nodes[v]

                    # Only get notes (not labs, procedures, etc.)
                    if event_data.get('type') == 'note':
                        # Get full text (don't truncate)
                        text_fields = [
                            event_data.get('value', ''),
                            event_data.get('name', ''),
                            event_data.get('description', '')
                        ]
                        full_text = ' '.join(text_fields)

                        oncology_notes.append({
                            'event_id': v,
                            'date': event_data.get('date', ''),
                            'timestamp': event_data.get('timestamp', ''),
                            'name': event_data.get('name', ''),
                            'visit_name': visit_name,
                            'full_text': full_text
                        })

        if not oncology_notes:
            return {'note': None, 'count': 0}

        # Get latest
        latest = max(oncology_notes, key=lambda n: n['date'])

        return {
            'note': latest,
            'count': len(oncology_notes)
        }

    def get_latest_chest_ct_report(self) -> Dict[str, Any]:
        """
        Get latest chest CT radiology report (FULL TEXT).

        Uses Gemini's algorithm:
        1. Find notes (type='note') with report structure keywords
        2. Check text contains CT modality + chest anatomy keywords
        3. Return latest by timestamp

        Returns:
            {
                'report': {
                    'event_id': '...',
                    'date': '...',
                    'timestamp': '...',
                    'name': '...',
                    'full_text': '...' (complete report, not truncated)
                },
                'count': total number of chest CT reports found
            }
        """
        ct_reports = []

        # First, get oncology visit IDs to exclude their notes
        oncology_visit_ids = set()
        for node_id, data in self.graph.G.nodes(data=True):
            if data.get('node_type') == 'Visit':
                visit_name = data.get('name', '').upper()
                if 'ONCOLOGY' in visit_name:
                    oncology_visit_ids.add(node_id)

        for node_id, data in self.graph.G.nodes(data=True):
            if data.get('node_type') != 'Event':
                continue

            # Must be a note (radiology reports are notes, not procedures)
            if data.get('type') != 'note':
                continue

            # Exclude notes from oncology visits (they summarize CT results but aren't the actual report)
            visit_id = data.get('visit_id')
            if visit_id in oncology_visit_ids:
                continue

            # Get full text
            text_fields = [
                data.get('value', ''),
                data.get('name', ''),
                data.get('description', '')
            ]
            full_text = ' '.join(text_fields)
            text_upper = full_text.upper()

            # Check for CT modality keywords
            has_ct = ('CT' in text_upper or 'COMPUTED TOMOGRAPHY' in text_upper)
            if not has_ct:
                continue

            # Check for chest/thorax anatomy
            has_chest = ('CHEST' in text_upper or 'THORAX' in text_upper)
            if not has_chest:
                continue

            # Check for report structure keywords (confirms it's a report, not an order)
            has_report_structure = (
                'IMPRESSION:' in text_upper or
                'FINDINGS:' in text_upper or
                'TECHNIQUE:' in text_upper
            )
            if not has_report_structure:
                continue

            # Exclude pathology reports (they mention CT in context but aren't radiology reports)
            is_pathology = (
                'PATHOLOGIST:' in text_upper or
                'SURGICAL PATHOLOGY REPORT' in text_upper or
                'SPECIMEN:' in text_upper
            )
            if is_pathology:
                continue

            # Prefer "Attending Note" (typical radiology report name) over other note types
            name = data.get('name', '')
            is_attending_note = 'Attending' in name

            # This is a chest CT report!
            ct_reports.append({
                'event_id': node_id,
                'date': data.get('date', ''),
                'timestamp': data.get('timestamp', ''),
                'name': name,
                'full_text': full_text,
                'is_attending_note': is_attending_note
            })

        if not ct_reports:
            return {'report': None, 'count': 0}

        # Get latest by timestamp, preferring "Attending Note" if multiple on same day
        # Sort by: (timestamp, is_attending_note)
        ct_reports_sorted = sorted(
            ct_reports,
            key=lambda r: (r['timestamp'] or r['date'], r['is_attending_note']),
            reverse=True
        )
        latest = ct_reports_sorted[0]

        return {
            'report': latest,
            'count': len(ct_reports)
        }

    def get_tnm_component_mentions(self) -> Dict[str, Any]:
        """
        Find latest mentions of T, N, M components separately.

        Returns:
            {
                'T': {'latest_event': {...}, 'count': ...},
                'N': {'latest_event': {...}, 'count': ...},
                'M': {'latest_event': {...}, 'count': ...}
            }
        """
        import re

        results = {}

        # Search for T component (T1, T2, T3, T4, Tis, Tx)
        t_matches = []
        n_matches = []
        m_matches = []

        for node_id, data in self.graph.G.nodes(data=True):
            if data.get('node_type') != 'Event':
                continue

            text_fields = [
                data.get('value', ''),
                data.get('name', ''),
                data.get('description', '')
            ]
            text = ' '.join(text_fields)

            # T component
            t_match = re.search(r'\bT[0-4is x]\b', text, re.IGNORECASE)
            if t_match:
                # Extract ±250 chars around match
                match_pos = t_match.start()
                start = max(0, match_pos - 250)
                end = min(len(text), match_pos + len(t_match.group()) + 250)
                snippet = text[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet = snippet + "..."

                t_matches.append({
                    'event_id': node_id,
                    'date': data.get('date', ''),
                    'timestamp': data.get('timestamp', ''),
                    'type': data.get('type'),
                    'visit_id': data.get('visit_id', ''),
                    'text': snippet
                })

            # N component
            n_match = re.search(r'\bN[0-3x]\b', text, re.IGNORECASE)
            if n_match:
                # Extract ±250 chars around match
                match_pos = n_match.start()
                start = max(0, match_pos - 250)
                end = min(len(text), match_pos + len(n_match.group()) + 250)
                snippet = text[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet = snippet + "..."

                n_matches.append({
                    'event_id': node_id,
                    'date': data.get('date', ''),
                    'timestamp': data.get('timestamp', ''),
                    'type': data.get('type'),
                    'visit_id': data.get('visit_id', ''),
                    'text': snippet
                })

            # M component
            m_match = re.search(r'\bM[01x]\b', text, re.IGNORECASE)
            if m_match:
                # Extract ±250 chars around match
                match_pos = m_match.start()
                start = max(0, match_pos - 250)
                end = min(len(text), match_pos + len(m_match.group()) + 250)
                snippet = text[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet = snippet + "..."

                m_matches.append({
                    'event_id': node_id,
                    'date': data.get('date', ''),
                    'timestamp': data.get('timestamp', ''),
                    'type': data.get('type'),
                    'visit_id': data.get('visit_id', ''),
                    'text': snippet
                })

        # Get latest of each
        if t_matches:
            results['T'] = {
                'latest_event': max(t_matches, key=lambda e: e['date']),
                'count': len(t_matches)
            }

        if n_matches:
            results['N'] = {
                'latest_event': max(n_matches, key=lambda e: e['date']),
                'count': len(n_matches)
            }

        if m_matches:
            results['M'] = {
                'latest_event': max(m_matches, key=lambda e: e['date']),
                'count': len(m_matches)
            }

        return results

    def get_smoking_status(self) -> Dict[str, Any]:
        """
        Find smoking/tobacco status mentions.

        Returns:
            {'latest_mention': {...}, 'count': ...}
        """
        keywords = ['smoking', 'tobacco', 'never smok', 'former smok', 'current smok']

        matches = []
        for kw in keywords:
            matches.extend(self._search_keyword(kw, context_chars=400))

        # Remove duplicates by event_id
        seen = set()
        unique_matches = []
        for m in matches:
            if m['event_id'] not in seen:
                seen.add(m['event_id'])
                unique_matches.append(m)

        if not unique_matches:
            return {'latest_mention': None, 'count': 0}

        latest = max(unique_matches, key=lambda e: e['date'])
        return {'latest_mention': latest, 'count': len(unique_matches)}

    def get_ecog_status(self) -> Dict[str, Any]:
        """
        Find ECOG performance status mentions.

        Returns:
            {'latest_mention': {...}, 'count': ...}
        """
        keywords = ['ecog', 'performance status']

        matches = []
        for kw in keywords:
            matches.extend(self._search_keyword(kw))

        # Remove duplicates
        seen = set()
        unique_matches = []
        for m in matches:
            if m['event_id'] not in seen:
                seen.add(m['event_id'])
                unique_matches.append(m)

        if not unique_matches:
            return {'latest_mention': None, 'count': 0}

        latest = max(unique_matches, key=lambda e: e['date'])
        return {'latest_mention': latest, 'count': len(unique_matches)}

    def get_code_status_mentions(self) -> Dict[str, Any]:
        """
        Find code-status / DNR / POLST / goals-of-care mentions.

        Crucial grounding for the `dnr` display field. Without it, the LLM
        guesses from stage-IV context and frequently inverts the patient's
        documented wishes (e.g., reads a chart phrase "DNR DNI given age" as
        evidence the patient IS DNR, when the same note records Full Code).

        Returns:
            {
                'mentions': [{'event_id', 'date', 'text'}, ...],  # all hits, latest first
                'count': int,
                'latest_mention': {...} or None
            }
        """
        # Case-sensitive word-boundary matches for ambiguous abbreviations
        cs_wb_keywords = ['DNR', 'DNI', 'POLST']
        # Case-insensitive phrase matches (unambiguous)
        ci_keywords = [
            'code status', 'full code', 'goals of care',
            'do not resuscitate', 'do-not-resuscitate',
            'advance directive', 'allow natural death',
            'comfort measures only', 'CMO',
        ]

        matches = []
        for kw in cs_wb_keywords:
            matches.extend(self._search_keyword(
                kw, context_chars=300, case_sensitive=True, word_boundary=True
            ))
        for kw in ci_keywords:
            matches.extend(self._search_keyword(kw, context_chars=300))

        # Dedup by event_id, keep the longest text snippet
        by_event: Dict[str, Dict[str, Any]] = {}
        for m in matches:
            eid = m['event_id']
            if eid not in by_event or len(m.get('text', '')) > len(by_event[eid].get('text', '')):
                by_event[eid] = m
        unique = list(by_event.values())
        unique.sort(key=lambda e: e.get('date', ''), reverse=True)

        if not unique:
            return {'mentions': [], 'count': 0, 'latest_mention': None}
        return {
            'mentions': unique[:8],
            'count': len(unique),
            'latest_mention': unique[0],
        }

    def get_all_allergy_mentions(self) -> Dict[str, Any]:
        """
        Find ALL allergy mentions (not just latest).

        For notes: extract sentence containing "allerg" (text between '. ')
        For observations/conditions: keep full value

        Returns:
            {
                'all_mentions': [
                    {'event_id': '...', 'date': '...', 'type': '...', 'text': '...'},
                    ...
                ],
                'count': ...
            }
        """
        keyword = 'allerg'  # Catches allergy, allergies, allergic

        matches = self._search_keyword(keyword)

        # Extract sentences for note types
        processed_matches = []
        for match in matches:
            if match['type'] == 'note':
                # Extract sentence containing "allerg"
                sentence = self._extract_sentence(match['text'], keyword)
                processed_matches.append({
                    'event_id': match['event_id'],
                    'date': match['date'],
                    'type': match['type'],
                    'text': sentence
                })
            else:
                # Keep full value for observations/conditions
                processed_matches.append(match)

        return {
            'all_mentions': processed_matches,
            'count': len(processed_matches)
        }

    def _extract_sentence(self, text: str, keyword: str) -> str:
        """
        Extract sentence containing keyword from text.

        Args:
            text: Full text
            keyword: Keyword to find

        Returns:
            Sentence containing keyword (text between '. ')
        """
        # Split by sentences (. followed by space or end)
        import re
        sentences = re.split(r'\.\s+', text)

        # Find sentence containing keyword
        keyword_lower = keyword.lower()
        for sentence in sentences:
            if keyword_lower in sentence.lower():
                return sentence.strip() + '.'

        # Fallback: return first 200 chars if no sentence boundary found
        return text[:200]

    def get_distinct_conditions(self) -> Dict[str, Any]:
        """
        Get DISTINCT conditions with earliest date for each.

        Groups conditions by name/code and returns earliest mention of each.

        Returns:
            {
                'conditions': [
                    {'name': 'Lung adenocarcinoma', 'code': 'ICD10/...', 'earliest_date': '...', 'event_id': '...'},
                    ...
                ],
                'count': number of distinct conditions
            }
        """
        conditions = {}

        for node_id, data in self.graph.G.nodes(data=True):
            if data.get('node_type') != 'Event':
                continue

            if data.get('type') != 'condition':
                continue

            # Get condition identifier (name or code)
            name = data.get('name', '')
            code = data.get('code', '')

            # Use code as primary key, name as fallback
            condition_key = code if code else name

            if not condition_key:
                continue

            date = data.get('date', '')

            # Keep earliest mention of each condition
            if condition_key not in conditions:
                conditions[condition_key] = {
                    'name': name,
                    'code': code,
                    'earliest_date': date,
                    'event_id': node_id
                }
            else:
                # Update if this is earlier
                if date < conditions[condition_key]['earliest_date']:
                    conditions[condition_key] = {
                        'name': name,
                        'code': code,
                        'earliest_date': date,
                        'event_id': node_id
                    }

        # Convert to list sorted by date
        condition_list = sorted(
            conditions.values(),
            key=lambda c: c['earliest_date']
        )

        return {
            'conditions': condition_list,
            'count': len(condition_list)
        }

    def get_distinct_drug_exposures(self) -> Dict[str, Any]:
        """
        Get DISTINCT drug exposures with first and last date for each.

        Groups drugs by name/code and returns date range (first → last).

        Returns:
            {
                'drugs': [
                    {
                        'name': 'Carboplatin',
                        'code': 'RxNorm/...',
                        'first_date': 'YYYY-MM-DD',
                        'last_date': 'YYYY-MM-DD',
                        'mention_count': 5,
                        'event_ids': ['...', '...']
                    },
                    ...
                ],
                'count': number of distinct drugs
            }
        """
        drugs = {}

        for node_id, data in self.graph.G.nodes(data=True):
            if data.get('node_type') != 'Event':
                continue

            if data.get('type') != 'drug_exposure':
                continue

            # Get drug identifier (name or code)
            name = data.get('name', '')
            code = data.get('code', '')

            # Use code as primary key, name as fallback
            drug_key = code if code else name

            if not drug_key:
                continue

            date = data.get('date', '')

            # Track all mentions of each drug
            if drug_key not in drugs:
                drugs[drug_key] = {
                    'name': name,
                    'code': code,
                    'first_date': date,
                    'last_date': date,
                    'mention_count': 1,
                    'event_ids': [node_id]
                }
            else:
                # Update date range and count
                if date < drugs[drug_key]['first_date']:
                    drugs[drug_key]['first_date'] = date
                if date > drugs[drug_key]['last_date']:
                    drugs[drug_key]['last_date'] = date

                drugs[drug_key]['mention_count'] += 1
                drugs[drug_key]['event_ids'].append(node_id)

        # Convert to list sorted by first date
        drug_list = sorted(
            drugs.values(),
            key=lambda d: d['first_date']
        )

        return {
            'drugs': drug_list,
            'count': len(drug_list)
        }

    def get_recent_imaging_studies(self, days: int = 365) -> Dict[str, Any]:
        """
        Get recent imaging studies (last N days) grouped by modality.

        Returns metadata only (modality, site, date) - NO full report text.

        Args:
            days: Look back period (default: 365 days = 1 year)

        Returns:
            {
                'CT': [
                    {'date': '2024-01-15', 'site': 'chest', 'event_id': '...'},
                    {'date': '2024-02-20', 'site': 'abdomen/pelvis', 'event_id': '...'},
                ],
                'MRI': [...],
                'PET': [...],
                'XR': [...],
                'total_studies': 15
            }
        """
        from datetime import datetime, timedelta

        # Calculate cutoff date (today - N days)
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        # Imaging modalities to search
        modalities = ['CT', 'MRI', 'PET', 'PET-CT', 'PET/CT', 'XR', 'X-RAY', 'ULTRASOUND']

        results = {}
        total_count = 0

        for node_id, data in self.graph.G.nodes(data=True):
            if data.get('node_type') != 'Event':
                continue

            # Check if this is imaging (procedure or image type)
            event_type = data.get('type')
            if event_type not in ['procedure', 'image']:
                continue

            name = data.get('name', '').upper()
            date = data.get('date', '')

            # Filter by date (only recent)
            if date < cutoff:
                continue

            # Check which modality using word-boundary matching
            # (bare "CT" substring matches "COLLECTION", "ACTIVITY", etc.)
            code = data.get('code', '').upper()
            matched_modality = self._match_imaging_modality(name, code)

            if matched_modality:
                # Extract anatomic site from name
                site = self._extract_anatomic_site(name)

                study = {
                    'date': date,
                    'site': site,
                    'event_id': node_id,
                    'name': data.get('name', '')[:100]  # Short name for reference
                }

                if matched_modality not in results:
                    results[matched_modality] = []

                results[matched_modality].append(study)
                total_count += 1

        # Sort each modality by date (most recent first)
        for modality in results:
            results[modality] = sorted(results[modality], key=lambda s: s['date'], reverse=True)

        results['total_studies'] = total_count

        return results

    @staticmethod
    def _match_imaging_modality(name: str, code: str = "") -> str:
        """Match imaging modality from procedure name/code with word-boundary safety.

        Avoids false positives like "CT" matching "COLLECTION", "ACTIVITY", "IMPACT".
        Uses the same logic as graph_serializer._detect_imaging_modality().

        Args:
            name: Procedure name (uppercase)
            code: Procedure code (uppercase), e.g., "CPT4/78815"

        Returns:
            Matched modality string (e.g., "CT", "PET-CT", "MRI", "XR") or None.
        """
        # Import and reuse the serializer's proven detection logic
        from toa.graph_serializer import _detect_imaging_modality
        modality = _detect_imaging_modality(name, code)
        if not modality:
            return None
        # Normalize modality names
        if modality in ('PET-CT',):
            return 'PET-CT'
        if modality in ('XR',):
            return 'XR'
        if modality in ('US',):
            return 'ULTRASOUND'
        if modality in ('NM',):
            return 'NM'
        return modality  # CT, MRI, PET

    def _extract_anatomic_site(self, imaging_name: str) -> str:
        """
        Extract anatomic site from imaging procedure name.

        Args:
            imaging_name: Procedure name (e.g., "CT CHEST W IV CONTRAST")

        Returns:
            Anatomic site (e.g., "chest")
        """
        name_upper = imaging_name.upper()

        # Common sites
        sites = [
            'CHEST', 'ABDOMEN', 'PELVIS', 'HEAD', 'BRAIN', 'NECK',
            'SPINE', 'LUMBAR', 'THORACIC', 'CERVICAL',
            'EXTREMITY', 'SHOULDER', 'KNEE', 'HIP'
        ]

        matched_sites = []
        for site in sites:
            if site in name_upper:
                matched_sites.append(site.lower())

        # Combine if multiple (e.g., "abdomen/pelvis")
        if matched_sites:
            return '/'.join(matched_sites)

        return 'unspecified'

    def get_radiation_therapy_mentions(self) -> Dict[str, Any]:
        """
        Find radiation therapy mentions in notes and radiology reports.

        Searches for: radiation, radiotherapy, RT, SBRT, IMRT, proton,
        chemoradiation, XRT, brachytherapy, Gy, stereotactic.

        Returns:
            {
                'latest_note': {'event_id': '...', 'date': '...', 'text': '...'},
                'latest_radiology': {'event_id': '...', 'date': '...', 'text': '...'},
                'count': number of matches
            }
        """
        keywords = [
            'radiation', 'radiotherapy', 'SBRT', 'IMRT', 'proton therapy',
            'chemoradiation', 'XRT', 'brachytherapy', 'stereotactic',
        ]

        matches = []
        for kw in keywords:
            matches.extend(self._search_keyword(kw))

        # Remove duplicates by event_id
        seen = set()
        unique_matches = []
        for m in matches:
            if m['event_id'] not in seen:
                seen.add(m['event_id'])
                unique_matches.append(m)

        if not unique_matches:
            return {
                'latest_note': None,
                'latest_radiology': None,
                'count': 0
            }

        # Separate by type
        notes = [m for m in unique_matches if m['type'] == 'note']
        radiology_keywords = ['ct', 'mri', 'pet', 'xr', 'x-ray', 'ultrasound', 'imaging', 'radiology']
        radiology = [m for m in unique_matches if any(kw in m.get('name', '').lower() for kw in radiology_keywords)]

        # Get latest of each
        latest_note = max(notes, key=lambda e: e['date']) if notes else None
        latest_rad = max(radiology, key=lambda e: e['date']) if radiology else None

        return {
            'latest_note': latest_note,
            'latest_radiology': latest_rad,
            'count': len(unique_matches)
        }

    def get_surgical_history_mentions(self) -> Dict[str, Any]:
        """
        Find surgical procedure mentions in notes and procedure records.

        Searches for: *tomy (lobectomy, pneumonectomy, craniotomy, nephrectomy, etc.),
        *ectomy, resection, VATS, thoracotomy, Whipple, operative report, and
        specific procedure types.

        Returns:
            {
                'latest_note': {'event_id': '...', 'date': '...', 'text': '...'},
                'procedures': [{'event_id': '...', 'date': '...', 'text': '...'}],
                'count': number of matches
            }
        """
        import re

        # Standard keywords (searchable via _search_keyword)
        keywords = [
            'resection', 'VATS', 'thoracotomy', 'Whipple',
            'operative report', 'surgical', 'lumpectomy',
        ]

        matches = []
        for kw in keywords:
            matches.extend(self._search_keyword(kw))

        # Regex pattern for *tomy/*ectomy suffix (catches lobectomy, pneumonectomy,
        # craniotomy, nephrectomy, mastectomy, etc. but not "anatomy", "phlebotomy")
        # Require word boundary after the suffix to avoid partial matches
        tomy_pattern = re.compile(
            r'\b\w{3,}(?:ectomy|otomy)\b', re.IGNORECASE
        )

        for node_id, data in self.graph.G.nodes(data=True):
            if data.get('node_type') != 'Event':
                continue

            text_fields = [
                data.get('value', ''),
                data.get('name', ''),
                data.get('description', '')
            ]
            text = ' '.join(text_fields)

            m = tomy_pattern.search(text)
            if m:
                # Skip common non-surgical *tomy words
                matched_word = m.group(0).lower()
                if matched_word in ('anatomy', 'phlebotomy', 'tracheostomy'):
                    continue

                match_pos = m.start()
                start = max(0, match_pos - 300)
                end = min(len(text), m.end() + 300)
                snippet = text[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet = snippet + "..."

                matches.append({
                    'event_id': node_id,
                    'date': data.get('date', ''),
                    'timestamp': data.get('timestamp', ''),
                    'type': data.get('type'),
                    'name': data.get('name', ''),
                    'visit_id': data.get('visit_id', ''),
                    'text': snippet,
                })

        # Deduplicate by event_id
        seen = set()
        unique_matches = []
        for m in matches:
            if m['event_id'] not in seen:
                seen.add(m['event_id'])
                unique_matches.append(m)

        if not unique_matches:
            return {
                'latest_note': None,
                'procedures': [],
                'count': 0
            }

        # Separate notes from procedure-type events
        notes = [m for m in unique_matches if m['type'] == 'note']
        procedures = [m for m in unique_matches if m['type'] in ('procedure', 'condition')]

        # Sort by date descending
        notes.sort(key=lambda e: e['date'], reverse=True)
        procedures.sort(key=lambda e: e['date'], reverse=True)

        latest_note = notes[0] if notes else None

        return {
            'latest_note': latest_note,
            'procedures': procedures[:5],
            'count': len(unique_matches)
        }

    def get_toxicity_comorbidity_mentions(self) -> Dict[str, Any]:
        """
        Find therapy toxicity and serious comorbidity mentions.

        Searches for toxicities (grade 3+, specific adverse events) and
        major comorbidities (COPD, CHF, diabetes, renal, cardiac, etc.).

        Returns:
            {
                'toxicities': [{'event_id': '...', 'date': '...', 'text': '...'}],
                'comorbidities': [{'event_id': '...', 'date': '...', 'text': '...'}],
                'count': total matches
            }
        """
        toxicity_keywords = [
            'toxicity', 'adverse event', 'grade 3', 'grade 4', 'grade 5',
            'neutropenia', 'thrombocytopenia', 'pneumonitis',
            'hepatotoxicity', 'neuropathy',
        ]
        comorbidity_keywords = [
            'COPD', 'CHF', 'heart failure', 'diabetes',
            'renal insufficiency', 'renal failure', 'chronic kidney',
            'cirrhosis', 'hepatic failure',
            'pulmonary embolism', 'deep vein thrombosis',
        ]

        # Gather toxicity matches
        tox_matches = []
        for kw in toxicity_keywords:
            tox_matches.extend(self._search_keyword(kw))
        # Deduplicate
        seen = set()
        unique_tox = []
        for m in tox_matches:
            if m['event_id'] not in seen:
                seen.add(m['event_id'])
                unique_tox.append(m)

        # Gather comorbidity matches
        comorbidity_matches = []
        for kw in comorbidity_keywords:
            comorbidity_matches.extend(self._search_keyword(kw))
        # Deduplicate
        seen2 = set()
        unique_comorbidity = []
        for m in comorbidity_matches:
            if m['event_id'] not in seen2:
                seen2.add(m['event_id'])
                unique_comorbidity.append(m)

        # Sort each by date descending
        unique_tox.sort(key=lambda e: e['date'], reverse=True)
        unique_comorbidity.sort(key=lambda e: e['date'], reverse=True)

        return {
            'toxicities': unique_tox[:10],  # Limit to 10 most recent
            'comorbidities': unique_comorbidity[:10],
            'count': len(unique_tox) + len(unique_comorbidity)
        }

    def get_ct_date_vector(self) -> List[str]:
        """
        All CT/PET-CT study dates, sorted descending.

        Extracted deterministically from structured imaging data (procedures/images).
        Used to ground LLM imaging event dates to real structured data.

        Deduplication: multiple procedure records for the same study (e.g.,
        scan-date Stanford proc + billing-date CPT4 charge) are clustered
        within a 7-day window and the EARLIEST date is kept. This ensures
        we use the actual scan date, not the billing/posting date — critical
        for correct temporal ordering relative to other clinical events.

        Returns:
            List of date strings (YYYY-MM-DD), most recent first.
        """
        from datetime import datetime, timedelta

        imaging = self.get_recent_imaging_studies(days=99999)

        # Collect entries per modality, then dedup per modality
        # (CT on June 24 and PET-CT on June 25 are DIFFERENT studies)
        all_entries = []
        for modality in ['CT', 'PET-CT', 'PET/CT']:
            for study in imaging.get(modality, []):
                if study.get('date'):
                    all_entries.append({
                        'date': study['date'],
                        'modality': modality,
                        'site': study.get('site', ''),
                    })

        if not all_entries:
            return []

        # Dedup per modality (7-day window: billing vs scan date)
        deduped = self._dedup_imaging_entries(all_entries, window_days=7)
        dates = sorted(set(e['date'] for e in deduped), reverse=True)
        return dates

    def get_imaging_date_vector(self) -> List[dict]:
        """
        All imaging study dates with modality, sorted descending.

        Returns structured records from ALL imaging modalities (CT, MRI, PET-CT, XR, etc.)
        for grounding LLM-extracted imaging events to exact institutional dates.

        Sources:
        1. Structured procedure/image events (from get_recent_imaging_studies)
        2. Radiology report notes — PET-CT and MRI reports are often stored as notes
           with exam dates embedded in the text (e.g., "EXAM DATE: 03/16/2022",
           "WHOLE-BODY F18-FDG PET-CT: 10/19/2016")

        Returns:
            List of {date, modality, site} dicts, most recent first.
        """
        import re

        # Source 1: structured procedure/image events
        imaging = self.get_recent_imaging_studies(days=99999)
        entries = []
        seen_dates_modality = set()  # Dedup across sources

        for modality, studies in imaging.items():
            if modality == 'total_studies':
                continue
            for study in studies:
                if study.get('date'):
                    key = (study['date'], modality)
                    if key not in seen_dates_modality:
                        seen_dates_modality.add(key)
                        entries.append({
                            'date': study['date'],
                            'modality': modality,
                            'site': study.get('site', ''),
                        })

        # Source 2: radiology report notes (catches PET-CT, MRI not in procedures)
        # Patterns for exam dates in radiology reports:
        #   "WHOLE-BODY F18-FDG PET-CT: MM/DD/YYYY"
        #   "EXAM DATE: MM/DD/YYYY"
        #   "Nuclear Medicine FDG-PET/CT ... EXAM DATE: MM/DD/YYYY"
        #   "MRI BRAIN ... MM/DD/YYYY"
        exam_date_patterns = [
            # "EXAM DATE: MM/DD/YYYY" or "EXAM DATE: M/D/YYYY"
            re.compile(r'EXAM\s+DATE:\s*(\d{1,2})/(\d{1,2})/(\d{4})'),
            # "PET-CT: MM/DD/YYYY" or "PET/CT: MM/DD/YYYY"
            re.compile(r'PET[/-]CT:\s*(\d{1,2})/(\d{1,2})/(\d{4})'),
            # "F18-FDG PET-CT: MM/DD/YYYY"
            re.compile(r'FDG[- ]PET[/-]CT:\s*(\d{1,2})/(\d{1,2})/(\d{4})'),
        ]
        # Modality detection in note text
        modality_patterns = [
            (re.compile(r'PET[/-]?CT|FDG[- ]PET', re.IGNORECASE), 'PET-CT'),
            (re.compile(r'\bMRI\b', re.IGNORECASE), 'MRI'),
        ]

        for node_id, data in self.graph.G.nodes(data=True):
            if data.get('node_type') != 'Event' or data.get('type') != 'note':
                continue
            val = data.get('value') or ''
            if len(val) < 20:
                continue
            val_upper = val.upper()

            # Only process notes that ARE radiology reports (not clinical notes mentioning imaging)
            # Radiology reports typically start with the study type or have it very early
            # Skip: boilerplate contact info ("RADIOLOGY 999-999-9999 (CT, PET/CT, MRI)")
            # Skip: clinical notes that just reference imaging ("recommend PET-CT", "as seen on PET/CT")
            has_imaging_keyword = False
            for kw in ['PET-CT', 'PET/CT', 'FDG-PET', 'FDG PET',
                        'WHOLE-BODY', 'NUCLEAR MEDICINE',
                        'MRI BRAIN', 'MRI SPINE', 'MRI ABDOMEN']:
                pos = val_upper.find(kw)
                if pos < 0:
                    continue
                # Skip if keyword is inside boilerplate contact block
                context_around = val_upper[max(0, pos-60):pos+60]
                if '999-999' in context_around or '(CT,' in context_around:
                    continue
                # Keyword must indicate this note IS a radiology report for that modality,
                # not a clinical note that merely references the study
                # Accept if: (1) keyword in first 1500 chars AND note has report structure
                # (FINDINGS, IMPRESSION, or TECHNIQUE), or (2) keyword is a report header
                # (followed by date, "whole-body", "skull", etc.)
                if pos < 1500:
                    # Check if note has radiology report structure
                    has_report_structure = any(
                        marker in val_upper[:3000]
                        for marker in ['FINDINGS:', 'IMPRESSION:', 'TECHNIQUE:',
                                       'COMPARISON:', 'HISTORY:', 'EXAM DATE:']
                    )
                    if has_report_structure:
                        has_imaging_keyword = True
                        break
                # Also accept if keyword is a clear report header
                after_kw = val_upper[pos:pos+80]
                if re.search(r'(?:WHOLE|SKULL|BODY|TUMOR|LOCALIZATION|\d{1,2}/\d{1,2}/\d{4})', after_kw):
                    has_imaging_keyword = True
                    break
            if not has_imaging_keyword:
                continue

            # Try to extract exam date
            exam_date = None
            for pattern in exam_date_patterns:
                m = pattern.search(val_upper)
                if m:
                    month, day, year = m.group(1), m.group(2), m.group(3)
                    try:
                        exam_date = f"{year}-{int(month):02d}-{int(day):02d}"
                    except ValueError:
                        continue
                    break

            # If no explicit exam date found, use the note date
            if not exam_date:
                exam_date = data.get('date', '')

            if not exam_date:
                continue

            # Determine modality from note content
            detected_modality = None
            for mod_pattern, mod_name in modality_patterns:
                if mod_pattern.search(val[:500]):
                    detected_modality = mod_name
                    break

            if not detected_modality:
                # Wider search — some reports bury the modality deeper in text
                for mod_pattern, mod_name in modality_patterns:
                    if mod_pattern.search(val[:2000]):
                        detected_modality = mod_name
                        break

            if not detected_modality:
                continue

            # Extract site
            site = self._extract_anatomic_site(val_upper[:500])

            key = (exam_date, detected_modality)
            if key not in seen_dates_modality:
                seen_dates_modality.add(key)
                entries.append({
                    'date': exam_date,
                    'modality': detected_modality,
                    'site': site,
                })

        # Deduplicate billing vs scan dates: cluster same-modality entries
        # within 7 days and keep the earliest date (actual scan date).
        entries = self._dedup_imaging_entries(entries)

        return sorted(entries, key=lambda e: e['date'], reverse=True)

    @staticmethod
    def _dedup_imaging_entries(entries: List[dict], window_days: int = 7) -> List[dict]:
        """Cluster same-modality imaging entries within a time window, keep earliest.

        Billing/CPT records are often posted days after the actual scan.
        This clusters entries of the same modality within `window_days`
        and keeps only the earliest date per cluster (the actual scan date).

        Args:
            entries: List of {date, modality, site} dicts
            window_days: Max days between entries to consider same study

        Returns:
            Deduplicated list of imaging entries.
        """
        from datetime import datetime, timedelta

        if not entries:
            return entries

        # Group by modality
        by_modality = {}
        for e in entries:
            mod = e.get('modality', 'UNKNOWN')
            by_modality.setdefault(mod, []).append(e)

        deduped = []
        for modality, mod_entries in by_modality.items():
            # Sort ascending by date
            mod_entries.sort(key=lambda x: x['date'])

            clusters = [[mod_entries[0]]]
            for e in mod_entries[1:]:
                try:
                    prev_date = datetime.strptime(clusters[-1][-1]['date'], "%Y-%m-%d")
                    curr_date = datetime.strptime(e['date'], "%Y-%m-%d")
                    if (curr_date - prev_date).days <= window_days:
                        clusters[-1].append(e)
                    else:
                        clusters.append([e])
                except ValueError:
                    clusters.append([e])

            # Keep earliest entry from each cluster
            for cluster in clusters:
                deduped.append(cluster[0])

        return deduped

    def get_comprehensive_context(self, imaging_lookback_days: int = 365) -> Dict[str, Any]:
        """
        Get all deterministic search hits for LLM context enhancement.

        Args:
            imaging_lookback_days: How far back to look for imaging studies (default: 365 = 1 year)

        Returns dictionary with:
        - metastasis_mentions
        - lymph_node_mentions
        - driver_mutation_mentions
        - tnm_mentions
        - smoking_status
        - ecog_status
        - radiation_therapy
        - toxicity_comorbidities
        - allergy_mentions (ALL, not just latest)
        - distinct_conditions (earliest of each)
        - distinct_drug_exposures (first → last date for each)
        - recent_imaging (metadata only, no full reports)
        - retrieval_timestamp
        """
        return {
            'metastasis': self.get_latest_metastasis_mentions(),
            'lymph_nodes': self.get_latest_lymph_node_mentions(),
            'driver_mutations': self.get_driver_mutation_mentions(),
            'molecular_testing': self.get_molecular_testing_mentions(),
            'latest_oncology_note': self.get_latest_oncology_note(),
            'latest_chest_ct': self.get_latest_chest_ct_report(),
            # 'tnm': self.get_tnm_component_mentions(),  # Disabled - too many false positives
            'smoking': self.get_smoking_status(),
            'ecog': self.get_ecog_status(),
            'radiation_therapy': self.get_radiation_therapy_mentions(),
            'toxicity_comorbidities': self.get_toxicity_comorbidity_mentions(),
            'allergies': self.get_all_allergy_mentions(),
            'conditions': self.get_distinct_conditions(),
            'drug_exposures': self.get_distinct_drug_exposures(),
            'surgical_history': self.get_surgical_history_mentions(),
            'code_status': self.get_code_status_mentions(),
            'recent_imaging': self.get_recent_imaging_studies(days=imaging_lookback_days),
            'retrieval_timestamp': datetime.now().isoformat()
        }


# ==========================================================================
# Convenience Functions
# ==========================================================================

def extract_deterministic_context(graph: TOAGraph) -> Dict[str, Any]:
    """
    One-line extraction of deterministic clinical context.

    Args:
        graph: TOAGraph for patient

    Returns:
        Dictionary with all deterministic search hits

    Example:
        >>> graph = build_hierarchical_graph("patient_123")
        >>> context = extract_deterministic_context(graph)
        >>> print(context['metastasis']['count'])
        >>> # 5
    """
    retriever = DeterministicRetriever(graph)
    return retriever.get_comprehensive_context()
