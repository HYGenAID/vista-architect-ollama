"""
XML filtering utilities for streamlined chunk processing.

Filters XML chunks to only include specific event types (notes, procedures, conditions)
for early chunks, while preserving structure for LLM extraction.
"""

from __future__ import annotations
import re
from typing import Set


def filter_xml_events(xml_text: str, keep_event_types: Set[str]) -> str:
    """
    Filter XML to only keep <event> elements with specified type attributes.

    This is a lightweight regex-based filter that preserves XML structure
    while removing unwanted event elements. Used for streamlined chunking
    where early chunks only process notes, procedures, and conditions.

    Args:
        xml_text: Raw XML text
        keep_event_types: Set of event type values to keep (e.g., {"note", "procedure", "condition"})

    Returns:
        Filtered XML text with only specified event types

    Example:
        >>> xml = '<entry><event type="note">Text</event><event type="lab">Value</event></entry>'
        >>> filter_xml_events(xml, {"note"})
        '<entry><event type="note">Text</event></entry>'
    """
    if not keep_event_types:
        return xml_text

    # Pattern to match <event ...> ... </event> blocks
    # Captures: opening tag with attributes, content, closing tag
    event_pattern = r'(<event\s+[^>]*type\s*=\s*["\'](\w+)["\'][^>]*>)(.*?)(</event>)'

    def keep_event(match):
        opening_tag = match.group(1)
        event_type = match.group(2)
        content = match.group(3)
        closing_tag = match.group(4)

        if event_type in keep_event_types:
            return opening_tag + content + closing_tag
        else:
            return ""  # Remove this event

    # Apply filter
    filtered = re.sub(event_pattern, keep_event, xml_text, flags=re.DOTALL)

    # Clean up empty entries (entries with no events left)
    # This helps reduce token usage
    filtered = re.sub(r'<entry[^>]*>\s*</entry>', '', filtered, flags=re.DOTALL)

    return filtered


def get_streamlined_chunk(xml_chunk: str) -> str:
    """
    Apply streamlined filtering for early chunks: keep only notes, procedures, conditions.

    Args:
        xml_chunk: Raw XML chunk

    Returns:
        Filtered XML with only notes, procedures, and conditions
    """
    return filter_xml_events(xml_chunk, {"note", "procedure", "condition"})
