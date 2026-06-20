"""
Profile management for TOA timeline extraction.

Profiles define tumor-type-specific extraction logic, chunking strategies,
and variable targets for clinical timeline generation.
"""

from __future__ import annotations
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class Profile:
    """Represents a clinical timeline extraction profile."""

    def __init__(self, profile_dict: Dict[str, Any]):
        self.data = profile_dict
        self.metadata = profile_dict.get("profile_metadata", {})
        self.name = self.metadata.get("name", "unknown")
        self.version = self.metadata.get("version", "1.0")
        self.tumor_type = self.metadata.get("tumor_type", "")

    @property
    def prompt_template(self) -> str:
        """Get the full prompt template from this profile."""
        return self.data.get("prompt_template", "")

    @property
    def chunking_config(self) -> Dict[str, Any]:
        """Get chunking configuration."""
        return self.data.get("chunking_config", {
            "max_chunk_size": 500000,
            "split_boundary": "entry",
            "min_final_chunk_ratio": 0.5,
            "strategy": "full"  # "full" or "streamlined"
        })

    @property
    def chunking_strategy(self) -> str:
        """Get chunking strategy: 'full' or 'streamlined'."""
        return self.chunking_config.get("strategy", "full")

    def assemble_prompt(self,
                       current_timeline: str = "",
                       current_summary: str = "",
                       xml_chunk: str = "",
                       is_final_chunk: bool = False,
                       is_streamlined: bool = False,
                       tumor_type: Optional[str] = None) -> str:
        """
        Assemble the full system prompt with template variables filled in.

        Args:
            current_timeline: JSON string of events from previous chunks
            current_summary: JSON string of minimal summary from previous chunks
            xml_chunk: Current XML chunk to process
            is_final_chunk: Whether this is the final chunk
            is_streamlined: Whether this chunk has been filtered to notes/procedures/conditions only
            tumor_type: Override tumor type (defaults to profile's tumor_type)

        Returns:
            Fully assembled prompt ready for LLM
        """
        template = self.prompt_template

        # Add streamlined note if applicable (for early chunks with filtered data)
        if is_streamlined and not is_final_chunk:
            streamlined_note = """
NOTE: This is an EARLY CHUNK that has been pre-filtered to contain only:
- <event type="note"> (clinical notes, progress notes, consult notes, discharge summaries)
- <event type="procedure"> (surgical procedures, biopsies, imaging studies)
- <event type="condition"> (diagnoses, pathology results)

Extract timeline events from these filtered elements using the same extraction rules below.
Most critical events (treatments, imaging results, surgeries) are documented in clinical notes.

"""
            template = streamlined_note + template

        # Replace template variables
        replacements = {
            "{TUMOR_TYPE}": tumor_type or self.tumor_type,
            "{current_timeline}": current_timeline,
            "{current_summary}": current_summary,
            "{xml_chunk}": xml_chunk,
            "{is_final_chunk}": str(is_final_chunk).lower()
        }

        assembled = template
        for placeholder, value in replacements.items():
            assembled = assembled.replace(placeholder, value)

        return assembled


def load_profile(name: str, profiles_dir: Optional[str] = None) -> Profile:
    """
    Load a profile by name.

    Args:
        name: Profile name (e.g., "thoracic_tumor_board")
        profiles_dir: Optional custom profiles directory path

    Returns:
        Loaded Profile object

    Raises:
        FileNotFoundError: If profile file doesn't exist
        ValueError: If profile JSON is invalid
    """
    if profiles_dir is None:
        # Default to toa/profiles/ directory
        profiles_dir = Path(__file__).parent / "profiles"
    else:
        profiles_dir = Path(profiles_dir)

    # Support both "name" and "name.json"
    profile_path = profiles_dir / f"{name}.json" if not name.endswith(".json") else profiles_dir / name

    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_dict = json.load(f)
        return Profile(profile_dict)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in profile {profile_path}: {e}")


def list_profiles(profiles_dir: Optional[str] = None) -> list[str]:
    """
    List all available profile names.

    Args:
        profiles_dir: Optional custom profiles directory path

    Returns:
        List of profile names (without .json extension)
    """
    if profiles_dir is None:
        profiles_dir = Path(__file__).parent / "profiles"
    else:
        profiles_dir = Path(profiles_dir)

    if not profiles_dir.exists():
        return []

    return [
        p.stem for p in profiles_dir.glob("*.json")
        if p.is_file()
    ]
