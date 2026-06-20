
"""
TOA: Timeline-Object Augmentation library for the VISTA Tumor Board Dashboard.

Public API:
- extract(pid: str, xml_text: str, chat_client: Optional[Callable], profile: Optional[str|Profile]) -> tuple[list[Event], list[Episode]]
- project_major_events(events: list[Event], episodes: list[Episode]) -> dict
- Profile, load_profile, list_profiles: Profile management
"""
from .engine import extract, project_major_events
from .model import Event, Episode
from .profiles import Profile, load_profile, list_profiles
