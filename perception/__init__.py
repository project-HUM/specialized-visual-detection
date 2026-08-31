"""Capture-local gameplay perception API.

The public functions deliberately keep profile construction separate from
inference.  A profile is tied to one resolution, map reconstruction, and
character appearance session.
"""

from .core import (
    FrameAnalyzer,
    SessionProfile,
    analyze_frame,
    analyze_video,
    build_session_profile,
)
from .specialized_detector import MonsterDetection, SpecializedMonsterDetector, TemplateMonsterDetector, YoloMonsterDetector

__all__ = [
    "FrameAnalyzer",
    "SessionProfile",
    "analyze_frame",
    "analyze_video",
    "build_session_profile",
    "MonsterDetection",
    "SpecializedMonsterDetector",
    "TemplateMonsterDetector",
    "YoloMonsterDetector",
]
