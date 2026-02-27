"""
Artifact generation module for NotebookLM clone.
Supports quiz, report, and podcast generation.
"""
from .quiz_generator import QuizGenerator
from .report_generator import ReportGenerator
from .podcast_generator import PodcastGenerator
from .tts_adapter import get_tts_adapter, TTSProvider

__all__ = [
    "QuizGenerator",
    "ReportGenerator",
    "PodcastGenerator",
    "get_tts_adapter",
    "TTSProvider",
]
