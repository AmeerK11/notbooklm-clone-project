"""
Artifact generation module for NotebookLM clone.
Supports quiz, report, and podcast generation.
"""
from .report_generator import ReportGenerator
from .quiz_generator import QuizGenerator
from .podcast_generator import PodcastGenerator
from .tts_adapter import get_tts_adapter, TTSProvider

__all__ = [
    'ReportGenerator',
    'QuizGenerator',
    'PodcastGenerator',
    'get_tts_adapter',
    'TTSProvider'
]
