"""
Artifact generation module for NotebookLM clone.
Supports quiz, report, and podcast generation.
"""
from .quiz_generator import QuizGenerator
from .podcast_generator import PodcastGenerator
from .tts_adapter import get_tts_adapter, TTSProvider

__all__ = [
    'QuizGenerator',
    'PodcastGenerator',
    'get_tts_adapter',
    'TTSProvider'
]
