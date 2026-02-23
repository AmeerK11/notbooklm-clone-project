"""
Text-to-Speech adapter supporting multiple providers.
"""
import os
from pathlib import Path
from typing import Literal, Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

# TTS Provider type
TTSProvider = Literal["openai", "elevenlabs", "edge"]


class TTSAdapter(ABC):
    """Base class for TTS providers."""
    
    @abstractmethod
    def synthesize(self, text: str, output_path: str, voice: Optional[str] = None) -> str:
        """
        Convert text to speech.
        
        Args:
            text: Text to synthesize
            output_path: Where to save audio file
            voice: Voice identifier (provider-specific)
        
        Returns:
            Path to generated audio file
        """
        pass


class OpenAITTS(TTSAdapter):
    """OpenAI TTS (good quality, moderate cost)."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        import openai
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("TTS_MODEL", "tts-1")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.default_voice = os.getenv("TTS_OPENAI_VOICE_1", "alloy")
    
    def synthesize(self, text: str, output_path: str, voice: Optional[str] = None) -> str:
        """
        Voices: alloy, echo, fable, onyx, nova, shimmer
        """
        voice = voice or self.default_voice
        
        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice,
            input=text
        )

        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path


class ElevenLabsTTS(TTSAdapter):
    """ElevenLabs TTS (highest quality, paid)."""
    
    def __init__(self, api_key: Optional[str] = None):
        from elevenlabs.client import ElevenLabs
        
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.client = ElevenLabs(api_key=self.api_key)
        self.default_voice = os.getenv("TTS_ELEVENLABS_VOICE_1", "Rachel")
    
    def synthesize(self, text: str, output_path: str, voice: Optional[str] = None) -> str:
        """
        Popular voices: Rachel, Domi, Bella, Antoni, Elli, Josh, Arnold, Adam, Sam
        """
        voice = voice or self.default_voice
        
        audio = self.client.generate(
            text=text,
            voice=voice,
            model="eleven_monolingual_v1"
        )
        
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        
        return output_path


class EdgeTTS(TTSAdapter):
    """Microsoft Edge TTS (free, good quality)."""
    
    def __init__(self):
        self.default_voice = os.getenv("TTS_EDGE_VOICE_1", "en-US-GuyNeural")
    
    def synthesize(self, text: str, output_path: str, voice: Optional[str] = None) -> str:
        """
        Popular voices:
        - en-US-AriaNeural (female)
        - en-US-GuyNeural (male)
        - en-GB-SoniaNeural (British female)
        - en-GB-RyanNeural (British male)
        """
        import edge_tts
        import asyncio
        
        voice = voice or self.default_voice
        
        async def _synthesize():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
        
        asyncio.run(_synthesize())
        return output_path


def get_tts_adapter(provider: Optional[TTSProvider] = None) -> TTSAdapter:
    """
    Factory function to get TTS adapter.
    
    Args:
        provider: TTS provider to use (defaults to TTS_PROVIDER from .env)
    
    Returns:
        Configured TTS adapter instance
    """
    provider = provider or os.getenv("TTS_PROVIDER", "edge")
    
    adapters = {
        "openai": OpenAITTS,
        "elevenlabs": ElevenLabsTTS,
        "edge": EdgeTTS,
    }
    
    if provider not in adapters:
        raise ValueError(f"Unknown TTS provider: {provider}. Choose from: {list(adapters.keys())}")
    
    return adapters[provider]()


# === CLI for testing ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TTS")
    parser.add_argument("--provider", choices=["openai", "elevenlabs", "edge"], help="TTS provider")
    parser.add_argument("--text", default="Hello, this is a test of text to speech synthesis.")
    parser.add_argument("--output", default="test_audio.mp3")
    parser.add_argument("--voice", help="Voice ID (provider-specific)")
    
    args = parser.parse_args()
    
    tts = get_tts_adapter(args.provider)
    output_file = tts.synthesize(args.text, args.output, args.voice)
    
    print(f"✓ Audio generated: {output_file}")
    print(f"  Provider: {args.provider or os.getenv('TTS_PROVIDER', 'edge')}")
    print(f"  Voice: {args.voice or 'default'}")