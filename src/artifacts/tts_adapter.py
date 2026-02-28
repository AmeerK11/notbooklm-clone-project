"""
Text-to-Speech adapter supporting multiple providers.
"""
import os
from pathlib import Path
from typing import Any, Literal, Optional
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
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")
        self.client = ElevenLabs(api_key=self.api_key)
        self.default_voice = os.getenv("TTS_ELEVENLABS_VOICE_1", "Rachel")
        self.default_model = os.getenv("TTS_ELEVENLABS_MODEL", "eleven_multilingual_v2")
        self._voice_aliases = self._load_voice_aliases()

    def _load_voice_aliases(self) -> dict[str, str]:
        """Best-effort map of configured voice names to voice IDs."""
        try:
            response = self.client.voices.get_all()
            voices = getattr(response, "voices", response)
        except Exception:
            return {}

        aliases: dict[str, str] = {}
        for voice in voices or []:
            if isinstance(voice, dict):
                name = voice.get("name")
                voice_id = voice.get("voice_id")
            else:
                name = getattr(voice, "name", None)
                voice_id = getattr(voice, "voice_id", None)
            if name and voice_id:
                aliases[str(name).strip().lower()] = str(voice_id).strip()
        return aliases

    def _resolve_voice(self, voice: str) -> str:
        candidate = str(voice or "").strip()
        if not candidate:
            candidate = self.default_voice
        return self._voice_aliases.get(candidate.lower(), candidate)

    def _write_audio_output(self, audio: Any, output_path: str) -> None:
        """
        ElevenLabs SDK returns either bytes, file-like, or iterable chunks depending
        on version/options. Handle all supported shapes safely.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            if isinstance(audio, (bytes, bytearray)):
                f.write(bytes(audio))
                return

            if hasattr(audio, "read"):
                data = audio.read()
                if isinstance(data, str):
                    data = data.encode("utf-8")
                if not isinstance(data, (bytes, bytearray)):
                    raise TypeError("ElevenLabs returned unsupported file-like payload.")
                f.write(bytes(data))
                return

            wrote_any = False
            for chunk in audio:
                if chunk is None:
                    continue
                wrote_any = True
                if isinstance(chunk, int):
                    f.write(bytes([chunk]))
                elif isinstance(chunk, str):
                    f.write(chunk.encode("utf-8"))
                elif isinstance(chunk, (bytes, bytearray)):
                    f.write(bytes(chunk))
                else:
                    raise TypeError(f"Unsupported ElevenLabs audio chunk type: {type(chunk)!r}")

            if not wrote_any:
                raise RuntimeError("ElevenLabs returned an empty audio stream.")
    
    def synthesize(self, text: str, output_path: str, voice: Optional[str] = None) -> str:
        """
        Popular voices: Rachel, Domi, Bella, Antoni, Elli, Josh, Arnold, Adam, Sam
        """
        requested_voice = voice or self.default_voice
        resolved_voice = self._resolve_voice(requested_voice)
        voice_candidates = [resolved_voice]
        if requested_voice != resolved_voice:
            voice_candidates.append(requested_voice)

        model_candidates = [self.default_model]
        if self.default_model != "eleven_multilingual_v2":
            model_candidates.append("eleven_multilingual_v2")

        errors: list[str] = []
        for voice_candidate in voice_candidates:
            for model_candidate in model_candidates:
                try:
                    audio = self.client.generate(
                        text=text,
                        voice=voice_candidate,
                        model=model_candidate,
                    )
                    self._write_audio_output(audio, output_path)
                    return output_path
                except Exception as exc:
                    errors.append(
                        f"voice={voice_candidate}, model={model_candidate}: "
                        f"{type(exc).__name__}: {exc}"
                    )

        preview = " | ".join(errors[:3]) if errors else "unknown ElevenLabs error"
        raise RuntimeError(f"ElevenLabs synthesis failed. {preview}")


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
