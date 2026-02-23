"""
Podcast generator - creates conversational audio from notebook content.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.ingestion.vectorstore import ChromaAdapter
from .tts_adapter import get_tts_adapter, TTSProvider

load_dotenv()


class PodcastGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        tts_provider: Optional[TTSProvider] = None,
    ):
        """
        Initialize podcast generator.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from .env)
            model: LLM model to use (defaults to LLM_MODEL from .env)
            tts_provider: TTS provider (defaults to TTS_PROVIDER from .env)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key)

        # TTS configuration
        tts_provider = tts_provider or os.getenv("TTS_PROVIDER", "edge")
        self.tts = get_tts_adapter(tts_provider)
        self.tts_provider = tts_provider

        # Default settings from .env
        self.default_duration = os.getenv("DEFAULT_PODCAST_DURATION", "5min")
        self.host_1 = os.getenv("PODCAST_HOST_1", "Alex")
        self.host_2 = os.getenv("PODCAST_HOST_2", "Jordan")

    def generate_podcast(
        self,
        user_id: str,
        notebook_id: str,
        duration_target: Optional[str] = None,
        hosts: Optional[List[str]] = None,
        topic_focus: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a podcast-style conversation from notebook content.

        Args:
            user_id: User identifier
            notebook_id: Notebook to generate podcast from
            duration_target: Target length ("5min", "10min", "15min")
            hosts: List of host names (defaults to PODCAST_HOST_1/2 from .env)
            topic_focus: Optional specific topic to focus on

        Returns:
            Dict with transcript, audio_path, and metadata
        """
        duration_target = duration_target or self.default_duration
        hosts = hosts or [self.host_1, self.host_2]

        print(f"🎙️  Generating {duration_target} podcast with {hosts[0]} & {hosts[1]}...")

        # 1. Retrieve comprehensive context
        context = self._get_notebook_context(user_id, notebook_id, topic_focus)

        if not context:
            return {
                "error": "No content found in notebook. Please ingest documents first.",
                "transcript": [],
                "audio_path": None,
                "metadata": {},
            }

        # 2. Generate conversational script
        print("🤖 Generating podcast script...")
        script = self._generate_script(context, duration_target, hosts)

        if not script:
            return {
                "error": "Failed to generate podcast script.",
                "transcript": [],
                "audio_path": None,
                "metadata": {},
            }

        # 3. Synthesize audio segments
        print(f"🎵 Synthesizing audio with {self.tts_provider}...")
        audio_segments = self._synthesize_segments(script, user_id, notebook_id, hosts)

        # 4. Combine audio
        print("🔗 Combining audio segments...")
        final_audio = self._combine_audio(audio_segments, user_id, notebook_id)

        return {
            "transcript": script,
            "audio_path": final_audio,
            "metadata": {
                "notebook_id": notebook_id,
                "duration_target": duration_target,
                "hosts": hosts,
                "tts_provider": self.tts_provider,
                "num_segments": len(script),
                "topic_focus": topic_focus,
                "generated_at": datetime.utcnow().isoformat(),
            },
        }

    def _get_notebook_context(
        self,
        user_id: str,
        notebook_id: str,
        topic_focus: Optional[str] = None,
    ) -> str:
        """Retrieve comprehensive context from notebook."""
        data_base = os.getenv("STORAGE_BASE_DIR", "data")
        chroma_dir = str(
            Path(data_base) / "users" / user_id / "notebooks" / notebook_id / "chroma"
        )

        if not Path(chroma_dir).exists():
            print(f"⚠️  Chroma directory not found: {chroma_dir}")
            return ""

        store = ChromaAdapter(persist_directory=chroma_dir)

        # Get diverse chunks for comprehensive coverage
        if topic_focus:
            sample_queries = [topic_focus]
        else:
            sample_queries = [
                "main topics and concepts",
                "key principles and ideas",
                "important details and facts",
                "conclusions and insights",
                "examples and applications",
            ]

        all_chunks: List[str] = []
        for query in sample_queries:
            try:
                results = store.query(user_id, notebook_id, query, top_k=5)
                for _, _, chunk_data in results:
                    all_chunks.append(chunk_data["document"])
            except Exception as e:
                print(f"⚠️  Error querying: {e}")
                continue

        if not all_chunks:
            return ""

        # Deduplicate and combine
        unique_chunks = list(set(all_chunks))
        context = "\n\n".join(unique_chunks[:15])  # Top 15 chunks

        print(f"✓ Retrieved {len(unique_chunks)} unique chunks ({len(context)} chars)")
        return context

    def _generate_script(
        self,
        context: str,
        duration: str,
        hosts: List[str],
    ) -> List[Dict[str, str]]:
        """Generate conversational script using LLM."""

        word_count_map = {
            "5min": 750,
            "10min": 1500,
            "15min": 2250,
            "20min": 3000,
        }
        target_words = word_count_map.get(duration, 750)

        prompt = self._build_podcast_prompt(context, target_words, hosts)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert podcast script writer. Create engaging, natural, educational conversations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                response_format={"type": "json_object"},
            )

            script_data = json.loads(response.choices[0].message.content)
            segments = script_data.get("segments", [])

            print(f"✓ Generated script with {len(segments)} segments")
            return segments

        except Exception as e:
            print(f"❌ Error generating script: {e}")
            return []

    def _build_podcast_prompt(
        self,
        context: str,
        target_words: int,
        hosts: List[str],
    ) -> str:
        """Build the podcast script generation prompt."""

        return f"""
Create a natural, engaging podcast conversation between {hosts[0]} and {hosts[1]} about the following content.

Content to discuss:
{context}

Requirements:
- Target length: approximately {target_words} words ({target_words // 150} minutes)
- Make it conversational, engaging, and educational
- {hosts[0]} is the curious host who asks insightful questions
- {hosts[1]} is the knowledgeable host who provides clear explanations
- Include natural reactions, follow-up questions, and transitions
- Break down complex topics into digestible explanations
- Use analogies and examples to clarify concepts
- Maintain an upbeat, friendly, and enthusiastic tone
- End with a brief summary and key takeaways

Structure:
1. Opening (introduce topic, set context)
2. Main discussion (explore key concepts)
3. Deep dive (detailed explanations with examples)
4. Closing (summary and takeaways)

Generate the script in this exact JSON format:
{{
    "segments": [
        {{
            "speaker": "{hosts[0]}",
            "text": "Welcome to today's episode! We're diving into..."
        }},
        {{
            "speaker": "{hosts[1]}",
            "text": "Thanks for having me! This is such a fascinating topic..."
        }}
    ]
}}

IMPORTANT:
- Each segment should be 1-3 sentences (natural speaking chunks)
- Alternate between speakers naturally
- Include pauses and transitions like "That's interesting..." or "Let me explain..."
- Make it sound like a real conversation, not a lecture
"""

    def _synthesize_segments(
        self,
        script: List[Dict[str, str]],
        user_id: str,
        notebook_id: str,
        hosts: List[str],
    ) -> List[str]:
        """Synthesize each script segment to audio."""

        audio_dir = Path(f"data/users/{user_id}/notebooks/{notebook_id}/artifacts/podcasts")
        audio_dir.mkdir(parents=True, exist_ok=True)

        voice_maps: Dict[str, Dict[str, str]] = {
            "openai": {
                hosts[0]: os.getenv("TTS_OPENAI_VOICE_1", "alloy"),
                hosts[1]: os.getenv("TTS_OPENAI_VOICE_2", "echo"),
            },
            "edge": {
                hosts[0]: os.getenv("TTS_EDGE_VOICE_1", "en-US-GuyNeural"),
                hosts[1]: os.getenv("TTS_EDGE_VOICE_2", "en-US-AriaNeural"),
            },
            "elevenlabs": {
                hosts[0]: os.getenv("TTS_ELEVENLABS_VOICE_1", "Antoni"),
                hosts[1]: os.getenv("TTS_ELEVENLABS_VOICE_2", "Rachel"),
            },
        }

        voices = voice_maps.get(self.tts_provider, voice_maps["edge"])

        audio_files: List[str] = []
        total = len(script)

        for i, segment in enumerate(script, 1):
            speaker = segment["speaker"]
            text = segment["text"]
            voice = voices.get(speaker, list(voices.values())[0])

            output_path = str(audio_dir / f"segment_{i:03d}_{speaker}.mp3")

            try:
                self.tts.synthesize(text, output_path, voice=voice)
                audio_files.append(output_path)
                print(f"  ✓ Segment {i}/{total}: {speaker}")
            except Exception as e:
                print(f"  ⚠️  Failed segment {i}: {e}")
                continue

        return audio_files

    def _combine_audio(
        self,
        audio_segments: List[str],
        user_id: str,
        notebook_id: str,
    ) -> str:
        """Combine audio segments into single file."""
        try:
            from pydub import AudioSegment
        except ImportError:
            print("⚠️  pydub not installed. Skipping audio combination.")
            print("   Install with: pip install pydub")
            return audio_segments[0] if audio_segments else ""

        if not audio_segments:
            return ""

        combined = AudioSegment.empty()

        for i, segment_path in enumerate(audio_segments, 1):
            try:
                audio = AudioSegment.from_file(segment_path)
                combined += audio
                combined += AudioSegment.silent(duration=500)  # 0.5s pause
            except Exception as e:
                print(f"  ⚠️  Error processing segment {i}: {e}")
                continue

        output_dir = Path(f"data/users/{user_id}/notebooks/{notebook_id}/artifacts/podcasts")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        final_path = str(output_dir / f"podcast_{timestamp}.mp3")

        combined.export(final_path, format="mp3")

        print(f"✓ Final podcast: {final_path}")
        print(f"  Duration: {len(combined) / 1000:.1f} seconds")

        return final_path

    def save_transcript(
        self,
        podcast_data: Dict[str, Any],
        user_id: str,
        notebook_id: str,
    ) -> str:
        """Save podcast transcript to file."""
        transcript_dir = Path(
            f"data/users/{user_id}/notebooks/{notebook_id}/artifacts/podcasts"
        )
        transcript_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{timestamp}.json"
        filepath = transcript_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(podcast_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Transcript saved to: {filepath}")
        return str(filepath)


# === CLI for testing ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate podcast from notebook")
    parser.add_argument("--user", required=True, help="User ID")
    parser.add_argument("--notebook", required=True, help="Notebook ID")
    parser.add_argument(
        "--duration",
        choices=["5min", "10min", "15min", "20min"],
        help="Target podcast duration",
    )
    parser.add_argument("--topic", help="Focus on specific topic")
    parser.add_argument(
        "--tts-provider",
        choices=["openai", "edge", "elevenlabs"],
        help="TTS provider (defaults to TTS_PROVIDER in .env)",
    )
    parser.add_argument("--save-transcript", action="store_true", help="Save transcript to file")

    args = parser.parse_args()

    generator = PodcastGenerator(tts_provider=args.tts_provider)
    result = generator.generate_podcast(
        args.user,
        args.notebook,
        args.duration,
        topic_focus=args.topic,
    )

    if "error" in result:
        print(f"\n❌ {result['error']}")
    else:
        print(f"\n✓ Podcast generated!")
        print(f"  Audio: {result['audio_path']}")
        print(f"  Segments: {len(result['transcript'])}")
        print(f"  Provider: {result['metadata']['tts_provider']}")

        if args.save_transcript:
            generator.save_transcript(result, args.user, args.notebook)
