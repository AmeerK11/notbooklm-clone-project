"""
Quiz generator using RAG context from ingested documents.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
import requests

from src.ingestion.vectorstore import ChromaAdapter

load_dotenv()

SUPPORTED_QUIZ_LLM_PROVIDERS = {"openai", "groq", "ollama"}
DEFAULT_QUIZ_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.1-8b-instant",
    "ollama": "qwen2.5:3b",
}
QUIZ_SYSTEM_PROMPT = (
    "You are an expert quiz generator. Create clear, educational, and well-structured "
    "multiple-choice questions. Return valid JSON only with a top-level 'questions' array."
)


class QuizGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ):
        """
        Initialize quiz generator.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from .env)
            model: LLM model to use (defaults to LLM_MODEL from .env)
            llm_provider: Quiz LLM provider (openai, groq, ollama)
        """
        self.llm_provider = (llm_provider or os.getenv("QUIZ_LLM_PROVIDER", "openai")).strip().lower()
        if self.llm_provider not in SUPPORTED_QUIZ_LLM_PROVIDERS:
            raise ValueError(
                f"Unsupported QUIZ_LLM_PROVIDER='{self.llm_provider}'. "
                f"Choose from: {sorted(SUPPORTED_QUIZ_LLM_PROVIDERS)}"
            )
        self.model = self._resolve_model_name(model)
        self._openai_client: OpenAI | None = None
        self._groq_client = None
        self._ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

        if self.llm_provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self._openai_client = OpenAI(api_key=self.api_key)
        elif self.llm_provider == "groq":
            from groq import Groq

            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY is required when QUIZ_LLM_PROVIDER=groq")
            self._groq_client = Groq(api_key=groq_api_key)
        else:
            self.api_key = None

        # Default settings from .env
        self.default_num_questions = int(os.getenv("DEFAULT_QUIZ_QUESTIONS", "5"))
        self.default_difficulty = os.getenv("DEFAULT_QUIZ_DIFFICULTY", "medium")

    def _resolve_model_name(self, explicit_model: Optional[str]) -> str:
        if explicit_model and explicit_model.strip():
            return explicit_model.strip()
        configured = os.getenv("QUIZ_LLM_MODEL", "").strip()
        if configured:
            return configured
        legacy = os.getenv("LLM_MODEL", "").strip()
        if legacy:
            return legacy
        return DEFAULT_QUIZ_MODELS.get(self.llm_provider, "gpt-4o-mini")

    def generate_quiz(
        self,
        user_id: str,
        notebook_id: str,
        num_questions: Optional[int] = None,
        difficulty: Optional[str] = None,
        topic_focus: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate quiz questions from notebook content.

        Args:
            user_id: User identifier
            notebook_id: Notebook to generate quiz from
            num_questions: Number of questions (defaults to DEFAULT_QUIZ_QUESTIONS)
            difficulty: "easy", "medium", or "hard" (defaults to DEFAULT_QUIZ_DIFFICULTY)
            topic_focus: Optional specific topic to focus on

        Returns:
            Dict with questions, answers, and metadata
        """
        num_questions = num_questions or self.default_num_questions
        difficulty = difficulty or self.default_difficulty

        print(f"🎯 Generating {num_questions} {difficulty} quiz questions...")

        # 1. Retrieve context from vector store
        context = self._get_quiz_context(user_id, notebook_id, topic_focus)

        if not context:
            return {
                "error": "No content found in notebook. Please ingest documents first.",
                "questions": [],
                "metadata": {},
            }

        # 2. Generate quiz using LLM
        print("🤖 Generating questions with LLM...")
        quiz_data = self._generate_with_llm(context, num_questions, difficulty)
        questions = quiz_data.get("questions", []) if isinstance(quiz_data, dict) else []
        if not questions:
            return {
                "error": "Failed to generate quiz questions from notebook context.",
                "questions": [],
                "metadata": {
                    "notebook_id": notebook_id,
                    "num_questions": num_questions,
                    "difficulty": difficulty,
                    "topic_focus": topic_focus,
                    "llm_provider": self.llm_provider,
                    "llm_model": self.model,
                    "generated_at": datetime.utcnow().isoformat(),
                },
            }

        # 3. Format and return
        return {
            "questions": questions,
            "metadata": {
                "notebook_id": notebook_id,
                "num_questions": num_questions,
                "difficulty": difficulty,
                "topic_focus": topic_focus,
                "llm_provider": self.llm_provider,
                "llm_model": self.model,
                "generated_at": datetime.utcnow().isoformat(),
            },
        }

    def _get_quiz_context(
        self,
        user_id: str,
        notebook_id: str,
        topic_focus: Optional[str] = None,
    ) -> str:
        """Retrieve relevant context from notebook."""
        data_base = os.getenv("STORAGE_BASE_DIR", "data")
        chroma_dir = str(
            Path(data_base) / "users" / user_id / "notebooks" / notebook_id / "chroma"
        )

        if not Path(chroma_dir).exists():
            print(f"⚠️  Chroma directory not found: {chroma_dir}")
            return ""

        store = ChromaAdapter(persist_directory=chroma_dir)

        # Get diverse chunks for quiz generation
        if topic_focus:
            sample_queries = [topic_focus]
        else:
            sample_queries = [
                "main concepts and definitions",
                "key principles and theories",
                "important facts and details",
                "examples and applications",
                "processes and mechanisms",
            ]

        all_chunks: List[str] = []
        for query in sample_queries:
            try:
                results = store.query(user_id, notebook_id, query, top_k=3)
                for _, _, chunk_data in results:
                    all_chunks.append(chunk_data["document"])
            except Exception as e:
                print(f"⚠️  Error querying: {e}")
                continue

        if not all_chunks:
            return ""

        # Deduplicate and combine
        unique_chunks = list(set(all_chunks))
        context = "\n\n".join(unique_chunks[:10])  # Use top 10 unique chunks

        print(f"✓ Retrieved {len(unique_chunks)} unique chunks ({len(context)} chars)")
        return context

    def _generate_with_llm(
        self,
        context: str,
        num_questions: int,
        difficulty: str,
    ) -> Dict[str, Any]:
        """Generate quiz using LLM."""
        prompt = self._build_quiz_prompt(context, num_questions, difficulty)

        try:
            raw_response = self._generate_quiz_json(prompt)
            payload = self._extract_json_object(raw_response)
            questions = payload.get("questions") if isinstance(payload, dict) else None
            if not isinstance(questions, list):
                return {"questions": []}
            return {
                "questions": self._normalize_questions(
                    questions=questions,
                    expected_count=num_questions,
                    difficulty=difficulty,
                )
            }

        except Exception as e:
            print(f"❌ Error generating quiz: {e}")
            return {"questions": []}

    def _generate_quiz_json(self, prompt: str) -> str:
        if self.llm_provider == "openai":
            assert self._openai_client is not None
            response = self._openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            return str(response.choices[0].message.content or "")

        if self.llm_provider == "groq":
            assert self._groq_client is not None
            response = self._groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            return str(response.choices[0].message.content or "")

        payload = {
            "model": self.model,
            "system": QUIZ_SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7},
        }
        response = requests.post(
            f"{self._ollama_base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        return str(body.get("response", ""))

    def _extract_json_object(self, raw_response: str) -> Dict[str, Any]:
        content = str(raw_response or "").strip()
        if not content:
            return {}

        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}

    def _normalize_questions(
        self,
        questions: List[Any],
        expected_count: int,
        difficulty: str,
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in questions:
            if not isinstance(item, dict):
                continue
            prompt = str(item.get("question", "")).strip()
            if not prompt:
                continue

            options = self._normalize_options(item.get("options"))
            if not options:
                continue

            answer = self._normalize_answer_letter(str(item.get("correct_answer", "")).strip())
            explanation = str(item.get("explanation", "")).strip()
            topic = str(item.get("topic", "")).strip()

            normalized.append(
                {
                    "id": len(normalized) + 1,
                    "question": prompt,
                    "options": options,
                    "correct_answer": answer or "N/A",
                    "explanation": explanation,
                    "difficulty": difficulty,
                    "topic": topic,
                }
            )
            if len(normalized) >= expected_count:
                break
        return normalized

    def _normalize_options(self, options_raw: Any) -> List[str]:
        """
        Normalize options into 2-6 labeled choices: A) ... B) ... etc.
        Accepts list, dict, or multiline string.
        """
        parsed: List[str] = []

        if isinstance(options_raw, list):
            parsed = [str(opt).strip() for opt in options_raw if str(opt).strip()]
        elif isinstance(options_raw, dict):
            for key in sorted(options_raw.keys()):
                value = str(options_raw.get(key, "")).strip()
                if not value:
                    continue
                parsed.append(f"{str(key).strip().upper()}) {value}")
        elif isinstance(options_raw, str):
            lines = [line.strip() for line in options_raw.replace("\r", "\n").split("\n")]
            parsed = [line for line in lines if line]

        if not parsed:
            return []

        cleaned: List[str] = []
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for idx, text in enumerate(parsed):
            body = self._option_text_only(text)
            if not body:
                continue
            label = letters[idx] if idx < len(letters) else str(idx + 1)
            cleaned.append(f"{label}) {body}")

        # Keep a reasonable range and ensure common quiz shape is preserved.
        return cleaned[:6]

    def _option_text_only(self, value: str) -> str:
        text = str(value or "").strip()
        # Strip prefixes like "A)", "A.", "(A)", "1)", "1."
        text = re.sub(r"^\(?[A-Z0-9]\)?[\.\):\-]\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _normalize_answer_letter(self, value: str) -> str:
        match = re.search(r"[A-Z]", value.upper())
        return match.group(0) if match else ""

    def _build_quiz_prompt(self, context: str, num_questions: int, difficulty: str) -> str:
        """Build the quiz generation prompt."""

        difficulty_guidelines = {
            "easy": "Focus on basic recall and understanding. Questions should test fundamental concepts.",
            "medium": "Mix recall with application. Questions should require understanding and basic analysis.",
            "hard": "Focus on analysis, synthesis, and application. Questions should require deep understanding.",
        }

        guideline = difficulty_guidelines.get(difficulty, difficulty_guidelines["medium"])

        return f"""
Based on the following content, generate {num_questions} multiple-choice quiz questions at {difficulty} difficulty level.

Content:
{context}

Difficulty Guidelines:
{guideline}

Generate questions in this exact JSON format:
{{
    "questions": [
        {{
            "id": 1,
            "question": "Clear question text here?",
            "options": [
                "A) First option",
                "B) Second option",
                "C) Third option",
                "D) Fourth option"
            ],
            "correct_answer": "A",
            "explanation": "Clear explanation of why this answer is correct and why others are wrong.",
            "difficulty": "{difficulty}",
            "topic": "Main topic this question covers"
        }}
    ]
}}

Requirements:
- Generate exactly {num_questions} questions
- All options must be plausible and relevant
- Correct answer must be clearly identifiable
- Explanations should be educational and thorough
- Questions should test {difficulty}-level understanding
- Vary question types (definition, application, analysis)
- Ensure questions are based solely on the provided content
"""

    def format_quiz_markdown(self, quiz_data: Dict[str, Any], title: str | None = None) -> str:
        """Render quiz questions and answer key as Markdown."""
        resolved_title = title or "Quiz"
        questions = quiz_data.get("questions", [])
        metadata = quiz_data.get("metadata", {}) if isinstance(quiz_data.get("metadata"), dict) else {}
        difficulty = metadata.get("difficulty")

        lines: list[str] = [f"# {resolved_title}", ""]
        if difficulty:
            lines.append(f"Difficulty: **{difficulty}**")
            lines.append("")
        lines.append("## Questions")
        lines.append("")

        for idx, question in enumerate(questions, 1):
            prompt = str(question.get("question", "")).strip()
            lines.append(f"### {idx}. {prompt or 'Question'}")
            options = question.get("options", [])
            for option in options if isinstance(options, list) else []:
                option_text = str(option).strip()
                if option_text:
                    lines.append(f"- {option_text}")
            lines.append("")

        lines.append("## Answer Key")
        lines.append("")
        for idx, question in enumerate(questions, 1):
            correct = str(question.get("correct_answer", "")).strip() or "N/A"
            explanation = str(question.get("explanation", "")).strip()
            lines.append(f"{idx}. **{correct}**")
            if explanation:
                lines.append(f"   - {explanation}")
        lines.append("")
        return "\n".join(lines)

    def save_quiz(self, quiz_markdown: str, user_id: str, notebook_id: str) -> str:
        """Save generated quiz Markdown to file."""
        quiz_dir = Path(f"data/users/{user_id}/notebooks/{notebook_id}/artifacts/quizzes")
        quiz_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"quiz_{timestamp}.md"
        filepath = quiz_dir / filename

        filepath.write_text(quiz_markdown, encoding="utf-8")

        print(f"✓ Quiz saved to: {filepath}")
        return str(filepath)


# === CLI for testing ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate quiz from notebook")
    parser.add_argument("--user", required=True, help="User ID")
    parser.add_argument("--notebook", required=True, help="Notebook ID")
    parser.add_argument("--num-questions", type=int, help="Number of questions")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="Difficulty level")
    parser.add_argument("--topic", help="Focus on specific topic")
    parser.add_argument("--save", action="store_true", help="Save quiz to file")

    args = parser.parse_args()

    generator = QuizGenerator()
    quiz = generator.generate_quiz(
        args.user,
        args.notebook,
        args.num_questions,
        args.difficulty,
        args.topic,
    )

    if "error" in quiz:
        print(f"\n❌ {quiz['error']}")
    else:
        print(f"\n✓ Generated {len(quiz['questions'])} questions")
        print(json.dumps(quiz, indent=2))

        if args.save:
            markdown = generator.format_quiz_markdown(quiz, title="Quiz")
            generator.save_quiz(markdown, args.user, args.notebook)
