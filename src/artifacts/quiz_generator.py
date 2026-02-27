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

from src.ingestion.vectorstore import ChromaAdapter
from utils.llm_client import generate_chat_completion

load_dotenv()


class QuizGenerator:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize quiz generator.

        Args:
            api_key: Unused legacy arg (kept for compatibility)
            model: Display model name to record in metadata
        """
        _ = api_key
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        # Default settings from .env
        self.default_num_questions = int(os.getenv("DEFAULT_QUIZ_QUESTIONS", "5"))
        self.default_difficulty = os.getenv("DEFAULT_QUIZ_DIFFICULTY", "medium")

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

        # 3. Format and return
        return {
            "questions": quiz_data.get("questions", []),
            "metadata": {
                "notebook_id": notebook_id,
                "num_questions": num_questions,
                "difficulty": difficulty,
                "topic_focus": topic_focus,
                "model": self.model,
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
            raw_text = generate_chat_completion(
                system_prompt=(
                    "You are an expert quiz generator. "
                    "Create clear, educational, and well-structured quiz questions."
                ),
                user_prompt=prompt,
            )
            return self._parse_quiz_json(raw_text)

        except Exception as e:
            print(f"❌ Error generating quiz: {e}")
            return {"questions": []}

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

    def _parse_quiz_json(self, raw_text: str) -> Dict[str, Any]:
        text = (raw_text or "").strip()
        if not text:
            return {"questions": []}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: extract the largest JSON object from a markdown/codefence response.
        candidates = re.findall(r"\{[\s\S]*\}", text)
        for candidate in sorted(candidates, key=len, reverse=True):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return {"questions": []}

    def to_markdown(self, quiz_data: Dict[str, Any], title: str | None = None) -> str:
        title_text = title or "Quiz"
        questions = quiz_data.get("questions", [])
        metadata = quiz_data.get("metadata", {})
        difficulty = metadata.get("difficulty", "unknown")
        generated_at = metadata.get("generated_at", datetime.utcnow().isoformat())

        lines: list[str] = [
            f"# {title_text}",
            "",
            f"- Difficulty: {difficulty}",
            f"- Generated at: {generated_at}",
            "",
            "## Questions",
            "",
        ]

        for idx, q in enumerate(questions, start=1):
            question_text = str(q.get("question", "")).strip() or f"Question {idx}"
            lines.append(f"{idx}. {question_text}")
            options = q.get("options", [])
            if isinstance(options, list):
                for option in options:
                    lines.append(f"   - {str(option)}")
            lines.append("")

        lines.extend(["## Answer Key", ""])
        for idx, q in enumerate(questions, start=1):
            answer = str(q.get("correct_answer", "")).strip() or "N/A"
            explanation = str(q.get("explanation", "")).strip()
            lines.append(f"{idx}. **Answer:** {answer}")
            if explanation:
                lines.append(f"   - Explanation: {explanation}")
            topic = str(q.get("topic", "")).strip()
            if topic:
                lines.append(f"   - Topic: {topic}")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def save_quiz(self, markdown: str, user_id: str, notebook_id: str) -> str:
        """Save generated quiz markdown to file."""
        quiz_dir = Path(f"data/users/{user_id}/notebooks/{notebook_id}/artifacts/quizzes")
        quiz_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"quiz_{timestamp}.md"
        filepath = quiz_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown)

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
            generator.save_quiz(
                generator.to_markdown(quiz, title=f"Quiz ({args.difficulty or 'mixed'})"),
                args.user,
                args.notebook,
            )
