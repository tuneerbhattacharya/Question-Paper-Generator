"""Eval metrics. Code-based metrics are deterministic; the judge metric uses an LLM with a rubric."""
from typing import Dict, List

from pydantic import BaseModel, Field

from qpg.guardrails import question_numbers
from qpg.nodes import paper_text, structured
from qpg.schemas import AnswerKey, QuestionPaper


def format_pass(paper: QuestionPaper, format_issues: List[str],
                expected_questions: int, expected_marks: float) -> bool:
    total_q = sum(len(s.questions) for s in paper.sections)
    total_m = sum(q.marks for s in paper.sections for q in s.questions)
    return not format_issues and total_q == expected_questions and abs(total_m - expected_marks) < 1e-6


def must_cover_rate(paper: QuestionPaper, must_cover: List[str]) -> float:
    text = paper_text(paper).lower()
    return sum(k.lower() in text for k in must_cover) / len(must_cover) if must_cover else 1.0


def citation_validity(key: AnswerKey, evidence: Dict[str, List[dict]]) -> float:
    """Share of answers whose cited pages were all actually retrieved (and non-empty)."""
    pages = {c["page"] for chunks in evidence.values() for c in chunks}
    ok = [bool(a.source_pages) and set(a.source_pages) <= pages for a in key.answers]
    return sum(ok) / len(ok) if ok else 0.0


class JudgeScores(BaseModel):
    reason: str = Field(description="Brief reasoning, written before the scores")
    groundedness: int = Field(ge=1, le=5, description="5 = every answer fully supported by the document")
    correctness: int = Field(ge=1, le=5, description="5 = every answer correctly answers its question")
    difficulty_match: int = Field(ge=1, le=5, description="5 = questions exactly match requested difficulty")


JUDGE_PROMPT = """You are grading an AI-generated exam paper and answer key against the source document.
Use the rubric strictly. Explain your reasoning briefly in "reason" BEFORE deciding scores.

<document>
{document}
</document>

Requested difficulty: {difficulty}

Question paper:
{paper}

Answer key:
{answers}
"""


def judge(judge_llm, document_text: str, paper: QuestionPaper, key: AnswerKey,
          difficulty: str) -> JudgeScores:
    answers = "\n".join(f"{a.question_number}. {a.answer}" for a in key.answers)
    return structured(judge_llm, JudgeScores, JUDGE_PROMPT.format(
        document=document_text, difficulty=difficulty, paper=paper_text(paper), answers=answers))


def answer_alignment(paper: QuestionPaper, key: AnswerKey) -> float:
    expected = set(question_numbers(paper))
    got = {a.question_number.strip().upper() for a in key.answers}
    return len(expected & got) / len(expected) if expected else 0.0
