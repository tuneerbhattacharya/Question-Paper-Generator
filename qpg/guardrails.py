"""Guardrails in plain Python: fast, free and testable.

Input  : check the PDF and the user's instructions before any LLM call.
Format : check the generated paper matches the planned blueprint.
Output : check the answer key is complete and cites retrieved pages.
"""
import re
from typing import Dict, List, Tuple

from qpg import config
from qpg.schemas import AnswerKey, Plan, QuestionPaper

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard (the|your|all) (rules|instructions|guidelines)",
    r"(reveal|print|show) (your|the) (system )?prompt",
    r"you are now\b",
    r"act as (an? )?(unrestricted|jailbroken)",
    r"api[_ ]?key",
]


# ---------------------------------------------------------------- input
def check_pdf(size_bytes: int, num_pages: int, extracted_chars: int) -> List[str]:
    problems = []
    if size_bytes > config.MAX_PDF_MB * 1024 * 1024:
        problems.append(f"PDF is larger than {config.MAX_PDF_MB} MB.")
    if num_pages > config.MAX_PDF_PAGES:
        problems.append(f"PDF has more than {config.MAX_PDF_PAGES} pages.")
    if extracted_chars < config.MIN_EXTRACTED_CHARS:
        problems.append("Very little text could be extracted. The PDF may be scanned images; "
                        "please upload a text-based PDF.")
    return problems


def check_user_text(instructions: str, format_text: str) -> List[str]:
    problems = []
    for label, text in [("Requirements", instructions), ("Format", format_text)]:
        if not text or not text.strip():
            problems.append(f"{label} is empty.")
            continue
        if len(text) > config.MAX_INSTRUCTION_CHARS:
            problems.append(f"{label} is too long (max {config.MAX_INSTRUCTION_CHARS} characters).")
        lowered = text.lower()
        if any(re.search(p, lowered) for p in INJECTION_PATTERNS):
            problems.append(f"{label} contains text that looks like a prompt-injection attempt.")
    return problems


def check_plan(plan: Plan) -> List[str]:
    problems = []
    total = sum(s.num_questions for s in plan.sections)
    if total > config.MAX_TOTAL_QUESTIONS:
        problems.append(f"Requested {total} questions; the limit is {config.MAX_TOTAL_QUESTIONS}.")
    if not plan.topics:
        problems.append("No topics could be identified in the document.")
    return problems


# ---------------------------------------------------------------- format
def check_paper_format(paper: QuestionPaper, plan: Plan) -> List[str]:
    """Deterministic checks: section count, question counts, marks, MCQ options, duplicates."""
    issues = []
    if len(paper.sections) != len(plan.sections):
        issues.append(f"Expected {len(plan.sections)} sections, got {len(paper.sections)}.")
    for planned, actual in zip(plan.sections, paper.sections):
        if len(actual.questions) != planned.num_questions:
            issues.append(f"'{planned.title}': expected {planned.num_questions} questions, "
                          f"got {len(actual.questions)}.")
        for i, q in enumerate(actual.questions, 1):
            if abs(q.marks - planned.marks_each) > 1e-6:
                issues.append(f"'{planned.title}' Q{i}: marks should be {planned.marks_each:g}, "
                              f"got {q.marks:g}.")
            if planned.question_type == "mcq" and len(q.options) != 4:
                issues.append(f"'{planned.title}' Q{i}: an MCQ needs exactly 4 options.")
    seen = {}
    for s in paper.sections:
        for q in s.questions:
            key = re.sub(r"[^a-z0-9 ]", "", q.text.lower()).strip()
            if key in seen:
                issues.append(f"Duplicate question: '{q.text[:60]}'")
            seen[key] = True
    return issues


def topic_coverage(paper: QuestionPaper, topics: List[str]) -> Tuple[float, List[str]]:
    """Share of planned topics that at least one question is tagged with."""
    used = {q.topic.strip().lower() for s in paper.sections for q in s.questions}
    missing = [t for t in topics if t.strip().lower() not in used]
    return (1 - len(missing) / len(topics)) if topics else 0.0, missing


# ---------------------------------------------------------------- output
def question_numbers(paper: QuestionPaper) -> List[str]:
    """A1, A2, ... B1 ... so questions and answers line up."""
    numbers = []
    for s_idx, s in enumerate(paper.sections):
        letter = chr(ord("A") + s_idx)
        numbers += [f"{letter}{i}" for i in range(1, len(s.questions) + 1)]
    return numbers


def check_answer_key(key: AnswerKey, paper: QuestionPaper,
                     evidence: Dict[str, List[dict]]) -> List[str]:
    issues = []
    expected = question_numbers(paper)
    got = [a.question_number.strip().upper() for a in key.answers]
    missing = [n for n in expected if n not in got]
    if missing:
        issues.append(f"Answer key is missing: {', '.join(missing)}")
    retrieved_pages = {c["page"] for chunks in evidence.values() for c in chunks}
    for a in key.answers:
        if not a.source_pages:
            issues.append(f"{a.question_number}: no source page cited.")
        elif not set(a.source_pages) <= retrieved_pages:
            bad = sorted(set(a.source_pages) - retrieved_pages)
            issues.append(f"{a.question_number}: cites page(s) {bad} that were never retrieved "
                          f"(possible hallucination).")
    return issues
