"""Agent steps (LangGraph nodes). Each takes the state and returns only the fields it changes.

Dependencies are injected, so every node can be unit-tested with a fake LLM and fake search.
"""
from typing import Callable, Dict, List

from qpg import config, guardrails, prompts
from qpg.schemas import AnswerKey, ContextGrade, Critique, Plan, QuestionPaper
from qpg.state import QPState

SearchFn = Callable[[str, int], List[dict]]   # (query, k) -> [{"text": str, "page": int}]


# ---------------------------------------------------------------- helpers
def structured(model, schema, prompt: str, attempts: int = 2):
    """Call the model with a required output schema; retry once if validation fails."""
    runnable = model.with_structured_output(schema, method="function_calling")
    for attempt in range(attempts):
        try:
            return runnable.invoke(prompt)
        except Exception:
            if attempt == attempts - 1:
                raise


def blueprint_text(plan: Plan) -> str:
    return "\n".join(
        f"- {s.title}: {s.num_questions} x {s.question_type} question(s), {s.marks_each:g} marks each"
        for s in plan.sections)


def build_context(evidence: Dict[str, List[dict]], topics: List[str],
                  max_chars: int = config.MAX_CONTEXT_CHARS) -> str:
    """Group evidence by topic, label pages, de-duplicate, and enforce the token budget."""
    blocks, seen, used = [], set(), 0
    for topic in topics:
        parts = []
        for chunk in evidence.get(topic, []):
            if chunk["text"] in seen:
                continue
            piece = f"[page {chunk['page']}] {chunk['text'].strip()}"
            if used + len(piece) > max_chars:
                break
            seen.add(chunk["text"])
            parts.append(piece)
            used += len(piece)
        if parts:
            blocks.append(f'<topic name="{topic}">\n' + "\n".join(parts) + "\n</topic>")
    return "\n".join(blocks)


def paper_text(paper: QuestionPaper) -> str:
    lines = [paper.title, paper.instructions]
    numbers = iter(guardrails.question_numbers(paper))
    for s in paper.sections:
        lines.append(f"\n{s.title}")
        for q in s.questions:
            lines.append(f"{next(numbers)}. {q.text} [{q.marks:g} marks]")
            lines += [f"   ({chr(97 + i)}) {o}" for i, o in enumerate(q.options)]
    return "\n".join(lines)


# ---------------------------------------------------------------- nodes
class Nodes:
    def __init__(self, llm, judge_llm, search: SearchFn):
        self.llm, self.judge, self.search = llm, judge_llm, search

    # 1. Input guardrail -------------------------------------------------
    def validate_input(self, state: QPState) -> dict:
        problems = guardrails.check_user_text(state["instructions"], state["format_text"])
        if problems:
            return {"blocked": True, "block_reason": " ".join(problems),
                    "trace": [f"Input guardrail: BLOCKED ({'; '.join(problems)})"]}
        return {"blocked": False, "trace": ["Input guardrail: passed"]}

    # 2. Planner -----------------------------------------------------------
    def plan(self, state: QPState) -> dict:
        plan: Plan = structured(self.llm, Plan, prompts.PLAN_PROMPT.format(
            doc_sample=state["doc_sample"], instructions=state["instructions"],
            format_text=state["format_text"], difficulty=state["difficulty"],
            max_topics=config.MAX_TOPICS, safety=prompts.SAFETY_RULES))
        plan.topics = list(dict.fromkeys(t.strip() for t in plan.topics if t.strip()))[:config.MAX_TOPICS]
        problems = guardrails.check_plan(plan)
        if problems:
            return {"plan": plan, "blocked": True, "block_reason": " ".join(problems),
                    "trace": [f"Planner: BLOCKED ({'; '.join(problems)})"]}
        return {"plan": plan, "queries": {t: t for t in plan.topics}, "evidence": {},
                "weak_topics": [], "retrieval_attempts": 0, "revisions": 0, "feedback": "",
                "trace": [f"Planner: {len(plan.topics)} topics, {len(plan.sections)} sections "
                          f"({sum(s.num_questions for s in plan.sections)} questions)"]}

    # 3. Retriever ---------------------------------------------------------
    def retrieve(self, state: QPState) -> dict:
        first_pass = state.get("retrieval_attempts", 0) == 0
        targets = state["plan"].topics if first_pass else state["weak_topics"]
        evidence = dict(state.get("evidence", {}))
        for topic in targets:
            evidence[topic] = self.search(state["queries"][topic], config.CHUNKS_PER_TOPIC)
        label = "all topics" if first_pass else f"re-searching {targets}"
        return {"evidence": evidence, "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
                "trace": [f"Retriever: {label}"]}

    # 4. Context grader (corrective RAG) -------------------------------------
    def grade_context(self, state: QPState) -> dict:
        evidence_text = "\n\n".join(
            f"TOPIC: {t}\n" + "\n".join(f"[page {c['page']}] {c['text'][:600]}" for c in chunks)
            for t, chunks in state["evidence"].items())
        result: ContextGrade = structured(
            self.judge, ContextGrade, prompts.GRADE_PROMPT.format(evidence=evidence_text, safety=prompts.SAFETY_RULES))
        weak, queries = [], dict(state["queries"])
        by_name = {t.lower(): t for t in queries}       # tolerate case differences from the LLM
        for g in result.grades:
            topic = by_name.get(g.topic.strip().lower())
            if topic and not g.supported:
                weak.append(topic)
                if g.better_query:
                    queries[topic] = g.better_query
        msg = f"Grader: weak topics {weak}" if weak else "Grader: all topics well supported"
        return {"weak_topics": weak, "queries": queries, "trace": [msg]}

    # 5. Generator (also handles revisions) ------------------------------------
    def generate(self, state: QPState) -> dict:
        plan = state["plan"]
        out_of_retries = state.get("retrieval_attempts", 0) > config.MAX_RETRIEVAL_RETRIES
        topics = [t for t in plan.topics if not (out_of_retries and t in state.get("weak_topics", []))]
        topics = topics or plan.topics
        feedback = state.get("feedback", "")
        paper: QuestionPaper = structured(
            self.llm, QuestionPaper, prompts.GENERATE_PROMPT.format(
                context=build_context(state["evidence"], topics), instructions=state["instructions"],
                difficulty=state["difficulty"], blueprint=blueprint_text(plan), topics=topics,
                feedback=prompts.FEEDBACK_BLOCK.format(feedback=feedback) if feedback else "",
                safety=prompts.SAFETY_RULES))
        issues = guardrails.check_paper_format(paper, plan)
        step = "Generator: revised draft" if feedback else "Generator: first draft"
        return {"paper": paper, "format_issues": issues,
                "trace": [f"{step} ({len(issues)} format issue(s))"]}

    # 6. Critic (evaluator-optimizer) -------------------------------------------
    def critique(self, state: QPState) -> dict:
        plan, paper = state["plan"], state["paper"]
        review: Critique = structured(
            self.judge, Critique, prompts.CRITIQUE_PROMPT.format(
                context=build_context(state["evidence"], plan.topics), difficulty=state["difficulty"],
                paper=paper_text(paper), safety=prompts.SAFETY_RULES))
        coverage, missing = guardrails.topic_coverage(paper, plan.topics)
        problems = list(state.get("format_issues", []))
        if missing:
            problems.append(f"No questions on these topics: {missing}")
        if not (review.difficulty_ok and review.grounded_ok and review.clarity_ok) and review.feedback:
            problems.append(review.feedback)
        revisions = state.get("revisions", 0) + (1 if problems else 0)
        verdict = "approved" if not problems else f"{len(problems)} problem(s) found"
        return {"feedback": "\n".join(f"- {p}" for p in problems), "revisions": revisions,
                "trace": [f"Critic: {verdict} (topic coverage {coverage:.0%})"]}

    # 7. Answer key ----------------------------------------------------------------
    def answer_key(self, state: QPState) -> dict:
        paper = state["paper"]
        key: AnswerKey = structured(
            self.llm, AnswerKey, prompts.ANSWER_PROMPT.format(
                context=build_context(state["evidence"], state["plan"].topics),
                paper=paper_text(paper), numbers=", ".join(guardrails.question_numbers(paper)),
                safety=prompts.SAFETY_RULES))
        return {"answer_key": key, "trace": [f"Answer writer: {len(key.answers)} answers"]}

    # 8. Output guardrail ----------------------------------------------------------
    def check_output(self, state: QPState) -> dict:
        issues = guardrails.check_answer_key(state["answer_key"], state["paper"], state["evidence"])
        msg = "Output guardrail: passed" if not issues else f"Output guardrail: {len(issues)} warning(s)"
        return {"output_issues": issues, "trace": [msg]}


# ---------------------------------------------------------------- routers
def route_after_input(state: QPState) -> str:
    return "stop" if state.get("blocked") else "continue"


def route_after_grading(state: QPState) -> str:
    if state.get("weak_topics") and state.get("retrieval_attempts", 0) <= config.MAX_RETRIEVAL_RETRIES:
        return "retry"
    return "generate"


def route_after_critique(state: QPState) -> str:
    if state.get("feedback") and state.get("revisions", 0) <= config.MAX_REVISIONS:
        return "revise"
    return "answer"
