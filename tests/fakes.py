"""Fake LLM and search so the whole agent can be tested without an API key."""
from qpg.schemas import (Answer, AnswerKey, ContextGrade, Critique, Plan, Question, QuestionPaper,
                         Section, SectionPlan, TopicGrade)


class FakeLLM:
    """Returns queued responses per output schema; the last one repeats."""

    def __init__(self, responses):
        self.responses = {k: list(v) for k, v in responses.items()}
        self.prompts = []

    def with_structured_output(self, schema, **_):
        llm = self

        class Runnable:
            def invoke(self, prompt):
                llm.prompts.append((schema.__name__, prompt))
                queue = llm.responses[schema]
                return queue.pop(0) if len(queue) > 1 else queue[0]
        return Runnable()


TOPICS = ["Process states", "CPU scheduling"]
PLAN = Plan(topics=TOPICS, sections=[
    SectionPlan(title="Section A - MCQs", question_type="mcq", num_questions=2, marks_each=1),
    SectionPlan(title="Section B - Short", question_type="short", num_questions=1, marks_each=3)])


def mcq(text, topic, pages=(1,)):
    return Question(text=text, marks=1, topic=topic, options=["a", "b", "c", "d"],
                    source_pages=list(pages))


GOOD_PAPER = QuestionPaper(title="OS Test", instructions="Answer all questions.", sections=[
    Section(title="Section A - MCQs", question_type="mcq", questions=[
        mcq("Which state follows 'ready' when a process is dispatched?", "Process states"),
        mcq("Which algorithm can cause the convoy effect?", "CPU scheduling", (2,))]),
    Section(title="Section B - Short", question_type="short", questions=[
        Question(text="Explain aging in priority scheduling.", marks=3, topic="CPU scheduling",
                 source_pages=[2])])])

# First draft: only 1 MCQ and no question on 'Process states' -> must trigger a revision.
BAD_PAPER = QuestionPaper(title="OS Test", instructions="Answer all questions.", sections=[
    Section(title="Section A - MCQs", question_type="mcq", questions=[
        mcq("Which algorithm can cause the convoy effect?", "CPU scheduling", (2,))]),
    Section(title="Section B - Short", question_type="short", questions=[
        Question(text="Explain aging in priority scheduling.", marks=3, topic="CPU scheduling",
                 source_pages=[2])])])

KEY = AnswerKey(answers=[Answer(question_number="A1", answer="Running", source_pages=[1]),
                         Answer(question_number="A2", answer="FCFS", source_pages=[2]),
                         Answer(question_number="B1", answer="Aging raises priority...",
                                source_pages=[2])])

OK_CRITIQUE = Critique(difficulty_ok=True, grounded_ok=True, clarity_ok=True, feedback="")


def fake_search(query, k):
    page = 1 if "process" in query.lower() else 2
    return [{"text": f"Material about {query} (page {page}).", "page": page}]


def make_llms(paper_drafts, grades):
    gen = FakeLLM({Plan: [PLAN], QuestionPaper: paper_drafts, AnswerKey: [KEY]})
    judge = FakeLLM({ContextGrade: grades, Critique: [OK_CRITIQUE]})
    return gen, judge


WEAK_THEN_OK = [
    ContextGrade(grades=[TopicGrade(topic="Process states", supported=False,
                                    better_query="process state transitions ready running"),
                         TopicGrade(topic="CPU scheduling", supported=True)]),
    ContextGrade(grades=[TopicGrade(topic="Process states", supported=True)])]
ALL_OK = [ContextGrade(grades=[TopicGrade(topic=t, supported=True) for t in TOPICS])]
ALWAYS_WEAK = [ContextGrade(grades=[TopicGrade(topic="Process states", supported=False,
                                               better_query="process lifecycle")])]
