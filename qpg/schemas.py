"""Structured outputs. The LLM must return these shapes, so every step is checkable in code."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

QuestionType = Literal["mcq", "short", "long", "numerical", "true_false", "fill_blank"]


class SectionPlan(BaseModel):
    title: str = Field(description="Section name, e.g. 'Section A - MCQs'")
    question_type: QuestionType
    num_questions: int = Field(ge=1, le=40)
    marks_each: float = Field(gt=0, le=100)


class Plan(BaseModel):
    """Planner output: what to cover and the exact paper blueprint."""
    topics: List[str] = Field(description="3-6 distinct, specific topics from the document")
    sections: List[SectionPlan] = Field(description="Blueprint that follows the user's format")


class TopicGrade(BaseModel):
    topic: str
    supported: bool = Field(description="True if the context has enough material to set questions")
    better_query: Optional[str] = Field(
        default=None, description="If not supported, a rewritten search query to find better material")


class ContextGrade(BaseModel):
    grades: List[TopicGrade]


class Question(BaseModel):
    text: str
    marks: float
    topic: str
    options: List[str] = Field(default_factory=list, description="Only for MCQs: 4 options")
    source_pages: List[int] = Field(default_factory=list,
                                    description="Page numbers from the context this question is based on")


class Section(BaseModel):
    title: str
    question_type: QuestionType
    questions: List[Question]


class QuestionPaper(BaseModel):
    title: str
    instructions: str
    sections: List[Section]


class Critique(BaseModel):
    difficulty_ok: bool = Field(description="Questions match the requested difficulty")
    grounded_ok: bool = Field(description="Every question can be answered from the context")
    clarity_ok: bool = Field(description="Questions are clear and unambiguous")
    feedback: str = Field(description="Specific fixes needed; empty if everything is fine")


class Answer(BaseModel):
    question_number: str = Field(description="e.g. 'A1', 'B3'")
    answer: str
    source_pages: List[int] = Field(default_factory=list)


class AnswerKey(BaseModel):
    answers: List[Answer]
