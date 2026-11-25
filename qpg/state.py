"""The shared state that flows through the LangGraph agent."""
import operator
from typing import Annotated, Dict, List, Optional, TypedDict

from qpg.schemas import AnswerKey, Plan, QuestionPaper


class QPState(TypedDict, total=False):
    # Inputs
    instructions: str
    format_text: str
    difficulty: str
    doc_sample: str                     # opening text of the document, for planning

    # Working memory
    plan: Optional[Plan]
    queries: Dict[str, str]             # topic -> search query
    evidence: Dict[str, List[dict]]     # topic -> [{"text", "page"}]
    weak_topics: List[str]
    retrieval_attempts: int
    paper: Optional[QuestionPaper]
    format_issues: List[str]
    feedback: str
    revisions: int
    answer_key: Optional[AnswerKey]
    output_issues: List[str]

    # Status
    blocked: bool
    block_reason: str
    trace: Annotated[List[str], operator.add]   # appended by every node (observability)
