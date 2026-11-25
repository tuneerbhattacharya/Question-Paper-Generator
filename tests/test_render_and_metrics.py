from evals import metrics
from qpg.nodes import build_context
from qpg.render import to_pdf
from tests.fakes import GOOD_PAPER, KEY


def test_pdf_renders_and_escapes_special_characters():
    paper = GOOD_PAPER.model_copy(deep=True)
    paper.sections[1].questions[0].text = "Compare FCFS & SJF when burst < quantum"
    pdf = to_pdf(paper, KEY)
    assert pdf.startswith(b"%PDF") and len(pdf) > 1000


def test_context_respects_token_budget_and_dedupes():
    evidence = {"A": [{"text": "x" * 500, "page": 1}] * 3, "B": [{"text": "y" * 500, "page": 2}]}
    ctx = build_context(evidence, ["A", "B"], max_chars=600)
    assert ctx.count("[page 1]") == 1          # duplicate chunk removed
    assert "[page 2]" not in ctx               # over budget, so dropped


def test_eval_metrics():
    assert metrics.format_pass(GOOD_PAPER, [], expected_questions=3, expected_marks=5)
    assert not metrics.format_pass(GOOD_PAPER, [], expected_questions=4, expected_marks=5)
    assert metrics.must_cover_rate(GOOD_PAPER, ["convoy", "aging", "paging"]) == 2 / 3
    evidence = {"t": [{"text": "a", "page": 1}, {"text": "b", "page": 2}]}
    assert metrics.citation_validity(KEY, evidence) == 1.0
    assert metrics.citation_validity(KEY, {"t": [{"text": "a", "page": 1}]}) == 1 / 3
    assert metrics.answer_alignment(GOOD_PAPER, KEY) == 1.0
