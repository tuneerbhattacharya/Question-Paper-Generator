"""Render the structured paper and answer key to a PDF."""
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from qpg.guardrails import question_numbers
from qpg.schemas import AnswerKey, QuestionPaper


def to_pdf(paper: QuestionPaper, key: AnswerKey | None = None) -> bytes:
    styles = getSampleStyleSheet()
    story = [Paragraph(escape(paper.title), styles["Title"]),
             Paragraph(escape(paper.instructions), styles["Italic"]), Spacer(1, 12)]
    numbers = iter(question_numbers(paper))
    for s in paper.sections:
        story += [Paragraph(escape(s.title), styles["Heading2"]), Spacer(1, 4)]
        for q in s.questions:
            story.append(Paragraph(f"<b>{next(numbers)}.</b> {escape(q.text)} "
                                   f"<i>[{q.marks:g} marks]</i>", styles["Normal"]))
            for i, opt in enumerate(q.options):
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;({chr(97 + i)}) {escape(opt)}",
                                       styles["Normal"]))
            story.append(Spacer(1, 8))
    if key:
        story += [PageBreak(), Paragraph("Answer Key", styles["Title"])]
        for a in key.answers:
            pages = f" <i>(source: p. {', '.join(map(str, a.source_pages))})</i>" if a.source_pages else ""
            story += [Paragraph(f"<b>{escape(a.question_number)}.</b> {escape(a.answer)}{pages}",
                                styles["Normal"]), Spacer(1, 6)]
    buf = BytesIO()
    SimpleDocTemplate(buf, title=paper.title).build(story)
    return buf.getvalue()
