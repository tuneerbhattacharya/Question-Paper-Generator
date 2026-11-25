"""AI Question Paper Generator: Streamlit UI on top of an agentic RAG workflow (LangGraph)."""
import hashlib
import os
import tempfile
import time

import streamlit as st
from langchain_community.callbacks import get_openai_callback

from qpg import config, guardrails
from qpg.graph import build_graph
from qpg.ingest import build_index, document_sample, load_pdf, make_search
from qpg.llm import generator_llm, judge_llm
from qpg.nodes import paper_text
from qpg.render import to_pdf

# The API key comes ONLY from Streamlit secrets or the environment. Never hard-code keys.
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:  # no secrets file locally: fall back to the environment / .env
    from dotenv import load_dotenv
    load_dotenv()

st.set_page_config(page_title="AI Question Paper Generator", page_icon="📄")
st.title("📄 AI Question Paper Generator")
st.caption("Agentic RAG with LangGraph: plans, retrieves, self-checks and revises before you see the paper.")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Input guardrails** check the PDF and your instructions\n"
        "2. **Planner** picks topics and builds the exact paper blueprint\n"
        "3. **Retriever** searches the PDF for each topic\n"
        "4. **Grader** checks the evidence; weak topics get a rewritten search\n"
        "5. **Generator** writes the paper as structured data\n"
        "6. **Critic** checks format, coverage, difficulty and grounding; "
        "the generator revises if needed\n"
        "7. **Answer writer** builds the key with source pages\n"
        "8. **Output guardrails** verify every answer cites retrieved pages")
    st.caption(f"Model: {config.GENERATOR_MODEL} · Max revisions: {config.MAX_REVISIONS}")


@st.cache_resource(show_spinner=False)
def index_pdf(file_hash: str, data: bytes):
    """Cached per file content, so re-generating from the same PDF is instant and free."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(data)
        path = f.name
    pages = load_pdf(path)
    os.unlink(path)
    chars = sum(len(p.page_content) for p in pages)
    problems = guardrails.check_pdf(len(data), len(pages), chars)
    if problems:
        return None, None, problems
    index, n_chunks = build_index(pages)
    return index, (document_sample(pages), len(pages), n_chunks), []


pdf_file = st.file_uploader("Upload study material (PDF)", type="pdf")
instructions = st.text_input("Requirements", placeholder="e.g. Class 12 board exam, conceptual questions")
format_text = st.text_input("Format", placeholder="e.g. 10 MCQs of 1 mark, 5 short answers of 3 marks")
difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)

if st.button("Generate Question Paper", type="primary"):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Add it in Streamlit secrets.")
        st.stop()
    if not pdf_file:
        st.error("Please upload a PDF.")
        st.stop()

    data = pdf_file.getvalue()
    with st.spinner("Reading and indexing the PDF..."):
        index, meta, problems = index_pdf(hashlib.sha256(data).hexdigest(), data)
    if problems:
        st.error(" ".join(problems))
        st.stop()
    doc_sample, n_pages, n_chunks = meta

    graph = build_graph(generator_llm(), judge_llm(), make_search(index))
    start, final = time.time(), None
    with get_openai_callback() as usage, st.status("Agent is working...", expanded=True) as status:
        shown = 0
        for final in graph.stream(
                {"instructions": instructions, "format_text": format_text, "difficulty": difficulty,
                 "doc_sample": doc_sample, "trace": []},
                config={"recursion_limit": config.RECURSION_LIMIT}, stream_mode="values"):
            for line in final.get("trace", [])[shown:]:
                status.write(f"✓ {line}")
            shown = len(final.get("trace", []))
        status.update(label="Done", state="complete", expanded=False)

    st.session_state["result"] = {
        "state": final, "seconds": time.time() - start, "tokens": usage.total_tokens,
        "cost": usage.total_cost, "pages": n_pages, "chunks": n_chunks}

result = st.session_state.get("result")
if result:
    state = result["state"]
    if state.get("blocked"):
        st.error(f"Request blocked by guardrails: {state.get('block_reason')}")
        st.stop()

    if not state.get("paper") or not state.get("answer_key"):
        st.error("The agent stopped before finishing. Please try again.")
        st.stop()
    paper, key, plan = state["paper"], state["answer_key"], state["plan"]
    coverage, missing = guardrails.topic_coverage(paper, plan.topics)
    format_issues = state.get("format_issues", [])
    output_issues = state.get("output_issues", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Topic coverage", f"{coverage:.0%}")
    c2.metric("Format checks", "Pass" if not format_issues else f"{len(format_issues)} issue(s)")
    c3.metric("Revisions", state.get("revisions", 0))
    c4.metric("Cost", f"${result['cost']:.4f}")

    tab_paper, tab_key, tab_quality, tab_trace = st.tabs(
        ["Question Paper", "Answer Key", "Quality Report", "Agent Trace"])
    with tab_paper:
        st.text(paper_text(paper))
    with tab_key:
        for a in key.answers:
            st.markdown(f"**{a.question_number}.** {a.answer}  \n"
                        f"<small>Source: page(s) {', '.join(map(str, a.source_pages)) or 'none'}</small>",
                        unsafe_allow_html=True)
    with tab_quality:
        st.write("**Topics planned:**", ", ".join(plan.topics))
        if missing:
            st.warning(f"Not covered: {', '.join(missing)}")
        st.write("**Format issues:**", format_issues or "None")
        st.write("**Answer-key warnings:**", output_issues or "None")
        if state.get("feedback"):
            st.write("**Last critic feedback:**")
            st.text(state["feedback"])
    with tab_trace:
        for i, line in enumerate(state["trace"], 1):
            st.write(f"{i}. {line}")
        st.caption(f"{result['pages']} pages · {result['chunks']} chunks · "
                   f"{result['tokens']:,} tokens · {result['seconds']:.1f}s")

    st.download_button("📥 Download PDF (paper + answer key)", to_pdf(paper, key),
                       file_name="question_paper.pdf", mime="application/pdf")
