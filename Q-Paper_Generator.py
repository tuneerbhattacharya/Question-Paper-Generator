# ==========================================
# 📌 IMPORTS
# ==========================================
import streamlit as st
import tempfile
from dotenv import load_dotenv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 🔐 LOAD ENV
# ==========================================
load_dotenv()

# ==========================================
# 🎯 STREAMLIT UI
# ==========================================
st.set_page_config(page_title="AI Question Paper Generator")

st.title("📄 Advanced Question Paper Generator")

pdf_file = st.file_uploader("Upload PDF", type="pdf")

instructions = st.text_input("Enter requirements (e.g. entrance exam, MCQs)")
format_text = st.text_input("Enter format (marks distribution)")

difficulty = st.selectbox(
    "Select Difficulty",
    ["Easy", "Medium", "Hard"]
)

generate_btn = st.button("Generate Question Paper")

# ==========================================
# ⚡ CACHE VECTOR DB
# ==========================================
@st.cache_resource
def create_vectorstore(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

    return db, chunks


# ==========================================
# 📚 TOPIC EXTRACTION
# ==========================================
def extract_topics(chunks, llm):
    text_sample = "\n\n".join([c.page_content for c in chunks[:20]])

    prompt = f"""
Extract important topics from the following text.

{text_sample}

Return as a simple comma-separated list.
"""

    response = llm.invoke(prompt)
    topics = response.content.split(",")

    return [t.strip() for t in topics if t.strip()]


# ==========================================
# 📄 PDF GENERATION
# ==========================================
def generate_pdf(content):
    file_path = "question_paper.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    story = []

    for line in content.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 10))

    doc.build(story)

    return file_path


# ==========================================
# 🚀 MAIN LOGIC
# ==========================================
if generate_btn:

    if not pdf_file or not instructions or not format_text:
        st.error("Please fill all fields")
        st.stop()

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.read())
        file_path = temp_file.name

    # Create vector DB
    with st.spinner("Processing document..."):
        db, chunks = create_vectorstore(file_path)

    st.success("✅ Document processed!")

    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(temperature=0.7)
    parser = StrOutputParser()

    # ==========================================
    # 🧠 EXTRACT TOPICS
    # ==========================================
    with st.spinner("Extracting topics..."):
        topics = extract_topics(chunks, llm)

    st.write("📚 Topics Identified:", topics)

    # ==========================================
    # 🔍 RETRIEVE CONTEXT PER TOPIC
    # ==========================================
    all_context = []

    for topic in topics[:5]:  # limit topics for performance
        docs = retriever.invoke(topic)
        all_context.extend(docs)

    context = "\n\n".join({doc.page_content for doc in all_context})

    # ==========================================
    # 🧾 QUESTION PAPER PROMPT
    # ==========================================
    qp_prompt = PromptTemplate(
        template="""
You are an expert exam paper setter.

Using the following context:
{context}

Generate a QUESTION PAPER:

Requirements:
{instructions}

Format:
{format_text}

Difficulty:
{difficulty}

Ensure:
- Questions are divided into sections
- Cover different topics
- Maintain proper academic level

""",
        input_variables=["context", "instructions", "format_text", "difficulty"]
    )

    # ==========================================
    # 🧠 ANSWER KEY PROMPT
    # ==========================================
    ans_prompt = PromptTemplate(
        template="""
Based on the following question paper:

{question_paper}

Generate a detailed ANSWER KEY.
""",
        input_variables=["question_paper"]
    )

    # Chains
    qp_chain = qp_prompt | llm | parser
    ans_chain = ans_prompt | llm | parser

    # ==========================================
    # ⚡ GENERATE QUESTION PAPER
    # ==========================================
    with st.spinner("Generating question paper..."):
        question_paper = qp_chain.invoke({
            "context": context,
            "instructions": instructions,
            "format_text": format_text,
            "difficulty": difficulty
        })

    # ==========================================
    # 🧠 GENERATE ANSWERS
    # ==========================================
    with st.spinner("Generating answer key..."):
        answer_key = ans_chain.invoke({
            "question_paper": question_paper
        })

    # ==========================================
    # 📄 DISPLAY
    # ==========================================
    st.subheader("📘 Question Paper")
    st.write(question_paper)

    st.subheader("🧠 Answer Key")
    st.write(answer_key)

    # ==========================================
    # 📥 PDF DOWNLOAD
    # ==========================================
    pdf_path = generate_pdf(question_paper + "\n\nANSWER KEY\n\n" + answer_key)

    with open(pdf_path, "rb") as f:
        st.download_button(
            "📥 Download PDF",
            f,
            file_name="question_paper.pdf"
        )