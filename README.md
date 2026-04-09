# 📄 AI Question Paper Generator (RAG-based)

## 🚀 Overview

The **AI Question Paper Generator** is an intelligent application that automatically generates structured question papers and answer keys from uploaded PDF documents using **Retrieval-Augmented Generation (RAG)**.

It combines **LLMs + vector search (FAISS)** to ensure that generated questions are **context-aware, relevant, and academically structured**.

---

## 🎯 Problem Statement

Creating question papers manually is:

* Time-consuming ⏳
* Requires domain expertise 📚
* Difficult to ensure balanced coverage of topics

This project solves that by:

* Extracting key topics automatically
* Generating questions based on actual content
* Providing structured formats and answer keys

---

## 🧠 Key Features

### ✅ RAG-Based Question Generation

* Uses document context instead of hallucinating
* Ensures high-quality, relevant questions

### ✅ Topic Extraction

* Automatically identifies important topics from the document

### ✅ Difficulty Control

* Easy / Medium / Hard question generation

### ✅ Section-wise Question Paper

* Structured format (like real exams)

### ✅ Answer Key Generation

* Generates detailed answers using LLM

### ✅ PDF Export

* Download ready-to-use question paper

### ✅ Optimized Performance

* Uses caching for faster processing

---

## 🏗️ System Architecture

```
User Upload PDF
        ↓
Document Loader (PyPDFLoader)
        ↓
Text Splitting (RecursiveCharacterTextSplitter)
        ↓
Embeddings (OpenAIEmbeddings)
        ↓
Vector Store (FAISS)
        ↓
Retriever (Top-K Context)
        ↓
LLM (ChatOpenAI)
        ↓
Question Paper + Answer Key
```

---

## 🔍 How RAG Works in This Project

### Step 1: Chunking

* Large document → split into smaller chunks

### Step 2: Embedding

* Each chunk converted into vector representation

### Step 3: Storage

* Stored in FAISS vector database

### Step 4: Retrieval

* Queries (topics) used to fetch relevant chunks

### Step 5: Generation

* LLM generates questions using retrieved context

---

## 💡 Why Not Direct LLM?

| Without RAG ❌        | With RAG ✅              |
| -------------------- | ----------------------- |
| Hallucinated content | Context-based questions |
| Generic output       | Domain-specific         |
| No grounding         | Factually accurate      |

---

## ⚙️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** OpenAI (ChatOpenAI)
* **Embeddings:** OpenAIEmbeddings
* **Vector DB:** FAISS
* **PDF Processing:** PyPDFLoader
* **Text Splitting:** RecursiveCharacterTextSplitter
* **PDF Export:** ReportLab

---

## 📦 Installation

```bash
pip install streamlit langchain langchain-community langchain-openai faiss-cpu python-dotenv reportlab
```

---

## 🔐 Environment Setup

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🧪 Sample Workflow

1. Upload a PDF (e.g., textbook chapter)
2. Enter instructions:

   * "Generate 10 MCQs and 5 descriptive questions"
3. Select difficulty level
4. Click Generate
5. Download the question paper PDF

---

## ⚡ Optimization Techniques Used

### 🔹 Caching

* `@st.cache_resource` prevents recomputation of embeddings

### 🔹 Chunking Strategy

* Overlapping chunks improve context continuity

### 🔹 Top-K Retrieval

* Fetches most relevant chunks for better answers

---

## 🧠 Interview Questions You Can Expect

### ❓ What is RAG?

Retrieval-Augmented Generation combines:

* Retrieval (fetch relevant context)
* Generation (LLM output)

---

### ❓ Why FAISS?

* Fast similarity search
* Efficient for large datasets

---

### ❓ Why chunking is needed?

* LLMs have token limits
* Improves retrieval accuracy

---

### ❓ What is the role of embeddings?

* Convert text → vectors for semantic search

---

### ❓ Why caching?

* Avoid recomputing embeddings (expensive operation)

---

## 🚀 Future Improvements

* 🎯 Bloom’s Taxonomy-based questions
* 📊 Automatic marks distribution
* 🌐 Deployment on cloud
* 🧾 Multi-document support
* 🔍 Semantic filtering of questions

---

## 🏆 Key Learning Outcomes

* Built a full **RAG pipeline**
* Understood **vector databases**
* Learned **prompt engineering**
* Developed **LLM-based applications**
* Implemented **end-to-end AI system**

---

## 🙌 Conclusion

This project demonstrates how modern AI systems can:

* Automate complex academic tasks
* Improve efficiency
* Deliver intelligent, context-aware outputs

It showcases practical implementation of **Generative AI + Retrieval Systems** in a real-world use case.

---
