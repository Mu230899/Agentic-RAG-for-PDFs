



# ============================================================
# AGENTIC RAG SYSTEM (GEMINI 2.5 FLASH, STUDENT FRIENDLY)
# ============================================================

import os
import re
import faiss
import numpy as np
import streamlit as st
from typing import List
from dataclasses import dataclass
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

genai.configure(api_key=os.getenv("gemini_api_key"))

# ============================================================
# CONFIG
# ============================================================

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "rag_index"
GEMINI_MODEL = "gemini-2.5-flash"

genai.configure(api_key=os.getenv("gemini_api_key"))

# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict

# ============================================================
# PDF INGESTION
# ============================================================

def read_pdfs(files) -> List[Chunk]:
    chunks = []

    for pdf in files:
        reader = PdfReader(pdf)
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            text = re.sub(r"\s+", " ", text).strip()
            words = text.split()

            chunk_size = 220
            overlap = 40

            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = " ".join(words[i:i + chunk_size])
                chunk_id = f"{pdf.name}_p{page_no}_{i}"

                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=chunk_text,
                        metadata={"pdf": pdf.name, "page": page_no}
                    )
                )

    return chunks

# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

# ============================================================
# FAISS VECTOR STORE
# ============================================================

class FAISSStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.meta = []

    def add(self, embeddings, chunks):
        self.index.add(embeddings)
        for c in chunks:
            self.texts.append(c.text)
            self.meta.append({"id": c.id, **c.metadata})

    def search(self, query_emb, top_k=5):
        D, I = self.index.search(np.array([query_emb]).astype("float32"), top_k)
        results = []
        for i in I[0]:
            if i < len(self.texts):
                results.append({
                    "text": self.texts[i],
                    "metadata": self.meta[i]
                })
        return results

    def save(self):
        faiss.write_index(self.index, INDEX_PATH + ".faiss")
        np.save(INDEX_PATH + "_texts.npy", self.texts)
        np.save(INDEX_PATH + "_meta.npy", self.meta)

    def load(self):
        self.index = faiss.read_index(INDEX_PATH + ".faiss")
        self.texts = np.load(INDEX_PATH + "_texts.npy", allow_pickle=True).tolist()
        self.meta = np.load(INDEX_PATH + "_meta.npy", allow_pickle=True).tolist()

# ============================================================
# AGENT 1 – QUERY UNDERSTANDING
# ============================================================

def classify_query(q: str) -> str:
    q = q.lower()
    if any(w in q for w in ["compare", "difference", "vs"]):
        return "COMPARE"
    if any(w in q for w in ["how", "steps", "procedure"]):
        return "PROCEDURE"
    if len(q.split()) > 12:
        return "COMPLEX"
    return "FACT"

# ============================================================
# AGENT 2 – RETRIEVAL STRATEGY
# ============================================================

def retrieval_top_k(q_type: str):
    return {
        "FACT": 3,
        "PROCEDURE": 6,
        "COMPARE": 8,
        "COMPLEX": 10
    }.get(q_type, 5)

# ============================================================
# AGENT 3 – GEMINI ANSWER GENERATION
# ============================================================

def generate_answer_gemini(question: str, context: str) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""
You are a university teaching assistant.

INSTRUCTIONS:
- Answer ONLY using the provided context
- Write in clear paragraphs
- Use bullet points when appropriate
- Maintain proper punctuation
- Cite sources using [chunk_id]
- If information is missing, state it clearly

Context:
{context}

Question:
{question}

Final Answer:
"""

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 500
        }
    )

    return response.text.strip()

# ============================================================
# FALLBACK (RARELY USED)
# ============================================================

def extractive_answer(results):
    paragraphs = []
    citations = []

    for r in results:
        sentences = re.split(r'(?<=[.!?])\s+', r["text"])
        paragraphs.append(" ".join(sentences[:2]))
        citations.append(f"{r['metadata']['id']} (page {r['metadata']['page']})")

    return "\n\n".join(paragraphs), sorted(set(citations))

# ============================================================
# FULL AGENTIC RAG PIPELINE
# ============================================================

def agentic_rag(question, store, embedder):
    q_type = classify_query(question)
    top_k = retrieval_top_k(q_type)

    q_emb = embedder.encode(question)
    results = store.search(q_emb, top_k=top_k)

    if not results:
        return "No relevant information found.", []

    context = "\n\n".join(
        f"[{r['metadata']['id']}] {r['text']}" for r in results
    )

    answer = generate_answer_gemini(question, context)

    citations = [
        f"{r['metadata']['id']} (page {r['metadata']['page']})"
        for r in results
    ]

    return answer, sorted(set(citations))

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config("Agentic RAG (Gemini)", layout="wide")
st.title("📚 Agentic RAG for Students (Gemini 2.5 Flash)")

st.markdown("""
✅ Multi-PDF Question Answering  
✅ Clean, well-structured answers  
✅ Proper punctuation  
✅ Gemini-powered reasoning  
✅ Citation-backed responses  
""")

embedder = load_embedder()
store = FAISSStore(dim=384)

if os.path.exists(INDEX_PATH + ".faiss"):
    store.load()
    st.success("✅ Existing index loaded")

uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded and st.button("Build Index"):
    with st.spinner("Indexing PDFs..."):
        chunks = read_pdfs(uploaded)
        embeddings = embedder.encode([c.text for c in chunks])
        store.add(np.array(embeddings).astype("float32"), chunks)
        store.save()
    st.success("✅ Index built successfully")

st.divider()

question = st.text_input("Ask a question based on the PDFs")

if st.button("Get Answer") and question:
    with st.spinner("Generating answer..."):
        answer, citations = agentic_rag(question, store, embedder)

    st.subheader("Answer")
    st.markdown(answer)

    st.subheader("Citations")
    for c in citations:
        st.write("•", c)

st.caption("Agentic RAG | Gemini 2.5 Flash | Portfolio-Ready")

