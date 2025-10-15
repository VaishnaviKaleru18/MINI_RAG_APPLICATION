#region Imports
import streamlit as st
from PyPDF2 import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
#endregion

#region Text Extraction Functions
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text + "\n"
    return text

def extract_text_from_txt(file):
    text = file.read().decode("utf-8")
    return text
#endregion

#region Text Chunking
def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    start = 0
    seen_chunks = set()
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) > 50 and chunk not in seen_chunks:
            chunks.append(chunk)
            seen_chunks.add(chunk)
        start += chunk_size - overlap
    return chunks
#endregion

#region Streamlit Page Configuration & CSS
st.set_page_config(page_title="Mini RAG Application", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main > div {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
        margin-left: auto;
        margin-right: auto;
    }
    .stApp {
        background: linear-gradient(to bottom, #f8f9fa, #e9ecef);
        color: #212529;
        font-family: 'Inter', sans-serif;
    }
    .css-1d391kg {
        padding: 1rem 2rem;
    }
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        color: #212529;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.5px;
        background: linear-gradient(to right, #212529, #495057);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1px solid #ced4da;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        background-color: #ffffff;
    }
    .stTextInput > div > div > input:focus {
        border-color: #495057;
        box-shadow: 0 0 0 3px rgba(73, 80, 87, 0.1);
    }
    .stFileUploader > div > div {
        border-radius: 12px;
        border: 2px dashed #adb5bd;
        padding: 2rem;
        background-color: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .stFileUploader > div > div:hover {
        border-color: #495057;
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(to right, #495057, #343a40);
        color: #ffffff;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(73, 80, 87, 0.2);
        font-family: 'Inter', sans-serif;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(73, 80, 87, 0.3);
    }
    .stMarkdown {
        font-family: 'Inter', sans-serif;
        line-height: 1.7;
    }
    .stExpander {
        border-radius: 12px;
        border: 1px solid #dee2e6;
        background-color: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    .stExpander > div > label {
        font-weight: 600;
        color: #343a40;
    }
    .chunk-container {
        background-color: #ffffff;
        border-left: 6px solid #495057;
        padding: 1.75rem;
        margin-bottom: 2rem;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    .chunk-container:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    .answer-box {
        background: linear-gradient(to bottom, #e9ecef, #f8f9fa);
        border-radius: 16px;
        padding: 2.5rem;
        margin-top: 2.5rem;
        border: 1px solid #ced4da;
        font-size: 1.2rem;
        line-height: 1.8;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        color: #212529;
    }
    .sources-box {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        font-size: 1rem;
        color: #495057;
        border: 1px dashed #adb5bd;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    section[data-testid="stFileUploader"] {
        width: 100%;
    }
    div[data-testid="stFileUploaderDropzone"] {
        width: 100%;
    }
    .stSpinner > div {
        border-top-color: #495057;
    }
    /* Full-width query input */
    .query-container .stTextInput {
        width: 100%;
    }
    .query-container {
        max-width: 100%;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)
#endregion

#region Header & Session State
st.title("Mini RAG (Retrieval-Augmented Generation) Application")

if "documents" not in st.session_state:
    st.session_state.documents = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
#endregion

#region Configuration Settings
with st.expander("Configuration Settings", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        chunk_size = st.number_input("Chunk size (words)", min_value=50, max_value=1000, value=300, step=50)
    with col2:
        overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=chunk_size-50, value=50, step=10)
    with col3:
        top_k = st.number_input("Top K retrieved chunks", min_value=1, max_value=10, value=3, step=1)
    with col4:
        confidence_threshold = st.slider("Minimum confidence score", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
#endregion

#region File Upload
st.markdown("### Upload Documents (PDF, DOCX, or TXT)")
uploaded_files = st.file_uploader(
    "", accept_multiple_files=True, type=['pdf', 'docx', 'txt'], key="files", label_visibility="collapsed"
)

if uploaded_files:
    with st.spinner("Processing uploaded documents..."):
        new_documents = []
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(uploaded_file)
            elif uploaded_file.type == "text/plain":
                text = extract_text_from_txt(uploaded_file)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                continue
            chunks = chunk_text(text, chunk_size, overlap)
            new_documents.extend([(uploaded_file.name, chunk) for chunk in chunks])
        
        st.session_state.documents.extend(new_documents)
        st.success(f"{len(new_documents)} new chunks processed and added. Total chunks available: {len(st.session_state.documents)}")
#endregion

#region Embeddings Generation
if st.session_state.documents:
    already_embedded = 0 if st.session_state.embeddings is None else len(st.session_state.embeddings)
    new_texts = [doc[1] for doc in st.session_state.documents[already_embedded:]]
    
    if new_texts:
        with st.spinner("Generating embeddings for new documents..."):
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            new_embeddings = embedder.encode(new_texts, convert_to_numpy=True, normalize_embeddings=True)

            if st.session_state.embeddings is None:
                st.session_state.embeddings = new_embeddings
                dimension = new_embeddings.shape[1]
                st.session_state.faiss_index = faiss.IndexFlatIP(dimension)
            else:
                st.session_state.embeddings = np.vstack([st.session_state.embeddings, new_embeddings])

            st.session_state.faiss_index.add(new_embeddings)
            st.success("Embeddings updated successfully!")
#endregion

#region Query & Retrieval
if st.session_state.documents and st.session_state.faiss_index is not None:
    st.markdown("### Submit Your Query")
    query = st.text_input("", placeholder="Enter your question about the uploaded documents here...", label_visibility="collapsed", key="query_input")
    
    if query:
        with st.spinner("Retrieving relevant content..."):
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            D, I = st.session_state.faiss_index.search(query_embedding, top_k)
            retrieved_chunks = [st.session_state.documents[i] for i in I[0]]
            
            filtered_chunks = []
            for idx, (doc_name, chunk) in enumerate(retrieved_chunks):
                score = max(min(D[0][idx], 1.0), 0.0)
                if score >= confidence_threshold:
                    filtered_chunks.append((doc_name, chunk, score))
            
            if not filtered_chunks:
                st.warning("No relevant content found matching the confidence threshold. Please refine your query or adjust settings.")
            else:
                context_text = "\n".join([chunk for _, chunk, _ in filtered_chunks])
                
                st.markdown("#### Retrieved Passages")
                for idx, (doc_name, chunk, score) in enumerate(filtered_chunks):
                    st.markdown(f"<div class='chunk-container'><strong>Document:</strong> {doc_name} | <strong>Chunk:</strong> {idx+1} | <strong>Confidence Score:</strong> {score:.4f}<br><br>{chunk}</div>", unsafe_allow_html=True)
                
                with st.spinner("Generating response using language model..."):
                    prompt = f"""
Answer the question using the context below. If the answer is not in the context, respond with "I’m sorry, but I could not locate an answer in the documents. Please review the materials or clarify your question."
Context:
{context_text}
Question:
{query}
"""
                    qa_pipeline = pipeline(
                        "text2text-generation",
                        model="google/flan-t5-base",
                        tokenizer="google/flan-t5-base"
                    )
                    result = qa_pipeline(prompt, max_length=256, do_sample=False)[0]['generated_text']
                    
                    top_sources = [f"{doc_name} – Chunk {idx+1}" for idx, (doc_name, _, _) in enumerate(filtered_chunks)]
                    citation_text = "Sources: " + ", ".join(top_sources)
                    
                    st.markdown("<div class='answer-box'><strong>Generated Answer:</strong><br><br>" + result + "</div>", unsafe_allow_html=True)
                    st.markdown("<div class='sources-box'>" + citation_text + "</div>", unsafe_allow_html=True)
#endregion
