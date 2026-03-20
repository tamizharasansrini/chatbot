import os
import json
import shutil
from datetime import datetime

import streamlit as st
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Advanced PDF Chatbot",
    page_icon="📘",
    layout="wide"
)

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
USERS_FILE = "users.json"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.title-box {
    background: linear-gradient(90deg, #1f4e79, #2e86c1);
    padding: 18px;
    border-radius: 14px;
    color: white;
    margin-bottom: 18px;
}
.small-note {
    font-size: 14px;
    color: #666;
}
.answer-box {
    background: #f7f9fc;
    border-left: 5px solid #2e86c1;
    padding: 16px;
    border-radius: 10px;
    margin-top: 10px;
}
.summary-box {
    background: #f5fff7;
    border-left: 5px solid #2ecc71;
    padding: 16px;
    border-radius: 10px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SESSION STATE
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# -----------------------------
# AUTH FUNCTIONS
# -----------------------------
def load_users():
    if not os.path.exists(USERS_FILE):
        default_users = {
            "admin": "admin123",
            "student": "student123"
        }
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=2)

    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def authenticate(username, password):
    users = load_users()
    return username in users and users[username] == password


# -----------------------------
# PDF + VECTOR FUNCTIONS
# -----------------------------
def save_uploaded_files(uploaded_files):
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths


def extract_text_from_pdfs(pdf_paths):
    documents = []

    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": os.path.basename(pdf_path),
                                "page": page_num
                            }
                        )
                    )
        except Exception as e:
            st.error(f"Error reading {pdf_path}: {e}")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vectorstore


def load_vector_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectorstore


def get_llm():
    return ChatOllama(
        model="llama3.1",
        temperature=0
    )


def summarize_pdfs(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    docs = retriever.invoke("Give an overall summary of this document including main topic, important points, and conclusion.")

    context = "\n\n".join(
        [f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}\n{doc.page_content}" for doc in docs]
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF summarizer.

Using only the provided context, generate:
1. Main topic
2. Important points
3. Short conclusion

If the content is insufficient, say so clearly.

Context:
{context}
""")

    chain = prompt | get_llm()
    response = chain.invoke({"context": context})
    return response.content, docs


def ask_question(question, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context = "\n\n".join(
        [f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}\n{doc.page_content}" for doc in docs]
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF chatbot.

Answer the user's question ONLY from the provided context.

Rules:
1. Give a short direct answer first.
2. Then add a small section called "Sources used".
3. If the answer is not found in the context, say:
   "I could not find the answer in the uploaded PDF."
4. Do not make up information.

Question:
{question}

Context:
{context}
""")

    chain = prompt | get_llm()
    response = chain.invoke({
        "question": question,
        "context": context
    })

    return response.content, docs


def prepare_download_text(username, answer, summary, history):
    lines = []
    lines.append("PDF CHATBOT REPORT")
    lines.append("=" * 50)
    lines.append(f"User: {username}")
    lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if summary:
        lines.append("DOCUMENT SUMMARY")
        lines.append("-" * 50)
        lines.append(summary)
        lines.append("")

    if answer:
        lines.append("LATEST ANSWER")
        lines.append("-" * 50)
        lines.append(answer)
        lines.append("")

    if history:
        lines.append("CHAT HISTORY")
        lines.append("-" * 50)
        for i, item in enumerate(history, start=1):
            lines.append(f"Q{i}: {item['question']}")
            lines.append(f"A{i}: {item['answer']}")
            lines.append("")

    return "\n".join(lines)


def clear_all():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    st.session_state.db_ready = False
    st.session_state.chat_history = []
    st.session_state.last_answer = ""
    st.session_state.last_summary = ""
    st.session_state.processed_files = []


# -----------------------------
# LOGIN PAGE
# -----------------------------
if not st.session_state.logged_in:
    st.markdown('<div class="title-box"><h1>📘 Advanced PDF Chatbot</h1><p>Login to access your final year project app</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

        st.markdown("**Demo users:**")
        st.code("admin / admin123\nstudent / student123")

    st.stop()


# -----------------------------
# MAIN APP
# -----------------------------
st.markdown(
    f'<div class="title-box"><h1>📘 Advanced PDF Chatbot</h1><p>Welcome, {st.session_state.username}</p></div>',
    unsafe_allow_html=True
)

left, right = st.columns([1, 2])

# -----------------------------
# SIDEBAR-LIKE LEFT PANEL
# -----------------------------
with left:
    st.subheader("⚙️ Controls")

    st.markdown("**Required Ollama models**")
    st.code("ollama pull llama3.1\nollama pull nomic-embed-text", language="bash")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing PDFs..."):
                saved_paths = save_uploaded_files(uploaded_files)
                raw_docs = extract_text_from_pdfs(saved_paths)

                if not raw_docs:
                    st.error("No readable text found in the uploaded PDFs.")
                else:
                    chunks = split_documents(raw_docs)
                    build_vector_store(chunks)
                    st.session_state.db_ready = True
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    st.success(f"Done. Indexed {len(chunks)} chunks.")

    if st.button("Generate PDF Summary", use_container_width=True):
        if not st.session_state.db_ready:
            st.warning("Process PDFs first.")
        else:
            with st.spinner("Generating summary..."):
                vectorstore = load_vector_store()
                summary, _ = summarize_pdfs(vectorstore)
                st.session_state.last_summary = summary
                st.success("Summary generated.")

    if st.button("Clear Database", use_container_width=True):
        clear_all()
        st.success("All data cleared.")

    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    st.markdown("---")
    st.subheader("📂 Processed Files")
    if st.session_state.processed_files:
        for name in st.session_state.processed_files:
            st.write(f"• {name}")
    else:
        st.write("No files processed yet.")

# -----------------------------
# MAIN CONTENT RIGHT
# -----------------------------
with right:
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Summary", "⬇️ Download"])

    # CHAT TAB
    with tab1:
        st.subheader("Ask Questions from PDF")

        question = st.text_input("Enter your question")

        if st.button("Get Answer"):
            if not st.session_state.db_ready:
                st.warning("Please process PDFs first.")
            elif not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Generating answer..."):
                    vectorstore = load_vector_store()
                    answer, source_docs = ask_question(question, vectorstore)

                    st.session_state.last_answer = answer
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": source_docs
                    })

        if st.session_state.last_answer:
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown("### Latest Answer")
            st.write(st.session_state.last_answer)
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("Chat History")

            for i, item in enumerate(reversed(st.session_state.chat_history), start=1):
                with st.expander(f"Question {i}: {item['question']}"):
                    st.markdown("**Answer:**")
                    st.write(item["answer"])

                    st.markdown("**Retrieved Source Chunks:**")
                    for doc in item["sources"]:
                        st.write(f"- {doc.metadata.get('source')} | Page {doc.metadata.get('page')}")

    # SUMMARY TAB
    with tab2:
        st.subheader("Document Summary")

        if st.session_state.last_summary:
            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
            st.write(st.session_state.last_summary)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Click 'Generate PDF Summary' to create a summary.")

    # DOWNLOAD TAB
    with tab3:
        st.subheader("Download Answers")

        download_text = prepare_download_text(
            st.session_state.username,
            st.session_state.last_answer,
            st.session_state.last_summary,
            st.session_state.chat_history
        )

        st.download_button(
            label="Download Report as TXT",
            data=download_text,
            file_name="pdf_chatbot_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.markdown('<p class="small-note">This downloads the latest summary, answer, and chat history.</p>', unsafe_allow_html=True)