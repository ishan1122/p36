
import os
import tempfile
import hashlib
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="PaperPilot | Research Paper Q&A",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4

# --------------------------------------------------
# CUSTOM UI
# --------------------------------------------------

st.markdown("""
<style>
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg,#101d36,#294c72);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 34px;
    margin-bottom: 5px;
}
.hero p {
    color: #dbeafe;
    font-size: 16px;
}
.info-card {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    padding: 18px;
    min-height: 115px;
}
div.stButton > button {
    border-radius: 10px;
    min-height: 43px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# API KEY MANAGEMENT
# --------------------------------------------------

def get_api_key():

    # Streamlit Cloud Secrets first
    try:
        key = st.secrets.get("GOOGLE_API_KEY", "")

        if key:
            return str(key).strip()

    except Exception:
        pass

    # Local .env fallback
    return os.getenv("GOOGLE_API_KEY", "").strip()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 📘 PaperPilot")

    st.caption("PS36 · Research Paper Q&A")

    st.divider()

    st.markdown("### Project information")

    st.write("**Project:** Research Paper Q&A")
    st.write("**Technique:** Retrieval-Augmented Generation")
    st.write("**Vector database:** FAISS")

    st.divider()

    api_key = get_api_key()

    if api_key:
        st.success("Gemini API key detected")
    else:
        st.error("Gemini API key missing")

    st.caption(
        "For Streamlit Cloud, configure GOOGLE_API_KEY "
        "under Settings → Secrets."
    )

    model_name = st.text_input(
        "Gemini chat model",
        value=CHAT_MODEL,
    )

    st.divider()

    if st.button("Test Gemini connection", use_container_width=True):

        if not api_key:
            st.error("API key not configured.")

        else:
            try:
                test_embedding = GoogleGenerativeAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    google_api_key=api_key,
                )

                test_embedding.embed_query("Test Gemini connection")

                st.success("Gemini embedding connection successful!")

            except Exception as e:
                st.error(f"Connection failed: {e}")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>Research Paper Q&A</h1>
    <p>
        Your AI-powered research assistant.
        Upload a research paper and explore its methodology,
        datasets, findings and limitations.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "vectorstore": None,
    "processed_file_id": None,
    "num_pages": 0,
    "num_chunks": 0,
    "paper_name": "",
    "active_question": "",
    "chat_history": [],
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value

# --------------------------------------------------
# PDF PROCESSING
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_vectorstore(pdf_bytes, api_key, file_hash):

    tmp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)

        documents = loader.load()

        if not documents:
            raise ValueError("No readable pages found in PDF.")

        for doc in documents:

            doc.metadata["page_num"] = (
                doc.metadata.get("page", 0) + 1
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError(
                "No readable text found. "
                "This PDF may be scanned or image-based."
            )

        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings,
        )

        return vectorstore, len(documents), len(chunks)

    finally:

        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# --------------------------------------------------
# GEMINI ERROR HANDLING
# --------------------------------------------------

def show_friendly_error(error):

    message = str(error).lower()

    if (
        "401" in message
        or "unauthenticated" in message
        or "access_token_type_unsupported" in message
        or "api key not valid" in message
    ):

        st.error(
            "Gemini authentication failed. Generate a fresh Gemini "
            "API key in Google AI Studio and add it to Streamlit Cloud "
            "Settings → Secrets as GOOGLE_API_KEY. Do not use an "
            "OAuth access token. Reboot the app after saving."
        )

    elif "403" in message or "permission_denied" in message:

        st.error(
            "Google denied API access. Check your Gemini API "
            "project permissions and API availability."
        )

    elif "429" in message or "resource_exhausted" in message:

        st.error(
            "Gemini API quota or rate limit reached. "
            "Wait and try again."
        )

    else:

        st.error(f"Error: {error}")

# --------------------------------------------------
# UPLOAD SECTION
# --------------------------------------------------

left, right = st.columns([1.4, 0.8])

with left:

    st.subheader("1. Upload research paper")

    uploaded_file = st.file_uploader(
        "Select a research paper PDF",
        type=["pdf"],
        help="Upload a text-based PDF.",
    )

with right:

    st.subheader("About PaperPilot")

    st.markdown("""
    <div class="info-card">
        <b>RAG-based research assistant</b>
        <p>Extracts PDF text, retrieves relevant sections,
        and generates answers with page references.</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# PROCESS BUTTON
# --------------------------------------------------

if uploaded_file:

    file_bytes = uploaded_file.getvalue()

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    st.caption(
        f"Selected: {uploaded_file.name} · "
        f"{len(file_bytes) / 1024 / 1024:.2f} MB"
    )

    if st.button(
        "⚡ Process research paper",
        type="primary",
        use_container_width=True,
    ):

        if not api_key:

            st.error(
                "Configure GOOGLE_API_KEY in Streamlit Cloud Secrets."
            )

        else:

            # Avoid retaining a previous paper's index.
            if st.session_state.processed_file_id != file_hash:

                st.session_state.vectorstore = None
                st.session_state.chat_history = []

            with st.spinner(
                "Extracting PDF text and creating Gemini embeddings..."
            ):

                try:

                    vectorstore, pages, chunks = build_vectorstore(
                        file_bytes,
                        api_key,
                        file_hash,
                    )

                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed_file_id = file_hash
                    st.session_state.num_pages = pages
                    st.session_state.num_chunks = chunks
                    st.session_state.paper_name = uploaded_file.name
                    st.session_state.active_question = ""
                    st.session_state.chat_history = []

                    st.success(
                        f"Successfully indexed {uploaded_file.name}"
                    )

                except Exception as e:

                    show_friendly_error(e)

                    with st.expander("Technical error details"):
                        st.code(str(e))

else:

    st.info("Upload a PDF to begin.")

# --------------------------------------------------
# PAPER DASHBOARD
# --------------------------------------------------

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader("2. Paper overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "PDF pages",
        st.session_state.num_pages,
    )

    c2.metric(
        "Text chunks",
        st.session_state.num_chunks,
    )

    c3.metric(
        "Retrieved passages",
        RETRIEVAL_K,
    )

    st.caption(
        f"Currently indexed: {st.session_state.paper_name}"
    )

# --------------------------------------------------
# ANSWER FUNCTION
# --------------------------------------------------

def answer_question(
    vectorstore,
    question,
    gemini_api_key,
    selected_model,
):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )

    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return (
            "I could not find relevant information in the uploaded paper.",
            [],
        )

    context_parts = []

    for doc in retrieved_docs:

        page = doc.metadata.get("page_num", "?")

        context_parts.append(
            f"[PDF Page {page}]\n{doc.page_content.strip()}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template("""
You are an academic research paper assistant.

Answer ONLY using the supplied excerpts from the uploaded PDF.

Rules:
1. Do not use outside knowledge.
2. Never invent methods, datasets, metrics or results.
3. If information is absent, say it is not found in the retrieved sections.
4. Cite relevant PDF pages using [PDF Page X].
5. Clearly distinguish author statements from your own inference.
6. Use structured bullet points when useful.

RESEARCH PAPER EXCERPTS:

{context}

USER QUESTION:

{question}

ANSWER:
""")

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        google_api_key=gemini_api_key,
        temperature=0.2,
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    return answer, retrieved_docs

# --------------------------------------------------
# QUESTION SECTION
# --------------------------------------------------

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader("3. Ask your research paper")

    st.caption(
        "Choose a preset question or ask something specific."
    )

    col1, col2, col3, col4 = st.columns(4)

    if col1.button("🔬 Method", use_container_width=True):

        st.session_state.active_question = (
            "Explain the methodology, model architecture, "
            "and proposed approach used in this paper."
        )

    if col2.button("📊 Dataset", use_container_width=True):

        st.session_state.active_question = (
            "What datasets were used? Explain their sources, "
            "size and preprocessing if reported."
        )

    if col3.button("🏆 Results", use_container_width=True):

        st.session_state.active_question = (
            "Explain the main experimental results, "
            "evaluation metrics and findings."
        )

    if col4.button("⚠️ Limitations", use_container_width=True):

        st.session_state.active_question = (
            "What limitations, weaknesses and future work "
            "are discussed by the authors?"
        )

    with st.form("question_form", clear_on_submit=True):

        user_input = st.text_input(
            "Your question",
            placeholder="Ask anything about the research paper...",
        )

        submitted = st.form_submit_button(
            "Ask PaperPilot",
            type="primary",
        )

    if submitted and user_input.strip():

        st.session_state.active_question = user_input.strip()

    # --------------------------------------------------
    # DISPLAY QUESTION AND ANSWER
    # --------------------------------------------------

    question = st.session_state.active_question

    if question:

        st.markdown("### Question")

        st.info(question)

        with st.spinner(
            "Retrieving passages and generating answer..."
        ):

            try:

                answer, docs = answer_question(
                    st.session_state.vectorstore,
                    question,
                    api_key,
                    model_name.strip() or CHAT_MODEL,
                )

                st.markdown("### Answer")

                st.markdown(answer)

                with st.expander(
                    "📄 View retrieved passages and page citations"
                ):

                    for i, doc in enumerate(docs, start=1):

                        page = doc.metadata.get("page_num", "?")

                        st.markdown(
                            f"**Passage {i} · PDF Page {page}**"
                        )

                        st.write(doc.page_content)

                        if i < len(docs):
                            st.divider()

            except Exception as e:

                show_friendly_error(e)

                with st.expander("Technical error details"):
                    st.code(str(e))

        st.session_state.active_question = ""

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption("""
PaperPilot · PS36 · Research Paper Q&A

Built using Python, Streamlit, Gemini, LangChain and FAISS.

Always verify important findings and page citations against
the original research paper.
""")

