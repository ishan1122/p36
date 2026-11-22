import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="Research Paper Q&A (PS36)",
    page_icon="📚",
    layout="wide",
)

# Sidebar configuration matching the project UI
with st.sidebar:
    st.markdown("## Project information")
    st.markdown("**Project:** Research Paper Q&A")
    st.markdown("**Pipeline:** PDF -> chunks -> Gemini embeddings -> FAISS -> Gemini answer")
    st.divider()

    default_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    api_key = st.text_input(
        "Gemini API key",
        value=default_key,
        type="password",
        placeholder="Paste your Gemini API key...",
    )
    model_name = st.text_input(
        "Gemini model",
        value="gemini-2.5-flash",
    )
    st.caption("Alternatively, set GOOGLE_API_KEY in your local .env file.")

# Main Header
st.markdown("# 📚 Research Paper Q&A")
st.caption("PS36 · AI-powered academic paper analysis using Retrieval-Augmented Generation (RAG)")
st.write("Upload a research paper and ask about its methodology, dataset, results and limitations.")

uploaded_file = st.file_uploader("Upload research paper (PDF)", type=["pdf"])


@st.cache_resource(show_spinner=False)
def build_vectorstore(pdf_bytes: bytes, gemini_api_key: str):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            return None, 0

        for doc in documents:
            doc.metadata["page_num"] = doc.metadata.get("page", 0) + 1

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_api_key,
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore, len(documents)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def answer_question(vectorstore, question: str, gemini_api_key: str, selected_model: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(question)

    formatted_chunks = []
    for doc in retrieved_docs:
        page_num = doc.metadata.get("page_num", "?")
        clean_text = doc.page_content.strip()
        formatted_chunks.append(f"[Page {page_num}]\n{clean_text}")

    context_text = "\n\n---\n\n".join(formatted_chunks)

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful research assistant. Answer the user's question strictly using the provided excerpts from the research paper.
If the answer is not clearly supported by the excerpts, state that it is not mentioned in the provided context.
Include page numbers (e.g., [Page 3]) when referencing claims.
Format the response cleanly with bullet points where helpful.

Context:
{context}

Question: {question}

Answer:"""
    )

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        google_api_key=gemini_api_key,
        temperature=0.2,
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": question})
    return answer, retrieved_docs


if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0
if "active_question" not in st.session_state:
    st.session_state.active_question = ""

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_id = f"{uploaded_file.name}_{len(file_bytes)}"

    if st.session_state.processed_file_id != file_id:
        st.session_state.vectorstore = None
        st.session_state.active_question = ""

    if st.button("Process research paper", type="primary", use_container_width=True):
        if not api_key.strip():
            st.error("Please enter your Gemini API key in the sidebar before processing.")
        else:
            with st.spinner("Extracting text, generating embeddings, and indexing into FAISS..."):
                try:
                    vs, total_pages = build_vectorstore(file_bytes, api_key.strip())
                    if vs is None:
                        st.error("Could not extract readable text from this PDF.")
                    else:
                        st.session_state.vectorstore = vs
                        st.session_state.num_pages = total_pages
                        st.session_state.processed_file_id = file_id
                        st.success(f"Indexed **{uploaded_file.name}** ({total_pages} pages) successfully!")
                except Exception as e:
                    st.error(f"Error indexing PDF: {e}")
else:
    st.info("Upload a PDF and select **Process research paper** to begin.")

st.markdown("**Preset questions:** Method · Dataset · Results · Limitations")

if st.session_state.vectorstore is not None:
    st.divider()
    st.subheader("Ask the Research Paper")

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("🔬 Method", use_container_width=True):
        st.session_state.active_question = (
            "What methodology, model architecture, or proposed approach is described in this research paper?"
        )
    if col2.button("📊 Dataset", use_container_width=True):
        st.session_state.active_question = (
            "What datasets, benchmarks, or data preprocessing steps were used in this paper?"
        )
    if col3.button("🏆 Results", use_container_width=True):
        st.session_state.active_question = (
            "What are the main experimental results, evaluation metrics, and key findings reported?"
        )
    if col4.button("⚠️ Limitations", use_container_width=True):
        st.session_state.active_question = (
            "What limitations, failure cases, or future work directions are mentioned in this paper?"
        )

    with st.form("custom_qa_form", clear_on_submit=False):
        user_input = st.text_input(
            "Or enter a custom question:",
            placeholder="e.g., How does the proposed model compare against baseline methods?",
        )
        submitted = st.form_submit_button("Ask Question")
        if submitted and user_input.strip():
            st.session_state.active_question = user_input.strip()

    if st.session_state.active_question:
        if not api_key.strip():
            st.error("Please provide your Gemini API key in the sidebar.")
        else:
            st.markdown(f"#### Question: *{st.session_state.active_question}*")
            with st.spinner("Retrieving relevant passages and generating grounded answer..."):
                try:
                    answer, docs = answer_question(
                        st.session_state.vectorstore,
                        st.session_state.active_question,
                        api_key.strip(),
                        model_name.strip() or "gemini-2.5-flash",
                    )
                    st.markdown(answer)

                    with st.expander("View Retrieved Source Passages & Page Citations"):
                        for i, doc in enumerate(docs, start=1):
                            page = doc.metadata.get("page_num", "?")
                            st.markdown(f"**Chunk {i} — Page {page}**")
                            st.caption(doc.page_content)
                            if i < len(docs):
                                st.divider()
                except Exception as e:
                    st.error(f"Error generating response: {e}")

st.divider()
st.caption("Verify important claims against the original paper. Answers may omit information not present in retrieved passages.")
