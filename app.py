import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

@st.cache_resource  
def load_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(uploaded_file, embeddings):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

st.set_page_config(page_title="RAG Document Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG Document Chatbot")
st.write("Upload a PDF and chat with it using AI — powered by local LLM!")

st.sidebar.header("About")
st.sidebar.write("""
This app uses RAG (Retrieval Augmented Generation) to answer questions from your documents.

**How it works:**
1. Upload a PDF
2. AI reads and indexes it
3. Ask any question
4. Get answers from the document

**Tech Stack:**
- LangChain (RAG pipeline)
- FAISS (vector database)
- TinyLlama (local LLM)
- Sentence Transformers (embeddings)
""")

st.sidebar.divider()
st.sidebar.info("🔒 100% Local\n\nNo data leaves your laptop!")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading and indexing document..."):
        embeddings = load_embeddings()
        vectorstore = process_pdf(uploaded_file, embeddings)
        llm = Ollama(model="tinyllama")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )
    st.success(f"✅ Document '{uploaded_file.name}' indexed successfully!")

    st.subheader("💬 Chat with your document")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if question := st.chat_input("Ask anything about your document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = qa_chain.invoke({"query": question})["result"]
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👆 Please upload a PDF to get started.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.write("Upload any PDF — textbook, notes, resume, research paper")
    with col2:
        st.markdown("### 2️⃣ Index")
        st.write("AI reads and indexes the document using vector embeddings")
    with col3:
        st.markdown("### 3️⃣ Chat")
        st.write("Ask questions and get answers directly from the document")