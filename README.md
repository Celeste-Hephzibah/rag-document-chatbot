# 🤖 RAG Document Chatbot

A locally running AI chatbot that lets you upload any PDF and have 
a conversation with it — powered by RAG (Retrieval Augmented Generation).

## 📌 Overview
Built a complete RAG pipeline from scratch using LangChain, FAISS 
vector database, and TinyLlama LLM. Upload any PDF — textbook, research 
paper, resume, notes — and ask questions to get answers directly from 
the document. Runs 100% locally with no API costs.

## ✨ Features
- Upload any PDF document
- AI reads and indexes the document using vector embeddings
- Ask questions in natural language
- Get answers sourced directly from the document
- Full chat history maintained during session
- 100% local — no data leaves your laptop

## 🛠️ Tech Stack
- Python
- LangChain (RAG pipeline)
- FAISS (vector database)
- Sentence Transformers (embeddings)
- TinyLlama via Ollama (local LLM)
- Streamlit
- PyPDF

## 🔬 How It Works
1. PDF is uploaded and text extracted using PyPDF
2. Text split into chunks using RecursiveCharacterTextSplitter
3. Chunks converted to vector embeddings using Sentence Transformers
4. Embeddings stored in FAISS vector database
5. User question converted to embedding and matched against database
6. Top 3 relevant chunks retrieved
7. TinyLlama generates answer based on retrieved context

## 🧠 RAG Architecture
PDF → Text Extraction → Chunking → Embeddings → FAISS Index

↓

User Question → Embedding → Similarity Search → Top 3 Chunks

↓

TinyLlama → Final Answer
## 🚀 Run Locally
```bash
# Install Ollama from https://ollama.com
ollama pull tinyllama

git clone https://github.com/Celeste-Hephzibah/rag-document-chatbot
cd rag-document-chatbot
pip install -r requirements.txt
streamlit run app.py
```

## 💡 Interview Note
TinyLlama is used for fully local, offline operation. For production 
deployment, I would replace it with the Gemini API or OpenAI API 
for better accuracy and cloud deployment.

## 📁 Project Structure
rag-document-chatbot/

├── app.py              # Streamlit web app

└── requirements.txt