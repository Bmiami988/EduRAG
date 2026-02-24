# EduRAG – AI-Powered Adaptive Tutoring SaaS

EduRAG is a production-ready AI tutoring platform designed to deliver personalized education using Retrieval-Augmented Generation (RAG).

Students can upload study materials (PDFs), ask questions, generate quizzes, simulate exams, and receive adaptive explanations based on detected difficulty level. The system tracks progress and schedules intelligent spaced repetition reviews.

Built for scalable, real-world deployment.

---

## Features

- Multi-user authentication system
- PDF document ingestion and indexing
- Retrieval-Augmented Generation (RAG)
- Auto difficulty detection (beginner / intermediate / advanced)
- Explain mode
- Quiz generation mode
- Exam simulation mode
- Student progress tracking
- Spaced repetition scheduling
- Persistent Chroma vector database
- Groq LLM (llama-3.3-70b-versatile)
- Environment-based API key management
- Production-ready Streamlit UI

---

## Tech Stack

- Streamlit
- LangChain
- LangGraph
- Groq API (LLaMA 3.3 70B)
- Chroma Vector Database
- HuggingFace Embeddings
- Python
- JSON-based persistent storage (upgradeable to PostgreSQL)

---

## Architecture Overview

1. User logs in
2. Uploads study material
3. Documents are chunked and embedded
4. Stored in persistent Chroma vector store
5. Multi-query retriever improves search quality
6. Difficulty classifier determines explanation depth
7. LLM generates response
8. Progress is tracked
9. Spaced repetition review is scheduled

---
