import streamlit as st
import os
import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import MultiQueryRetriever

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="EduRAG AI Tutor",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.chat-bubble-user {
    background-color: #2563eb;
    padding: 12px 18px;
    border-radius: 18px;
    margin-bottom: 8px;
    width: fit-content;
    max-width: 70%;
}

.chat-bubble-ai {
    background-color: #334155;
    padding: 12px 18px;
    border-radius: 18px;
    margin-bottom: 8px;
    width: fit-content;
    max-width: 70%;
}

.sidebar .sidebar-content {
    background: #0f172a;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Environment
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Session State
# -----------------------------
if "user" not in st.session_state:
    st.session_state.user = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Authentication UI
# -----------------------------
st.title("🎓 EduRAG AI Tutor")

if not st.session_state.user:

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            email = email.strip().lower()
            try:
                res = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })
                st.session_state.user = res.user
                st.rerun()
            except Exception as e:
                st.error("Login failed. Check credentials.")

    with tab2:
        email = st.text_input("New Email")
        password = st.text_input("New Password", type="password")

        if st.button("Register"):
            email = email.strip().lower()
            try:
                res = supabase.auth.sign_up({
                    "email": email,
                    "password": password
                })
                st.success("Account created. Check your email if confirmation is enabled.")
            except Exception:
                st.error("Registration failed.")

    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
user = st.session_state.user

st.sidebar.success(f"Logged in as {user.email}")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.session_state.messages = []
    st.rerun()

mode = st.sidebar.selectbox(
    "Learning Mode",
    ["Explain", "Quiz", "Exam"]
)

uploaded_file = st.sidebar.file_uploader("Upload Study Material (PDF)", type="pdf")

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = None

if os.path.exists("edurag_db"):
    vectorstore = Chroma(
        persist_directory="edurag_db",
        embedding_function=embedding_model
    )

# -----------------------------
# PDF Upload
# -----------------------------
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory="edurag_db"
    )
    vectorstore.persist()

    st.sidebar.success("Material indexed successfully!")

# -----------------------------
# Prompts
# -----------------------------
difficulty_prompt = ChatPromptTemplate.from_template("""
Classify question difficulty:
beginner, intermediate, advanced.

Question:
{question}
Return one word only.
""")

simple_prompt = ChatPromptTemplate.from_template("""
Explain clearly and simply.

Context:
{context}

Question:
{question}
""")

quiz_prompt = ChatPromptTemplate.from_template("""
Generate 5 quiz questions with answers.

Context:
{context}

Topic:
{question}
""")

exam_prompt = ChatPromptTemplate.from_template("""
Generate 10-question exam with answer key.

Context:
{context}

Topic:
{question}
""")

# -----------------------------
# Chat Interface
# -----------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-ai">{msg["content"]}</div>', unsafe_allow_html=True)

question = st.chat_input("Ask me anything about your material...")

if question and vectorstore:

    st.session_state.messages.append({"role": "user", "content": question})

    diff_chain = difficulty_prompt | llm
    difficulty = diff_chain.invoke({"question": question}).content.strip()

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )

    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    if mode == "Explain":
        chain = simple_prompt | llm
    elif mode == "Quiz":
        chain = quiz_prompt | llm
    else:
        chain = exam_prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    answer = f"**Difficulty:** {difficulty}\n\n{response.content}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# -----------------------------
# Save Progress
# -----------------------------
if question and vectorstore:

    supabase.table("progress").insert({
        "user_id": user.id,
        "topic": question[:100],
        "score": 100 if mode == "Explain" else 80
    }).execute()

    next_review = datetime.date.today() + datetime.timedelta(days=3)

    supabase.table("reviews").insert({
        "user_id": user.id,
        "topic": question[:100],
        "next_review": str(next_review)
    }).execute()
