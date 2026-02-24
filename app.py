import streamlit as st
import os
import json
import hashlib
import datetime
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

# -------------------------
# Load Environment
# -------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------
# App Config
# -------------------------
st.set_page_config(
    page_title="EduRAG AI Tutor",
    layout="wide"
)

# -------------------------
# User Database Setup
# -------------------------
USER_DB = "users.json"

if not os.path.exists(USER_DB):
    with open(USER_DB, "w") as f:
        json.dump({}, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    with open(USER_DB, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f, indent=4)

def register(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = {
        "password": hash_password(password),
        "progress": {},
        "reviews": {}
    }
    save_users(users)
    return True

def login(username, password):
    users = load_users()
    if username in users and users[username]["password"] == hash_password(password):
        return True
    return False

# -------------------------
# Authentication UI
# -------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("EduRAG AI Tutor")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            if register(new_user, new_pass):
                st.success("Registered successfully")
            else:
                st.error("User already exists")

    st.stop()

# -------------------------
# Main App
# -------------------------
st.sidebar.success(f"Logged in as {st.session_state.username}")

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# Vectorstore setup
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if os.path.exists("edurag_db"):
    vectorstore = Chroma(
        persist_directory="edurag_db",
        embedding_function=embedding_model
    )
else:
    vectorstore = None

# Upload PDF
st.sidebar.header("Upload Study Material")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

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
    st.sidebar.success("Material indexed successfully")

# -------------------------
# Modes
# -------------------------
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Explain", "Quiz", "Exam Simulation"]
)

# -------------------------
# Prompts
# -------------------------
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
Generate a 10-question exam.
Include MCQs and short answers.
Provide answer key at end.

Context:
{context}

Topic:
{question}
""")

difficulty_prompt = ChatPromptTemplate.from_template("""
Classify question difficulty:
beginner, intermediate, advanced.

Question:
{question}
Return one word only.
""")

# -------------------------
# Question Input
# -------------------------
st.title("Ask Your AI Tutor")

question = st.text_area("Enter your question")

if st.button("Submit") and vectorstore:

    # Auto Difficulty Detection
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

    st.subheader(f"Detected Difficulty: {difficulty}")
    st.write(response.content)

    # Track progress
    users = load_users()
    user = st.session_state.username
    topic = question[:50]

    users[user]["progress"].setdefault(topic, 0)
    users[user]["progress"][topic] += 1

    # Spaced repetition scheduling
    next_review = datetime.date.today() + datetime.timedelta(days=3)
    users[user]["reviews"][topic] = str(next_review)

    save_users(users)

    st.info(f"Next review scheduled: {next_review}")
