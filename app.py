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
from langchain.retrievers.multi_query import MultiQueryRetriever

# -----------------------------
# Load Environment
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="EduRAG AI Tutor", layout="wide")

# -----------------------------
# Authentication
# -----------------------------
if "user" not in st.session_state:
    st.session_state.user = None

st.title("EduRAG AI Tutor")

if not st.session_state.user:

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            res = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            if res.user:
                st.session_state.user = res.user
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        email = st.text_input("New Email")
        password = st.text_input("New Password", type="password")
        if st.button("Register"):
            res = supabase.auth.sign_up({
                "email": email,
                "password": password
            })
            if res.user:
                supabase.table("profiles").insert({
                    "id": res.user.id,
                    "email": email
                }).execute()
                st.success("Account created")
            else:
                st.error("Registration failed")

    st.stop()

# -----------------------------
# Logged In
# -----------------------------
user = st.session_state.user
st.sidebar.success(f"Logged in as {user.email}")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# -----------------------------
# Vector Store
# -----------------------------
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

# -----------------------------
# Upload PDF
# -----------------------------
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

    st.sidebar.success("Material indexed")

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

mode = st.sidebar.selectbox(
    "Mode",
    ["Explain", "Quiz", "Exam"]
)

# -----------------------------
# Main Interaction
# -----------------------------
question = st.text_area("Ask your question")

if st.button("Submit") and vectorstore:

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

    # -----------------------------
    # Save Progress to Supabase
    # -----------------------------
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

    st.info(f"Next review scheduled: {next_review}")
