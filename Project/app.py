import os
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import streamlit as st

load_dotenv("C:/Users/my pc/Desktop/Agentic_AI/Learning/Langgraph/Workflow/.env")

st.set_page_config(page_title="College Assistant", page_icon="🎓", layout="centered")

# STEP 1 -> Building The RAG

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

#Retriever
@st.cache_resource  # (added) build each PDF index once, not on every Streamlit rerun
def build_retriever(pdf_path : str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100
    )

    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks,embeddings)

    return vectorstore.as_retriever(search_kwargs = {"k" : 4})

academic_retriever=build_retriever("Project/academics_handbook.pdf")
fee_retriever = build_retriever("Project/fee_structure.pdf")

llm = ChatGroq(model="openai/gpt-oss-20b",temperature=0.3)

# Step 2 -> creating the STATE

class State(TypedDict):
    programme : str
    messages : Annotated[list,add_messages]
    query_type : str
    retrieved_context : str

# Nodes

def classifier_node(state : State) -> dict:
    """Look at the latest human message and decide which path to take."""

    last_message = state["messages"][-1].content
    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type":category}

def academic_rag_node(state: State) ->dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state['messages'][-1].content
    docs = academic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def fee_rag_node(state: State) ->dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state['messages'][-1].content
    docs= fee_retriever.invoke(query)
    context = '\n\n'.join([doc.page_content for doc in docs])
    return {"retrieved_context": context}
def general_node(state: State) -> dict:
    """Answer's directly using the LLm's own knowledge, no retireval needed"""
    return {"retrieved_context" : "NO_RETRIEVAL_NEEDED"}

def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state['messages'][-1].content
    programme = state.get('programme','Unknown')
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt=(
            f"you are a friendly college assistant talking to a {programme} student."
            f"Answer this question using your own general knowledge:\n\n {query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )
    response=llm.invoke(prompt)
    return{"messages": [("ai",response.content.strip())]}

# Router Function
def route_query(state: State) :
    if state['query_type']=='academic':
        return "academic_rag_node"
    elif state['query_type'] == "fee":
        return "fee_rag_node"
    else:
        return "general_node"

# building the graph
graph = StateGraph(State)

graph.add_node("classifier_node",classifier_node)
graph.add_node("academic_rag_node",academic_rag_node)
graph.add_node("fee_rag_node",fee_rag_node)
graph.add_node("general_node",general_node)
graph.add_node("response_node",response_node)

#edges
graph.add_edge(START,"classifier_node")
graph.add_conditional_edges(
    "classifier_node",route_query
    )
graph.add_edge("academic_rag_node","response_node")
graph.add_edge("fee_rag_node","response_node")
graph.add_edge("general_node","response_node")

graph.add_edge("response_node",END)

#compile
app = graph.compile()

# run the code  ->  Streamlit UI (replaces the input()/print() loop)

programme_map={
    "1" : "BCA",
    "2" : "BBA",
    "3" : "B.com (H)"
}

# Sidebar: choose programme
with st.sidebar:
    st.header("🎓 College Assistant")
    student_programme = st.radio("Which programme are you in?", list(programme_map.values()))
    st.caption(f"You're set as a **{student_programme}** student.")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

st.title("Welcome to the College Assistant")
st.caption("Ask about academics, fees, or anything else.")

if "chat" not in st.session_state:
    st.session_state.chat = []

# Show history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("route"):
            st.caption(f"Route: {msg['route']}")

# Chat input
user_query = st.chat_input("Type your question...")

if user_query:
    st.session_state.chat.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result=app.invoke(
                {
                    "programme":student_programme,
                    "messages" :[("human",user_query)]
                }
            )
        answer = result["messages"][-1].content
        route = result.get("query_type", "")
        st.markdown(answer)
        st.caption(f"Route: {route}")

    st.session_state.chat.append({"role": "assistant", "content": answer, "route": route})