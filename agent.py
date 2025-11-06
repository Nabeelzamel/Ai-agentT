"""LangGraph Agent with Streamlit Support (Groq)"""

import os
import requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from duckduckgo_search import DDGS
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
# from langchain.chains import RetrievalQA

# -----------------------------
# Setup RAG
# -----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load documents 
if os.path.exists("knowledge.txt"):
    loader = TextLoader("knowledge.txt", encoding="utf-8")
    documents = loader.load()
else:
    documents = []

# Build vectorstore if docs exist
if documents:
    vectorstore = FAISS.from_documents(documents, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    vectorstore, retriever = None, None


# Load environment variables
load_dotenv()

# -----------------------------
# Define Tools
# -----------------------------
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers (avoid division by zero)."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool("web_search", return_direct=True)
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

        if not results:  # empty list
            return "No results found."

        print("🔍 Web search results:", results)
        return "\n".join([f"{r['title']}: {r['href']}" for r in results])
    

# qa_chain = RetrievalQA.from_chain_type(
#     llm=ChatGroq(model="openai/gpt-oss-120b"),
#     retriever=retriever,
#     return_source_documents=False  
# )

# @tool("rag_search", return_direct=True)
# def rag_search(query: str) -> str:
#     """Retrieve concise answers from the local knowledge base using RAG."""
#     if retriever is None:
#         return ""  


#     answer = qa_chain.run(query)
#     return answer if answer else ""



tools = [add, subtract, multiply, divide, web_search]


# -----------------------------
# Load System Prompt
# -----------------------------
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()
sys_msg = SystemMessage(content=system_prompt)

# -----------------------------
# Build Graph (Groq only)
# -----------------------------
def build_graph():
    """Build the agent graph with Groq"""
    llm = ChatGroq(
        model="openai/gpt-oss-120b",  
        temperature=0
    )

    # # Bind tools
    llm_with_tools = llm.bind_tools(tools)


    def assistant(state: MessagesState):

        messages = [sys_msg] + state["messages"]
        user_input = state["messages"][-1].content

        # 1️⃣ Try knowledge base first
        # kb_answer = rag_search(user_input)
        # if kb_answer:
        #     print("📚 Retrieved from knowledge base:", kb_answer)
        #     # Prepend the KB answer as context for the LLM
        #     kb_context_message = SystemMessage(
        #         content=f"Context from knowledge base:\n{kb_answer}"
        #     )
        #     messages = [sys_msg, kb_context_message] + state["messages"]

        # 2️⃣ Otherwise, use LLM (with tools)
        response = llm_with_tools.invoke(messages)
        print("🤖 Raw model response:", response)

        # Handle tool calls
        tool_messages = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                tool_id = tool_call["id"]
                for t in tools:
                    if t.name == tool_name:
                        tool_result = t.invoke(args)
                        tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))
                        print(f"🛠️ Ran tool {tool_name}: {tool_result}")
            return {"messages": [response] + tool_messages}

        return {"messages": [response]}


    # Build the state graph
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.set_entry_point("assistant")
    builder.set_finish_point("assistant")
    return builder.compile()
