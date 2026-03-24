from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from app.prompts import GENERATE_PROMPT, ANSWER_GRADER_PROMPT
from app.retriever import retrieve_documents
from app.llm import VertexLLM


# ---------------------------------------------------------------------------
# 1. AGENT STATE
# This is the shared memory that flows between every node in the graph.
# Every node reads from it and writes back to it.
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str               # The user's original question
    documents: list[Document]   # Retrieved context documents
    answer: str                 # The generated answer
    retries: int                # How many times we've retried


# ---------------------------------------------------------------------------
# 2. LLM SETUP — Real Gemini 1.5 Flash via Vertex AI SDK
# ---------------------------------------------------------------------------
llm = VertexLLM()


# ---------------------------------------------------------------------------
# 3. NODES — each node is a Python function that takes state, returns state
# ---------------------------------------------------------------------------

def retrieve(state: AgentState) -> AgentState:
    """
    Node 1: Retrieve relevant document chunks from the vector store.
    Uses semantic similarity search — finds chunks closest in meaning to the question.
    """
    print(f"[retrieve] Searching for: {state['question']}")
    docs = retrieve_documents(state["question"], k=3)
    return {**state, "documents": docs}


def generate(state: AgentState) -> AgentState:
    """
    Node 2: Generate an answer using the LLM + retrieved documents.
    """
    print(f"[generate] Generating answer (attempt {state.get('retries', 0) + 1})")

    # Format documents into a single context string
    context = "\n\n---\n\n".join(doc.page_content for doc in state["documents"])

    # Format the prompt into messages, then call LLM directly
    messages = GENERATE_PROMPT.format_messages(context=context, question=state["question"])
    response = llm.invoke(messages)

    return {**state, "answer": response.content, "retries": state.get("retries", 0) + 1}


def check_answer(state: AgentState) -> str:
    """
    Edge function (not a node): Decides what happens next.
    Returns END if the answer is good, "retrieve" if we should retry.
    LangGraph uses the return value to route to the next node.
    """
    print("[check_answer] Grading answer...")

    # Hard stop: don't loop more than 2 times to avoid infinite loops + cost
    if state.get("retries", 0) >= 2:
        print("[check_answer] Max retries reached. Accepting answer.")
        return END

    # Format the grader prompt, then call LLM directly
    messages = ANSWER_GRADER_PROMPT.format_messages(
        question=state["question"],
        answer=state["answer"],
    )
    result = llm.invoke(messages)

    grade = result.content.strip().lower()
    print(f"[check_answer] Grade: {grade}")

    if "yes" in grade:
        return END         # Answer is good → finish
    else:
        return "retrieve"  # Answer is bad → loop back and try again


# ---------------------------------------------------------------------------
# 4. GRAPH ASSEMBLY — wire nodes and edges together
# ---------------------------------------------------------------------------
def build_agent():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    # Set entry point
    graph.set_entry_point("retrieve")

    # Fixed edge: retrieve always goes to generate
    graph.add_edge("retrieve", "generate")

    # Conditional edge: after generate, check_answer decides what's next
    graph.add_conditional_edges(
        "generate",
        check_answer,
        {
            END: END,           # "yes" → end the graph
            "retrieve": "retrieve",  # "no" → loop back
        },
    )

    return graph.compile()


# Build the agent once at startup (not on every request)
agent = build_agent()
