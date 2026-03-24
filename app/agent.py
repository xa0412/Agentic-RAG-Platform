from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from app.prompts import GENERATE_PROMPT, ANSWER_GRADER_PROMPT
from app.retriever import retrieve_documents
from app.llm import VertexLLM


class AgentState(TypedDict):
    question: str
    documents: list[Document]
    answer: str
    retries: int


llm = VertexLLM()


def retrieve(state: AgentState) -> AgentState:
    print(f"[retrieve] Searching for: {state['question']}")
    docs = retrieve_documents(state["question"], k=3)
    return {**state, "documents": docs}


def generate(state: AgentState) -> AgentState:
    print(f"[generate] Generating answer (attempt {state.get('retries', 0) + 1})")
    context = "\n\n---\n\n".join(doc.page_content for doc in state["documents"])
    messages = GENERATE_PROMPT.format_messages(context=context, question=state["question"])
    response = llm.invoke(messages)
    return {**state, "answer": response.content, "retries": state.get("retries", 0) + 1}


def check_answer(state: AgentState) -> str:
    print("[check_answer] Grading answer...")

    # Cap retries to prevent infinite loops and runaway API costs
    if state.get("retries", 0) >= 2:
        print("[check_answer] Max retries reached. Accepting answer.")
        return END

    messages = ANSWER_GRADER_PROMPT.format_messages(
        question=state["question"],
        answer=state["answer"],
    )
    result = llm.invoke(messages)
    grade = result.content.strip().lower()
    print(f"[check_answer] Grade: {grade}")

    return END if "yes" in grade else "retrieve"


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges(
        "generate",
        check_answer,
        {END: END, "retrieve": "retrieve"},
    )
    return graph.compile()


agent = build_agent()
