# app/nodes.py
from typing import TypedDict
from src.router.query_router import route_query
from src.chain.rag_chain import retrieve_context, check_relevance, build_augmented_prompt
from src.tools.web_search import web_search
from src.utils.helper import log_node, format_docs, get_llm_response
from app.state import GraphState

def node_router(state: GraphState) -> GraphState:
    log_node("Router", state)
    decision = route_query(state["query"])
    return {**state, "source": decision}

def node_retrieve_qna(state: GraphState) -> GraphState:
    log_node("Retrieve_QnA", state)
    context, source = retrieve_context(state["query"])
    return {**state, "context": context, "source": source}

def node_web_search(state: GraphState) -> GraphState:
    log_node("Web_Search", state)
    docs = web_search(state["query"], k=1)
    context = format_docs(docs)
    return {**state, "context": context, "source": "Web Search"}

def node_check_relevance(state: GraphState) -> GraphState:
    log_node("Relevant_Context_Checker", state)
    is_relevant = check_relevance(state["query"], state["context"])
    return {**state, "is_relevant": is_relevant}

def node_augment(state: GraphState) -> GraphState:
    log_node("Augment", state)
    prompt = build_augmented_prompt(
        state["query"], state["context"], state["source"]
    )
    return {**state, "augmented_prompt": prompt}

def node_generate(state: GraphState) -> GraphState:
    log_node("Generate", state)
    answer = get_llm_response(
        state["augmented_prompt"],
        query=state["query"],
        session_id=state["session_id"]  # ← truyền session_id
    )
    return {**state, "answer": answer}

def node_chitchat(state: GraphState) -> GraphState:
    log_node("Chitchat", state)
    CHITCHAT_PROMPT = f"""/no_think
Bạn là trợ lý ảo Techcombank. Trả lời ngắn gọn bằng tiếng Việt, tối đa 2 câu.
Nếu user hỏi tóm tắt hoặc lịch sử → dựa vào lịch sử hội thoại để trả lời.
Nếu không liên quan ngân hàng → trả lời lịch sự rồi hỏi cần hỗ trợ gì về Techcombank.

Câu hỏi: {state["query"]}
Trả lời:"""
    answer = get_llm_response(
        CHITCHAT_PROMPT,
        query=state["query"],
        session_id=state["session_id"]  # ← truyền session_id
    )
    return {**state, "answer": answer}