from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from app.nodes import (
    node_router, node_retrieve_qna, node_web_search,
    node_check_relevance, node_augment, node_generate, node_chitchat
)
from app.state import GraphState


def route_decision(state: GraphState) -> str:
    return state["source"]

def relevance_decision(state: GraphState) -> str:
    count = state.get("iteration_count", 0) + 1
    state["iteration_count"] = count
    if count >= 2 or state.get("source") == "Web Search":
        return "Yes"
    return state["is_relevant"]

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("Router",                   node_router)
    g.add_node("Retrieve_QnA",             node_retrieve_qna)
    g.add_node("Web_Search",               node_web_search)
    g.add_node("Relevant_Context_Checker", node_check_relevance)
    g.add_node("Augment",                  node_augment)
    g.add_node("Generate",                 node_generate)
    g.add_node("Chitchat",                 node_chitchat)

    g.add_edge(START, "Router")
    g.add_conditional_edges("Router", route_decision, {
        "Retrieve_QnA": "Retrieve_QnA",
        "Web_Search":   "Web_Search",
        "Chitchat":     "Chitchat"
    })
    g.add_edge("Retrieve_QnA", "Relevant_Context_Checker")
    g.add_edge("Web_Search",   "Relevant_Context_Checker")
    g.add_conditional_edges("Relevant_Context_Checker", relevance_decision, {
        "Yes": "Augment",
        "No":  "Web_Search"
    })
    g.add_edge("Augment",  "Generate")
    g.add_edge("Generate", END)
    g.add_edge("Chitchat", END)

    return g.compile()