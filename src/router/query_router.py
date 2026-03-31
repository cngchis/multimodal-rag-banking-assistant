from src.utils.helper import get_llm_response, log_node

ROUTER_PROMPT = """
You are a routing agent. Based on the user query, decide where to look for information.

Options:
- Retrieve_QnA: If it's a Techcombank customer service question that can be answered from a QnA collection, choose this.
- Web_Search: If it's a general question or requires up-to-date information, choose this.
- Chitchat: Any question that is not related to Techcombank or is a casual conversation.
Query: "{query}"

Respond ONLY with one of: Retrieve_QnA, Web_Search, Chitchat. No explanation.
"""

def route_query(query: str) -> str:
    decision = get_llm_response(ROUTER_PROMPT.format(query=query)).strip()
    print(f"[Router] → {decision}")
    return decision