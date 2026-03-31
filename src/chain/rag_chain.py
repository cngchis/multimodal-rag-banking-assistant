from src.utils.helper import get_llm_response, format_docs
from src.vectorstore.pinecone_store import similarity_search

RELEVANCE_PROMPT = """
Check the context below to see if the context is relevant to user query or not.
####
Context:
{context}
####
User Query: {query}

Options:
- Yes: if the context is relevant
- No: if the context is not relevant

Please answer with only Yes or No.
"""

AUGMENT_PROMPT = """You are a helpful assistant. You must answer by vietnamese. Use the context below to answer the user's question.
Source: {source}
Context:
{context}
Question: {query}
Answer:"""

def retrieve_context(query: str) -> tuple[str, str]:
    """Trả về (context, source)."""
    docs = similarity_search(query, k=3)
    context = format_docs(docs)
    return context, "Techcombank Support QA Collection"

def check_relevance(query: str, context: str) -> str:
    """Trả về 'Yes' hoặc 'No'."""
    result = get_llm_response(
        RELEVANCE_PROMPT.format(query=query, context=context)
    ).strip()
    print(f"[Relevance] → {result}")
    return result

def build_augmented_prompt(query: str, context: str, source: str) -> str:
    return AUGMENT_PROMPT.format(query=query, context=context, source=source)

def generate_answer(augmented_prompt: str) -> str:
    return get_llm_response(augmented_prompt)