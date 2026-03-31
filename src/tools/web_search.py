from tavily import TavilyClient
from langchain_core.documents import Document
from src.utils.helper import get_env

def get_tavily_client() -> TavilyClient:
    return TavilyClient(api_key=get_env("TAVILY_API_KEY"))

def web_search(query: str, k: int = 3) -> list[Document]:
    """
    web search and return list Document
    to format same with similarity_search of Pinecone.
    """
    client = get_tavily_client()
    results = client.search(
        query=query,
        max_results=k,
        search_depth="advanced"
    )

    docs = []
    for r in results["results"]:
        docs.append(Document(
            page_content=r["content"],
            metadata={
                "source": r["url"],
                "title":  r.get("title", "")
            }
        ))

    return docs