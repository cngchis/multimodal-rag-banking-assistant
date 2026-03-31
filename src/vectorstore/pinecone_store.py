from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from src.utils.helper import get_env

# ── Custom Embedding ────────────────────────────────
_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2")

class CustomEmbedding(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return _model.encode([text])[0].tolist()

INDEX_NAME = "techcombank-rag"
DIMENSION  = 1024

def _get_pinecone_index():
    pc = Pinecone(api_key=get_env("PINECONE_API_KEY"))
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=get_env("PINECONE_REGION")
            )
        )
        print(f"Created Pinecone index: {INDEX_NAME} (dim={DIMENSION})")
    return pc.Index(INDEX_NAME)

def get_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore(
        index=_get_pinecone_index(),
        embedding=CustomEmbedding(),
    )

def similarity_search(query: str, k: int = 3) -> list:
    return get_vectorstore().similarity_search(query, k=k)