from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.vectorstore.chroma_store import get_vectorstore
from src.utils.helper import get_env

def ingest_pdfs():
    pdf_dir = get_env("PDF_DIR")

    # 1. Load all PDFs
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()

    # 2. Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # 3. Upsert to Pinecone
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks from CSV into Pinecone")

if __name__ == "__main__":
    ingest_pdfs()