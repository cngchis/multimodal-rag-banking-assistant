# src/ingestion/csv_loader.py
import pandas as pd
from langchain_core.documents import Document
from src.vectorstore.pinecone_store import get_vectorstore
from src.utils.helper import get_env

def ingest_csv():
    csv_dir = get_env("CSV_DIR")

    # 1. Read CSV
    df = pd.read_csv(csv_dir)
    print(f"Loaded {len(df)} rows from {csv_dir}")

    # 2. Convert each row → Document
    docs = []
    for idx, row in df.iterrows():
        content = " | ".join(f"{col}: {row[col]}" for col in df.columns)
        docs.append(Document(
            page_content=content,
            metadata={"source": csv_dir, "row": idx}
        ))

    # 3. Upsert to Pinecone
    vectorstore = get_vectorstore()
    vectorstore.add_documents(docs)
    print(f"Ingested {len(docs)} documents from CSV into Pinecone")

if __name__ == "__main__":
    ingest_csv()