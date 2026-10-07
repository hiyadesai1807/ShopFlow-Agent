"""
STEP 2 - Turn the store's data into a searchable vector database.

Concepts:
- Document : one piece of text plus some labels (metadata).
- Chunking : cutting a long text into small pieces, so we can fetch only
             the piece that answers the question.
- Embedding: a list of numbers that represents the meaning of a text.
- Vector database (Chroma): stores the embeddings and finds the closest ones.

Run this once, and again whenever you change the files in data/.

Run:  python step2_build_index.py
"""

import shutil

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import COLLECTION_NAME, DATA_FOLDER, DB_FOLDER, get_embeddings

# ---- 1. Products: one Document per product ----------------------------------
# A product row is already short, so it does not need chunking.

products = pd.read_csv(DATA_FOLDER / "products.csv")

product_docs = []
for row in products.itertuples():
    text = (
        f"{row.name} by {row.brand}. Category: {row.category}. "
        f"Price: Rs. {row.price}. Rating: {row.rating} out of 5. {row.description}"
    )
    labels = {
        "source": "product",
        "product_id": row.product_id,
        "name": row.name,
        "category": row.category,
        "price": int(row.price),   # kept as a number so we can filter by budget
    }
    product_docs.append(Document(page_content=text, metadata=labels))

# ---- 2. Policies: one long text, so we cut it into chunks -------------------

policy_text = (DATA_FOLDER / "store_policies.txt").read_text(encoding="utf-8")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # each chunk is at most 500 characters
    chunk_overlap=80,   # neighbouring chunks share 80 characters, so a
                        # sentence cut at the edge is not lost
)
policy_docs = splitter.create_documents([policy_text], metadatas=[{"source": "policy"}])

print(f"Products : {len(product_docs)} documents")
print(f"Policies : {len(policy_docs)} chunks")
print("\nExample policy chunk:\n" + policy_docs[1].page_content)

# ---- 3. Embed everything and save it in Chroma ------------------------------

# Remove the old database first, otherwise every run adds duplicate copies.
if DB_FOLDER.exists():
    shutil.rmtree(DB_FOLDER)

vector_store = Chroma.from_documents(
    documents=product_docs + policy_docs,
    embedding=get_embeddings(),
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_FOLDER),
)
print(f"\nSaved {len(product_docs) + len(policy_docs)} items to {DB_FOLDER.name}/")

# ---- 4. Try a search: no LLM involved yet, only embeddings ------------------

question = "earphones for running"
print(f"\nClosest matches for: '{question}'")
for doc in vector_store.similarity_search(question, k=3):
    print(" -", doc.page_content[:90], "...")
