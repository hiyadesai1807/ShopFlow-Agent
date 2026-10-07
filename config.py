"""
Settings shared by every step.
If you want to change a model or a folder name, change it here only.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

# Reads GOOGLE_API_KEY from the .env file so the LLM can be called.
load_dotenv()

# Hides a harmless warning that Google's library prints on every run.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# ---- Names you may want to change -------------------------------------------

# The LLM (the model that writes answers). Called through Google's API.
CHAT_MODEL = "gemini-3.7-flash"

# The embedding model (turns text into numbers). Called through the same Google API key.
EMBEDDING_MODEL = "gemini-embedding-001"

# ---- Folders and files ------------------------------------------------------

PROJECT_FOLDER = Path(__file__).parent
DATA_FOLDER = PROJECT_FOLDER / "data"
DB_FOLDER = PROJECT_FOLDER / "chroma_db"   # the vector database is saved here
COLLECTION_NAME = "voltkart"


# ---- Small helper functions used by the steps -------------------------------

def get_llm():
    """Returns the LLM. temperature=0 makes answers steady instead of creative."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)


def get_embeddings():
    """Returns the embedding model. Uses the same API key as the LLM."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store():
    """Opens the vector database that step2_build_index.py created."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(DB_FOLDER),
    )
