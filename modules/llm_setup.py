# modules/llm_setup.py
from langchain_openai import OpenAIEmbeddings

def get_embeddings_model():
    """Initializes and returns the OpenAI embeddings model."""
    return OpenAIEmbeddings(model="text-embedding-3-small")