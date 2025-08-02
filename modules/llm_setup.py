# modules/llm_setup.py

from langchain_openai import OpenAIEmbeddings
import os

def get_embeddings_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",  # or "text-embedding-ada-002"
        openai_api_key=os.getenv("OPENAI_API_KEY", "")
    )
