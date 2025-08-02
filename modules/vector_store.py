# modules/vector_store.py

import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from modules.llm_setup import get_embeddings_model
from modules.file_handler import extract_text_from_doc

async def build_vector_store(file_bytes: bytes, file_name: str):
    print("📥 [VECTOR] Building vector store")

    extracted_pages = extract_text_from_doc(file_bytes, file_name)
    if not extracted_pages:
        raise ValueError("❌ No text extracted from document.")

    documents = [
        Document(page_content=page_text, metadata={"page": page_num})
        for page_num, page_text in extracted_pages.items()
    ]

    print(f"📄 Extracted {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"✂️ Generated {len(chunks)} chunks")

    embeddings = get_embeddings_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("✅ Vector store created")
    return vectorstore
