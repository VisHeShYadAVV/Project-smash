# modules/vector_store.py

import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from modules.llm_setup import get_embeddings_model
from modules.file_handler import extract_text_from_doc  # ✅ FIXED: missing import

async def build_vector_store(file_bytes: bytes, file_name: str):
    print("📥 [VECTOR] Building vector store")

    # Extract text per page using file_handler.py
    extracted_pages = extract_text_from_doc(file_bytes, file_name)
    if not extracted_pages:
        raise ValueError("❌ No text extracted from document.")

    documents = []
    for page_num, page_text in extracted_pages.items():
        documents.append(Document(page_content=page_text, metadata={"page": page_num}))

    print(f"📄 [VECTOR] {len(documents)} pages extracted")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"✂️ [VECTOR] {len(chunks)} chunks generated")

    embeddings = get_embeddings_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ [VECTOR] Vector store created")

    return vectorstore
