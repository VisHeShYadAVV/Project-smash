import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

async def build_vector_store(file_bytes: bytes, file_name: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    if file_name.endswith(".pdf"):
        loader = PyMuPDFLoader(tmp_path)
    elif file_name.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(tmp_path)
    else:
        raise ValueError("Unsupported file format")

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    os.remove(tmp_path)
    return vector_store
