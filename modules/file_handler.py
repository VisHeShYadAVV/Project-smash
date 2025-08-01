# modules/file_handler.py

import fitz  
import docx
import io
import logging

def extract_text_from_doc(file_bytes: bytes, filename: str) -> dict:
    """Extracts text from PDF or DOCX files page by page."""
    file_ext = filename.lower().split('.')[-1]
    
    if file_ext == 'pdf':
        pages = {}
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                pages[i+1] = page.get_text()
        return pages
        
    elif file_ext == 'docx':
        with io.BytesIO(file_bytes) as docx_file:
            doc = docx.Document(docx_file)
            return {1: "\n".join([para.text for para in doc.paragraphs])}
            
    logging.error(f"Unsupported file type: {file_ext}")
    return None