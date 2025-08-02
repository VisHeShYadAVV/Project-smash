# api.py

import os
import httpx
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Sanitize API keys
OPENAI_API_KEY_RAW = os.getenv("OPENAI_API_KEY", "")
AUTH_TOKEN_RAW = os.getenv("HACKBOX_AUTH_TOKEN", "")
OPENAI_API_KEY = OPENAI_API_KEY_RAW.strip().lstrip("=").replace("\n", "")
AUTH_TOKEN = AUTH_TOKEN_RAW.strip()

# Validate
if not AUTH_TOKEN:
    raise RuntimeError("❌ HACKBOX_AUTH_TOKEN is not set or cleaned properly!")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY is not set!")

# Set OpenAI key globally
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Security setup
security_scheme = HTTPBearer(auto_error=False)

# Imports after env is loaded
from modules.vector_store import build_vector_store
from modules.retriever_chain import get_answers_as_json

# API models
class HackRXRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

# FastAPI app
app = FastAPI()

def validate_token(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    if not credentials:
        logging.warning("❌ Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    incoming_token = credentials.credentials.strip()
    expected_token = AUTH_TOKEN

    if credentials.scheme != "Bearer" or incoming_token != expected_token:
        logging.warning("❌ Bearer token mismatch!")
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")

    return True

@app.post("/api/v1/hackrx/run", response_model=HackRXResponse)
async def process_document_and_answer(
    request: HackRXRequest,
    is_authenticated: bool = Security(validate_token)
):
    try:
        print("📥 Fetching document:", request.documents)
        async with httpx.AsyncClient() as client:
            response = await client.get(str(request.documents))
            response.raise_for_status()
            file_bytes = response.content

        parsed_url = urlparse(str(request.documents))
        file_name = os.path.basename(parsed_url.path)
        if not file_name.endswith((".pdf", ".docx")):
            raise HTTPException(status_code=400, detail="Unsupported file type")

        print("📄 Document downloaded:", file_name)
        vector_store = await build_vector_store(file_bytes, file_name)

        print("💬 Getting answers from RAG chain...")
        json_response = await get_answers_as_json(request.questions, vector_store)

        print("✅ Answers:", json_response["answers"])
        return HackRXResponse(answers=json_response["answers"])

    except httpx.RequestError as e:
        logging.error(f"❌ Document download error: {e}")
        raise HTTPException(status_code=400, detail="Could not fetch document from URL.")

    except Exception as e:
        logging.exception("💥 Unexpected internal error")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# Dev only
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
