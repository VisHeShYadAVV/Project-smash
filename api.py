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

from modules.vector_store import build_vector_store
from modules.retriever_chain import get_answers_as_json

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
AUTH_TOKEN = os.getenv("HACKBOX_AUTH_TOKEN", "").strip()

# Validate env values
if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
    raise RuntimeError("❌ OPENAI_API_KEY is missing or invalid!")

if not AUTH_TOKEN:
    raise RuntimeError("❌ HACKBOX_AUTH_TOKEN is not set!")

# Set the API key for OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Logging config
logging.basicConfig(level=logging.INFO)

# FastAPI and Security
app = FastAPI()
security = HTTPBearer(auto_error=False)

# Request and Response models
class HackRXRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

# Token validation function
def validate_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not credentials or credentials.scheme != "Bearer":
        logging.warning("🔒 Missing or invalid Authorization header")
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    incoming_token = credentials.credentials.strip()
    if incoming_token != AUTH_TOKEN:
        logging.warning("🔑 Invalid token provided")
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")

    return True

# Endpoint for processing PDF and answering questions
@app.post("/api/v1/hackrx/run", response_model=HackRXResponse)
async def process_document_and_answer(
    request: HackRXRequest,
    is_authenticated: bool = Security(validate_token)
):
    try:
        # Download document
        async with httpx.AsyncClient() as client:
            response = await client.get(str(request.documents))
            response.raise_for_status()
            file_bytes = response.content

        # Extract filename
        parsed_url = urlparse(request.documents)
        file_name = os.path.basename(parsed_url.path)
        if not file_name.endswith((".pdf", ".docx")):
            file_name = "input.pdf"

        # Build vector store and answer
        vector_store = await build_vector_store(file_bytes, file_name)
        json_response = await get_answers_as_json(request.questions, vector_store)

        return HackRXResponse(answers=json_response["answers"])

    except httpx.HTTPError as e:
        logging.error(f"📥 Failed to fetch document: {e}")
        raise HTTPException(status_code=400, detail="Could not access the provided document URL")

    except Exception as e:
        logging.exception("💥 Unexpected internal error")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# Local dev entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
