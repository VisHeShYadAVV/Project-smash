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

# --- Load Environment Variables ---
load_dotenv()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip().replace("\n", "")
AUTH_TOKEN = (os.getenv("HACKBOX_AUTH_TOKEN") or "").strip().replace("\n", "")

if not AUTH_TOKEN:
    raise RuntimeError("❌ HACKBOX_AUTH_TOKEN is not set in environment variables")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY is not set in environment variables")

# Set as env var for libraries like OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Debug logs (can comment these in production) ---
print("✅ Loaded AUTH_TOKEN:", repr(AUTH_TOKEN))
print("✅ Loaded OPENAI_API_KEY:", repr(OPENAI_API_KEY[:10]) + "...")


# --- Security Setup ---
security_scheme = HTTPBearer(auto_error=False)

# --- Your modules ---
from modules.vector_store import build_vector_store
from modules.retriever_chain import get_answers_as_json

# --- Pydantic Models ---
class HackRXRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

# --- FastAPI App ---
app = FastAPI()

# --- Token Validation ---
def validate_token(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    incoming_token = credentials.credentials.strip().replace("\n", "")
    expected_token = AUTH_TOKEN

    if credentials.scheme != "Bearer" or incoming_token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")

    return True

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=HackRXResponse)
async def process_document_and_answer(
    request: HackRXRequest,
    is_authenticated: bool = Security(validate_token)
):
    try:
        # Step 1: Download Document
        async with httpx.AsyncClient() as client:
            response = await client.get(str(request.documents))
            response.raise_for_status()
            file_bytes = response.content

        # Step 2: Extract filename
        parsed_url = urlparse(str(request.documents))
        file_name = os.path.basename(parsed_url.path) or "input.pdf"
        if not file_name.endswith((".pdf", ".docx")):
            file_name = "input.pdf"

        # Step 3: Vectorize + Answer
        vector_store = await build_vector_store(file_bytes, file_name)
        json_response = await get_answers_as_json(request.questions, vector_store)

        return HackRXResponse(answers=json_response["answers"])

    except httpx.RequestError as e:
        logging.error(f"📄 Document download error: {e}")
        raise HTTPException(status_code=400, detail=f"Could not access document at URL: {request.documents}")

    except Exception as e:
        logging.exception("💥 Internal error during /hackrx/run")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# --- Run Locally ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
