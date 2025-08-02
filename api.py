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

# --- Load environment variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AUTH_TOKEN = os.getenv("HACKBOX_AUTH_TOKEN")

print("✅ Loaded AUTH_TOKEN:", repr(AUTH_TOKEN))  # Debug: shows if token has newline or space

if not AUTH_TOKEN:
    raise RuntimeError("❌ HACKBOX_AUTH_TOKEN is not set in environment variables")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""

# --- Security Setup ---
security_scheme = HTTPBearer(auto_error=False)

# --- Import your modules ---
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

# --- Token validation ---
def validate_token(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    if not credentials:
        logging.warning("❌ Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    incoming_token = credentials.credentials.strip()
    expected_token = AUTH_TOKEN.strip()

    print("🔐 Incoming token:", repr(incoming_token))
    print("🔐 Expected token:", repr(expected_token))

    if credentials.scheme != "Bearer" or incoming_token != expected_token:
        logging.warning(f"❌ Token mismatch!\nIncoming: {repr(incoming_token)}\nExpected: {repr(expected_token)}")
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")

    return True

# --- Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=HackRXResponse)
async def process_document_and_answer(
    request: HackRXRequest,
    is_authenticated: bool = Security(validate_token)
):
    try:
        # Step 1: Download the document
        async with httpx.AsyncClient() as client:
            response = await client.get(str(request.documents))
            response.raise_for_status()
            file_bytes = response.content

        # Step 2: Extract filename
        parsed_url = urlparse(str(request.documents))
        file_name = os.path.basename(parsed_url.path)
        if not file_name.endswith((".pdf", ".docx")):
            file_name = "input.pdf"

        # Step 3: Process document
        vector_store = await build_vector_store(file_bytes, file_name)
        json_response = await get_answers_as_json(request.questions, vector_store)

        return HackRXResponse(answers=json_response["answers"])

    except httpx.RequestError as e:
        logging.error(f"📄 Document download error: {e}")
        raise HTTPException(status_code=400, detail=f"Could not access document at URL: {request.documents}")

    except Exception as e:
        logging.exception("💥 Internal error during /hackrx/run")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# --- For local testing only ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
