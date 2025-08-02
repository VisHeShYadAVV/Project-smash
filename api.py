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

load_dotenv()

OPENAI_API_KEY_RAW = os.getenv("OPENAI_API_KEY", "")
AUTH_TOKEN_RAW = os.getenv("HACKBOX_AUTH_TOKEN", "")

OPENAI_API_KEY = OPENAI_API_KEY_RAW.strip().lstrip("=").replace("\n", "")
AUTH_TOKEN = AUTH_TOKEN_RAW.strip()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if not AUTH_TOKEN:
    raise RuntimeError("❌ HACKBOX_AUTH_TOKEN is not set or cleaned properly!")

security_scheme = HTTPBearer(auto_error=False)

from modules.vector_store import build_vector_store
from modules.retriever_chain import get_answers_as_json
class HackRXRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

app = FastAPI()

def validate_token(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    if not credentials:
        logging.warning("❌ Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    incoming_token = credentials.credentials.strip()
    expected_token = AUTH_TOKEN

    print("🔒 Incoming Bearer Token:", repr(incoming_token))
    print("🗝️  Expected Bearer Token:", repr(expected_token))

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
            file_name = "input.pdf"

        print("📄 Document downloaded, name:", file_name)
        print("⚙️  Invoking build_vector_store()")

        vector_store = await build_vector_store(file_bytes, file_name)

        print("💬 Getting answers from retriever_chain")
        json_response = await get_answers_as_json(request.questions, vector_store)

        print("✅ Answers generated:", json_response["answers"])
        return HackRXResponse(answers=json_response["answers"])

    except httpx.RequestError as e:
        logging.error(f"📄 Document download error: {e}")
        raise HTTPException(status_code=400, detail=f"Could not access document at URL: {request.documents}")

    except Exception as e:
        logging.exception("💥 Internal error during /hackrx/run")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# --- Dev Runner ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
