import os
import uvicorn
import traceback
import logging
import time
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
AUTH_TOKEN = os.getenv("HACKBOX_AUTH_TOKEN", "").strip()

# Debugging .env variables
print("🔑 OPENAI_API_KEY:", repr(OPENAI_KEY))
print("🛡️ HACKBOX_AUTH_TOKEN:", repr(AUTH_TOKEN))

# Auth setup
security = HTTPBearer()
app = FastAPI()

# Import internal logic
from modules.vector_store import build_vector_store
from modules.retriever_chain import get_answers_as_json

# Pydantic models
class HackboxRequest(BaseModel):
    documents: str  # ✅ Changed from HttpUrl to str
    questions: List[str]

class HackboxResponse(BaseModel):
    answers: List[str]

@app.get("/ping")
def ping():
    return {"status": "ok", "message": "API is alive!"}

@app.post("/api/v1/hackrx/run", response_model=HackboxResponse)
async def process_request(
    request_data: HackboxRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        # Validate token
        bearer_token = credentials.credentials
        if bearer_token != AUTH_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid Bearer Token")
        print("✅ Authenticated request")

        total_start = time.time()

        # Step 1: Download document
        import httpx
        print(f"📥 Downloading document from {request_data.documents}")
        download_start = time.time()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(request_data.documents)
            response.raise_for_status()
            file_bytes = response.content
        print(f"✅ Document downloaded in {time.time() - download_start:.2f} sec")

        # Step 2: Infer file type
        ext = request_data.documents.split('.')[-1].split('?')[0].lower()
        file_name = f"temp_file.{ext}"
        print(f"📄 Inferred file type: {ext}")

        # Step 3: Build vector store
        vector_start = time.time()
        vector_store = await build_vector_store(file_bytes, file_name)
        print(f"✅ Vector store built in {time.time() - vector_start:.2f} sec")

        # Step 4: Get answers
        answer_start = time.time()
        answers = await get_answers_as_json(request_data.questions, vector_store)
        print(f"✅ Answers generated in {time.time() - answer_start:.2f} sec")

        print(f"✅ Total time: {time.time() - total_start:.2f} sec")

        return answers

    except httpx.HTTPError as e:
        print("❌ Error downloading document:", traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"❌ Error downloading document: {str(e)}")

    except Exception as e:
        print("❌ General error:", traceback.format_exc())
        raise HTTPException(status_code=502, detail=f"❌ Internal Server Error: {str(e)}")

# Entry point
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
