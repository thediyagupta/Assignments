from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import os
import httpx

from chunk import split_text

app = FastAPI(
    title="Text Chunker API",
    description="Split large texts into chunks using RecursiveCharacterTextSplitter, and query an LLM via Groq.",
    version="1.0.0",
)


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class ChunkRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to split into chunks.")
    chunk_size: Optional[int] = Field(50, ge=1, description="Max characters per chunk.")
    chunk_overlap: Optional[int] = Field(10, ge=0, description="Overlap between chunks.")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v, info):
        chunk_size = info.data.get("chunk_size", 50)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")
        return v


class LLMRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Your question for the LLM.")


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/")
def welcome():
    """Welcome endpoint."""
    return {"message": "Welcome to the Text Chunker API! Visit /docs for the interactive UI."}


@app.post("/chunk")
def chunk_text(request: ChunkRequest):
    """
    Split a large text into chunks.

    - **text**: The text you want to split (required).
    - **chunk_size**: Max characters per chunk (default 50).
    - **chunk_overlap**: Overlap between consecutive chunks (default 10).
    """
    try:
        result = split_text(
            text=request.text,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    return {
        "chunks": result["chunks"],
        "total_chunks": result["total_chunks"],
        "chunk_size_used": request.chunk_size,
        "chunk_overlap_used": request.chunk_overlap,
    }


@app.post("/ask")
async def ask_llm(request: LLMRequest):
    """
    Send a query to the Groq LLM (llama3-8b-8192) and get an answer.

    Requires the GROQ_API_KEY environment variable to be set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY environment variable is not set on the server.",
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": request.query}],
        "max_tokens": 512,
        "temperature": 0.7,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Groq API error: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Groq: {str(e)}")

    return {"query": request.query, "answer": answer}


# ──────────────────────────────────────────────
# Run locally: python main.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
