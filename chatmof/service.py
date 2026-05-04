"""
ChatMOF HTTP service.

Exposes the ChatMOF agent as a REST endpoint so any client can send
natural-language questions about MOFs and receive answers.

Start (stub moftransformer backend, default):
    uvicorn chatmof.service:app --host 0.0.0.0 --port 8001

Start (real moftransformer):
    CHATMOF_MOFTRANSFORMER_BACKEND=local \\
        uvicorn chatmof.service:app --host 0.0.0.0 --port 8001

Start (remote moftransformer service):
    CHATMOF_MOFTRANSFORMER_BACKEND=remote \\
    CHATMOF_MOFTRANSFORMER_SERVER_URL=http://moftransformer-host:8000 \\
        uvicorn chatmof.service:app --host 0.0.0.0 --port 8001
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler

from chatmof.agents.agent import ChatMOF
from chatmof.config import config
from chatmof.moftransformer_api import get_api

app = FastAPI(
    title="ChatMOF Service",
    description="Natural-language interface for predicting and generating Metal-Organic Frameworks.",
    version="0.0.0",
)

# ── agent singleton ────────────────────────────────────────────────────────────
# Initialised once on startup so the LLM client and tools are reused across requests.

_agent: Optional[ChatMOF] = None


def _get_agent() -> ChatMOF:
    global _agent
    if _agent is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        llm = ChatOpenAI(temperature=config["temperature"], model="gpt-3.5-turbo")
        _agent = ChatMOF.from_llm(llm=llm, verbose=False, search_internet=False)
    return _agent


# ── schemas ────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    backend: str


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "moftransformer_backend": type(get_api()).__name__,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty.")
    try:
        answer = _get_agent().run(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return AskResponse(
        question=req.question,
        answer=answer,
        backend=type(get_api()).__name__,
    )
