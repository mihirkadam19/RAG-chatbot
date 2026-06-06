"""
RAG Chatbot — FastAPI Backend
------------------------------
Receives a question from the frontend, embeds it via OpenAI,
retrieves the top matching chunks from Supabase, and generates
an answer using gpt-4o-mini.

Setup:
  pip install fastapi uvicorn openai supabase python-dotenv

Run:
  uvicorn main:app --reload --port 8000

Environment variables (.env file or export):
  OPENAI_API_KEY
  SUPABASE_URL
  SUPABASE_KEY
"""

import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from supabase import create_client

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("chatbot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
SUPABASE_URL   = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY   = os.environ.get("SUPABASE_KEY", "")

EMBED_MODEL  = "text-embedding-3-small"
CHAT_MODEL   = "gpt-4o-mini"
TOP_K        = 5   # number of chunks to retrieve per question

SYSTEM_PROMPT = """You are a helpful training assistant for student staff.
Answer the user's question using ONLY the training material provided below.
If the answer is not covered in the material, say so clearly — do not guess.
Be concise, friendly, and use plain language."""

# ── startup checks ────────────────────────────────────────────────────────────

missing = [k for k in ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
           if not os.environ.get(k)]
if missing:
    log.error("Missing environment variables: %s", ", ".join(missing))
    sys.exit(1)

oc = OpenAI(api_key=OPENAI_API_KEY)
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

log.info("All clients initialised successfully.")

# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Training Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this when you deploy
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]   # filenames of retrieved chunks


# ── helpers ───────────────────────────────────────────────────────────────────

def embed_question(question: str) -> list[float]:
    log.info("Embedding question: %r", question)
    resp = oc.embeddings.create(model=EMBED_MODEL, input=[question])
    return resp.data[0].embedding


def retrieve_chunks(embedding: list[float]) -> list[dict]:
    log.info("Querying Supabase for top %d chunks.", TOP_K)
    result = sb.rpc(
        "match_chunks",
        {"query_embedding": embedding, "match_count": TOP_K},
    ).execute()
    chunks = result.data or []
    log.info("Retrieved %d chunk(s). Sources: %s",
             len(chunks), [c["source"] for c in chunks])
    return chunks


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['content']}")
    return "\n\n".join(parts)


def ask_openai(question: str, context: str) -> str:
    log.info("Sending request to OpenAI (%s).", CHAT_MODEL)
    user_message = f"Training material:\n\n{context}\n\n---\n\nQuestion: {question}"
    response = oc.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )
    answer = response.choices[0].message.content
    log.info("OpenAI responded (%d chars).", len(answer))
    return answer


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    log.info("Health check.")
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    log.info("── New question ──────────────────────────────")
    log.info("Question: %r", question)

    try:
        embedding = embed_question(question)
        chunks    = retrieve_chunks(embedding)

        if not chunks:
            log.warning("No relevant chunks found for question: %r", question)
            return AnswerResponse(
                answer="I couldn't find anything relevant in the training materials.",
                sources=[],
            )

        context = build_context(chunks)
        answer  = ask_openai(question, context)
        sources = list(dict.fromkeys(c["source"] for c in chunks))  # deduplicated

        log.info("Response sent. Sources used: %s", sources)
        return AnswerResponse(answer=answer, sources=sources)

    except Exception as e:
        log.exception("Unexpected error while handling question: %r", question)
        raise HTTPException(status_code=500, detail=str(e))