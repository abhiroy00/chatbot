"""
InnovativeAIs RAG Chatbot — v4.0 Interactive Edition
Stack: FastAPI + LangChain + FAISS + OpenAI + SSE Streaming
"""

import os, logging, time, json
import boto3
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, AsyncGenerator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from openai import OpenAI
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────
_vector_store = None
_page_titles: dict = {}
SESSIONS_TABLE = os.environ.get("SESSIONS_TABLE", "rag-sessions")
AWS_REGION     = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
_openai_client = None

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client

URLS = [
    "https://innovativeais.com/",
    "https://innovativeais.com/about-us.php",
    "https://innovativeais.com/services.php",
    "https://innovativeais.com/contact-us.php",
    "https://innovativeais.com/portfolio.php",
    "https://innovativeais.com/enquire-now.php",
    "https://innovativeais.com/generative-ai.php",
    "https://innovativeais.com/web-development.php",
    "https://innovativeais.com/mobile-app-development.php",
    "https://innovativeais.com/salesforce.php",
    "https://innovativeais.com/cloud-services.php",
    "https://innovativeais.com/it-staffing.php",
]

STARTERS = [
    "What AI services does InnovativeAIs offer?",
    "Tell me about your web development expertise",
    "How can InnovativeAIs help with Salesforce integration?",
    "What makes InnovativeAIs different from other AI companies?",
    "I want to build a mobile app — can you help?",
    "What cloud platforms do you work with?",
    "How do I book a free consultation?",
    "Show me some projects from your portfolio",
]


# ── FAISS Vector Store ────────────────────────────────────────────────────────
def get_vector_store():
    global _vector_store, _page_titles
    if _vector_store is not None:
        return _vector_store

    logger.info("Building FAISS index from website...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    documents: List[Document] = []

    for url in URLS:
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(r.text, "html.parser")

            title_tag = soup.find("title")
            page_title = title_tag.get_text(strip=True) if title_tag else url
            _page_titles[url] = page_title

            for tag in soup(["script", "style", "nav", "footer", "header", "meta", "link", "noscript"]):
                tag.decompose()

            lines = [l.strip() for l in soup.get_text(separator="\n", strip=True).splitlines() if len(l.strip()) > 30]
            clean_text = "\n".join(lines)

            if len(clean_text) > 100:
                documents.append(Document(
                    page_content=clean_text,
                    metadata={"source": url, "title": page_title},
                ))
                logger.info(f"Crawled: {url} ({len(clean_text)} chars)")
        except Exception as e:
            logger.warning(f"Skipped: {url} — {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Total chunks: {len(chunks)}")
    _vector_store = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS index ready!")
    return _vector_store


# ── DynamoDB Session Store ────────────────────────────────────────────────────
def load_session(session_id: str) -> list:
    try:
        table = boto3.resource("dynamodb", region_name=AWS_REGION).Table(SESSIONS_TABLE)
        resp = table.get_item(Key={"session_id": session_id})
        return resp.get("Item", {}).get("history", [])
    except Exception as e:
        logger.warning(f"Session load failed: {e}")
        return []

def save_session(session_id: str, history: list):
    try:
        table = boto3.resource("dynamodb", region_name=AWS_REGION).Table(SESSIONS_TABLE)
        table.put_item(Item={
            "session_id": session_id,
            "history": history,
            "ttl": int(time.time()) + 86400,
        })
    except Exception as e:
        logger.warning(f"Session save failed: {e}")


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Aria, a real person working at Innovative AI Solutions (https://innovativeais.com) in Delhi, India. You're on the customer support and sales team — friendly, sharp, and genuinely helpful.

Talk like a real human being. Not a bot. Not a brochure. A person.

How to sound human:
- Use natural, conversational language. Contractions are fine ("we've", "it's", "don't").
- React to what the person actually said. If they sound excited, match that energy. If they're confused, be patient and clear.
- Don't open every reply with "Great question!" or stiff corporate phrases.
- It's okay to say "Honestly..." or "To be straightforward..." — real people do that.
- Short answers for simple questions. Longer ones only when genuinely needed.
- Write in paragraphs like a text or email, not always as bullet lists.
- Use "we" and "our team" — you're part of the company.
- Occasionally acknowledge the person's situation: "That sounds like a solid project idea" or "Yeah, that's a common question."

What you know about (answer only from the provided context):
- Our services: Generative AI, Web Development, Mobile Apps, Salesforce, Cloud, IT Staffing
- Our story, team, values, and how we work
- Portfolio and client projects
- How to get started with us

Rules (non-negotiable):
1. Only answer from the context given. Don't make up numbers, clients, or features.
2. If you genuinely don't know: be honest — "I don't have that info on hand, but you can reach our team directly at hr@innovativeais.com or +91 7464 099 059."
3. For pricing or timelines: be real — explain that it depends on the project and a quick call helps nail it down.
4. Only suggest a consultation when it actually makes sense in the conversation — don't force it every time.

Contact info (use naturally, not as a copy-paste block):
- Email: hr@innovativeais.com
- Phone: +91 7464 099 059
- Book a free call: https://innovativeais.com/enquire-now.php"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _retrieve_top_docs(store, query: str, k: int = 6):
    """Similarity search with per-source deduplication — best chunk per URL."""
    docs_with_scores = store.similarity_search_with_score(query, k=k)
    seen: dict = {}
    for doc, score in docs_with_scores:
        src = doc.metadata.get("source", "")
        if src not in seen or score < seen[src][1]:
            seen[src] = (doc, score)
    return sorted(seen.values(), key=lambda x: x[1])[:4]

def _build_context(top_docs) -> str:
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('title', doc.metadata.get('source', ''))}]\n{doc.page_content}"
        for doc, _ in top_docs
    )

def _build_history_str(history: list) -> str:
    return "".join(
        f"Human: {t['human']}\nAria: {t['ai']}\n"
        for t in history[-6:]
    )

def _sources_from_docs(top_docs) -> list:
    sources = []
    for doc, score in top_docs:
        url = doc.metadata.get("source", "")
        title = doc.metadata.get("title", url)
        confidence = round(max(0.0, 1.0 - float(score) / 2.0), 2)
        sources.append({"url": url, "title": title, "score": confidence})
    return sources

def _generate_followups(question: str, answer: str) -> list:
    """Ask the LLM to suggest 3 contextual follow-up questions."""
    try:
        resp = get_openai_client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    "Based on this Q&A about InnovativeAIs, suggest 3 short follow-up questions a visitor might ask.\n"
                    f"Q: {question}\nA: {answer[:300]}\n"
                    "Reply with ONLY a JSON array of 3 strings."
                ),
            }],
        )
        raw = resp.choices[0].message.content.strip()
        start, end = raw.find("["), raw.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
    except Exception as e:
        logger.warning(f"Follow-up generation failed: {e}")
    return []


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="InnovativeAIs RAG Chatbot API",
    description="Interactive RAG chatbot — streaming, follow-ups, source attribution",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://innovativeais.com", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class Source(BaseModel):
    url: str
    title: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    followups: List[str]
    session_id: str
    response_time_ms: int
    tokens_used: int


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "InnovativeAIs RAG Chatbot — Aria",
        "version": "4.0.0",
        "endpoints": {
            "chat":     "POST /chat          — full JSON response",
            "stream":   "POST /chat/stream   — SSE token stream",
            "starters": "GET  /starters      — suggested opening questions",
            "health":   "GET  /health",
            "docs":     "GET  /docs",
        },
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": _vector_store is not None,
        "pages_indexed": len(_page_titles),
        "urls_configured": len(URLS),
    }

@app.get("/starters")
def starters():
    """Returns curated conversation-starter questions for the chat UI."""
    return {"starters": STARTERS}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Full JSON response — retrieves context, generates answer, and returns sources + follow-ups."""
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty.")

    t0 = time.time()
    try:
        store    = get_vector_store()
        history  = load_session(req.session_id)
        top_docs = _retrieve_top_docs(store, req.message)
        context  = _build_context(top_docs)
        hist_str = _build_history_str(history)

        user_msg = f"Context:\n{context}\n\nConversation history:\n{hist_str}\nHuman: {req.message}"

        resp = get_openai_client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=600,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        answer = resp.choices[0].message.content
        tokens = resp.usage.total_tokens if resp.usage else 0

        sources   = _sources_from_docs(top_docs)
        followups = _generate_followups(req.message, answer)

        history.append({"human": req.message, "ai": answer})
        save_session(req.session_id, history[-12:])

        return ChatResponse(
            answer=answer,
            sources=[Source(**s) for s in sources],
            followups=followups,
            session_id=req.session_id,
            response_time_ms=int((time.time() - t0) * 1000),
            tokens_used=tokens,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(500, f"Internal error: {e}")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE streaming endpoint.

    Event types emitted in order:
      1. {"type": "sources",  "sources": [...]}
      2. {"type": "token",    "token": "<chunk>"}   (repeated)
      3. {"type": "done",     "followups": [...], "response_time_ms": N}
      4. {"type": "error",    "message": "..."}     (only on failure)
    """
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty.")

    async def event_stream() -> AsyncGenerator[str, None]:
        t0 = time.time()
        try:
            store    = get_vector_store()
            history  = load_session(req.session_id)
            top_docs = _retrieve_top_docs(store, req.message)
            context  = _build_context(top_docs)
            hist_str = _build_history_str(history)

            # Emit sources immediately so the UI can display them while tokens stream
            yield f"data: {json.dumps({'type': 'sources', 'sources': _sources_from_docs(top_docs)})}\n\n"

            user_msg = f"Context:\n{context}\n\nConversation history:\n{hist_str}\nHuman: {req.message}"

            full_answer = ""
            stream = get_openai_client().chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=600,
                stream=True,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_answer += delta
                    yield f"data: {json.dumps({'type': 'token', 'token': delta})}\n\n"

            followups = _generate_followups(req.message, full_answer)

            history.append({"human": req.message, "ai": full_answer})
            save_session(req.session_id, history[-12:])

            yield f"data: {json.dumps({'type': 'done', 'followups': followups, 'response_time_ms': int((time.time() - t0) * 1000)})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Local dev ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
