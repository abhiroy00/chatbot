"""
InnovativeAIs RAG Chatbot — EC2 Docker Edition
Stack: FastAPI + LangChain + FAISS + OpenAI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os, logging, boto3, requests, time
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────
_vector_store = None
SESSIONS_TABLE = os.environ.get("SESSIONS_TABLE", "rag-sessions")
AWS_REGION     = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")

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

# ── FAISS Vector Store ────────────────────────────────────────────────────────
def get_vector_store():
    global _vector_store
    if _vector_store is None:
        logger.info("Building FAISS index from website...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        texts = []
        for url in URLS:
            try:
                r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                if len(text) > 100:
                    texts.append("Source: " + url + "\n\n" + text)
                    logger.info("Crawled: " + url)
            except Exception as e:
                logger.warning("Skipped: " + url + " - " + str(e))

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.create_documents(texts)
        logger.info("Total chunks: " + str(len(chunks)))
        _vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("FAISS index ready!")
    return _vector_store


# ── DynamoDB Session Store ────────────────────────────────────────────────────
def load_session(session_id: str) -> list:
    try:
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        table = dynamodb.Table(SESSIONS_TABLE)
        resp = table.get_item(Key={"session_id": session_id})
        return resp.get("Item", {}).get("history", [])
    except Exception as e:
        logger.warning("Session load failed: " + str(e))
        return []

def save_session(session_id: str, history: list):
    try:
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        table = dynamodb.Table(SESSIONS_TABLE)
        table.put_item(Item={
            "session_id": session_id,
            "history": history,
            "ttl": int(time.time()) + 86400,
        })
    except Exception as e:
        logger.warning("Session save failed: " + str(e))


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="InnovativeAIs RAG Chatbot API",
    description="Conversational RAG chatbot for InnovativeAIs",
    version="3.0.0",
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


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "InnovativeAIs RAG Chatbot API",
        "health": "/health",
        "chat": "POST /chat",
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {"status": "ok", "store_loaded": _vector_store is not None}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty.")
    try:
        store = get_vector_store()
        history = load_session(req.session_id)

        # Get relevant docs
        docs = store.similarity_search(req.message, k=4)
        context = "\n\n".join([d.page_content for d in docs])

        # Build history string
        history_str = ""
        for turn in history[-6:]:
            history_str += "Human: " + turn["human"] + "\nAssistant: " + turn["ai"] + "\n"

        system = (
            "You are the official AI assistant for Innovative AI Solutions (https://innovativeais.com), "
            "a software and AI company based in Delhi, India.\n\n"
            "You help visitors learn about:\n"
            "- Services (Web Dev, Mobile Apps, Generative AI, Salesforce, Cloud, IT Staffing)\n"
            "- Company background, leadership, portfolio, testimonials\n"
            "- How to get in touch or book a free consultation\n\n"
            "Rules:\n"
            "1. Answer ONLY from the provided context.\n"
            "2. If unsure, say: I am not sure about that, please contact us at hr@innovativeais.com or call +91 7464 099 059.\n"
            "3. Be friendly, concise, and professional.\n"
            "4. Always end sales-oriented questions with: Book a free consultation at https://innovativeais.com/enquire-now.php"
        )

        user_msg = "Context:\n" + context + "\n\nConversation:\n" + history_str + "\nHuman: " + req.message

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg}
            ]
        )
        answer = response.choices[0].message.content

        history.append({"human": req.message, "ai": answer})
        save_session(req.session_id, history[-12:])

        return {"answer": answer, "sources": []}

    except Exception as e:
        logger.error("Chat error: " + str(e))
        raise HTTPException(500, "Internal error: " + str(e))


# ── Local dev ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)