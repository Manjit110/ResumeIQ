# resumeiq/api.py
import os
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from resumeiq.qa_chain import get_resume_bot

# ===== Local dev convenience (optional) =====
# Put OPENAI_API_KEY in a local .env (never commit it)
if os.getenv("ENV") != "prod":
    try:
        from dotenv import load_dotenv  # pip install python-dotenv (dev only)
        load_dotenv()  # loads .env if present
    except Exception:
        pass

# Resolve index dir robustly (absolute path, works regardless of cwd)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "index"))
INDEX_DIR = os.getenv("RESUME_INDEX_DIR", DEFAULT_INDEX_DIR)

logger = logging.getLogger("resumeiq")
logging.basicConfig(level=logging.INFO)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        # Crash early with a clear error so logs tell you exactly what's missing
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class QuestionRequest(BaseModel):
    question: str


def _preload_bot(app: FastAPI):
    """Load retriever/LLM in a background thread so startup doesn't block."""
    try:
        app.state.qa_bot = get_resume_bot(INDEX_DIR)
        logger.info("qa_bot initialized")
    except Exception as e:
        logger.exception("Startup: failed to init qa_bot: %r", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure key exists before we start
    _require_env("OPENAI_API_KEY")

    # Start with no bot; load it in the background to avoid blocking uvicorn bind
    app.state.qa_bot = None
    threading.Thread(target=_preload_bot, args=(app,), daemon=True).start()

    yield
    # nothing to clean up on shutdown (yet)


app = FastAPI(title="ResumeIQ API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "botReady": app.state.qa_bot is not None,
        "indexDir": INDEX_DIR,
    }


@app.post("/ask")
def ask_question(req: QuestionRequest):
    bot = app.state.qa_bot
    if bot is None:
        raise HTTPException(status_code=503, detail="Bot not ready")
    # get_resume_bot should internally use OPENAI_API_KEY from env and FAISS index on disk
    result = bot.invoke({"query": req.question})
    # LangChain chains sometimes return dicts; handle both cases
    if isinstance(result, dict):
        answer = result.get("result") or result.get("answer") or str(result)
        return {"question": req.question, "answer": answer}
    else:
        return {"question": req.question, "answer": str(result)}


# --- Azure App Service warmup & health-friendly routes ---
# Liveness: fast 200 at root (Azure sometimes probes '/')
@app.get("/", response_class=JSONResponse)
def root():
    return {"ok": True, "service": "ResumeIQ", "health": "/health"}

# Azure warmup probe often calls this exact path; return 200 instead of 404.
if os.getenv("WEBSITE_SITE_NAME"):
    @app.get("/robots933456.txt", response_class=PlainTextResponse)
    def _azure_warmup() -> str:
        # Valid, tiny robots.txt content
        return "User-agent: *\nDisallow:\n"

# Avoid noisy 404s on /favicon.ico during probes
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)
