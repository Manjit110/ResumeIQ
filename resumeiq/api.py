# resumeiq/api.py
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from resumeiq.qa_chain import get_resume_bot

# ===== Local dev convenience (optional) =====
# Put OPENAI_API_KEY in a local .env (never commit it)
if os.getenv("ENV") != "prod":
    try:
        from dotenv import load_dotenv  # pip install python-dotenv (dev only)
        load_dotenv()  # loads .env if present
    except Exception:
        pass

PICKLE_PATH = "data/resume_vectorstore.pkl"

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure key exists before we start
    _ = _require_env("OPENAI_API_KEY")

    app.state.qa_bot = None
    try:
        app.state.qa_bot = get_resume_bot(PICKLE_PATH)
        logger.info("qa_bot initialized")
    except Exception as e:
        logger.exception("Startup: failed to init qa_bot: %r", e)
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
    return {"ok": True, "botReady": app.state.qa_bot is not None}


@app.post("/ask")
def ask_question(req: QuestionRequest):
    bot = app.state.qa_bot
    if bot is None:
        raise HTTPException(status_code=503, detail="Bot not ready")
    # get_resume_bot should internally use OPENAI_API_KEY from env
    response = bot.invoke({"query": req.question})
    return {"question": req.question, "answer": response}
