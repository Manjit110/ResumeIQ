# resumeiq/api.py
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from resumeiq.qa_chain import get_resume_bot

os.environ["OPENAI_API_KEY"] = "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"

PICKLE_PATH = "data/resume_vectorstore.pkl"
logger = logging.getLogger("resumeiq")
logging.basicConfig(level=logging.INFO)

class QuestionRequest(BaseModel):
    question: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.qa_bot = None
    try:
        app.state.qa_bot = get_resume_bot(PICKLE_PATH)
        logger.info("qa_bot initialized")
    except Exception as e:
        # Keep process alive so Azure can hit /health and you can see logs
        logger.exception("Startup: failed to init qa_bot: %r", e)
    yield
    # Shutdown (nothing to clean yet)

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
    # report readiness (bot may still be None; that’s fine)
    return {"ok": True, "botReady": app.state.qa_bot is not None}

@app.post("/ask")
def ask_question(req: QuestionRequest):
    bot = app.state.qa_bot
    if bot is None:
        raise HTTPException(status_code=503, detail="Bot not ready")
    response = bot.invoke({"query": req.question})
    return {"question": req.question, "answer": response}
