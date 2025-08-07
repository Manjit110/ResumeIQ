from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
try:
    from resumeiq.qa_chain import get_resume_bot
except Exception as e:
    print("Import error in api.py:", e)
    raise

import os

# Set your OpenAI key here or use env variables
os.environ["OPENAI_API_KEY"] = "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"
app = FastAPI(title="ResumeIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # For dev; lock down in prod!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the bot once on startup (NOW loads the pickled vectorstore)
PICKLE_PATH = "data/resume_vectorstore.pkl"
if not os.path.isfile(PICKLE_PATH):
    print("==> MISSING PICKLE FILE at", PICKLE_PATH, flush=True)
else:
    print("==> FOUND PICKLE FILE at", PICKLE_PATH, flush=True)

try:
    qa_bot = get_resume_bot(PICKLE_PATH)
except Exception as e:
    print("Error loading vectorstore or initializing qa_bot:", e)
    raise


class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QuestionRequest):
    response = qa_bot.invoke({"query": req.question})
    return {"question": req.question, "answer": response}
