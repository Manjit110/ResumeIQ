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
