from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from resumeiq.qa_chain import get_resume_bot
import os

# Set your OpenAI key here or use env variables
os.environ["OPENAI_API_KEY"] = "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"  # <--- use your real key or ENV

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
qa_bot = get_resume_bot(PICKLE_PATH)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QuestionRequest):
    response = qa_bot.invoke({"query": req.question})
    return {"question": req.question, "answer": response}
