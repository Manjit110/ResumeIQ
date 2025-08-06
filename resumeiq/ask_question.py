# examples/ask_question.py

from resumeiq.qa_chain import get_resume_bot
import os

# STEP 1: Set your OpenAI API key (recommend using environment variable)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

# STEP 2: Load the bot with your resume PDF
pdf_path = "data/Manjit_Singh.pdf"  # Update if path differs
bot = get_resume_bot(pdf_path)

# STEP 3: Ask a question
question = "What is Manjit's experience with Solace and FIX protocol?"
response = bot.run(question)

print("Q:", question)
print("A:", response)
