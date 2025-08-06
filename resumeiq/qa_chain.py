# resumeiq/qa_chain.py

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from resumeiq.vectorstore import create_vectorstore

# Wrapper function to build the QA bot
def get_resume_bot(pdf_path: str) -> RetrievalQA:
    """
    Builds a QA chain that can answer questions from the resume.

    Args:
        pdf_path (str): Path to the resume PDF file.

    Returns:
        RetrievalQA: A LangChain QA chain object
    """
    vectorstore = create_vectorstore(pdf_path)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    return qa_chain
