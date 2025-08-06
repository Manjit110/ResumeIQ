# resumeiq/vectorstore.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from resumeiq.loader import load_and_split_resume

# Wrapper to initialize vector store
def create_vectorstore(pdf_path: str) -> DocArrayInMemorySearch:
    """
    Loads, splits, and indexes resume into an in-memory vector store.

    Args:
        pdf_path (str): Path to the resume PDF file.

    Returns:
        DocArrayInMemorySearch: In-memory vector store for similarity search
    """
    chunks = load_and_split_resume(pdf_path)
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = DocArrayInMemorySearch.from_documents(chunks, embedding=embeddings)
    return vectorstore
