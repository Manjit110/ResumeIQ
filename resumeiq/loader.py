# resumeiq/loader.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


def load_and_split_resume(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Loads and splits the resume PDF into text chunks.

    Args:
        pdf_path (str): Path to the resume PDF file.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[Document]: List of split documents.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(pages)

    return chunks
