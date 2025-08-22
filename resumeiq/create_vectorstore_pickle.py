import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# explicitly load from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "", ".env"))


# ======= CONFIG =======
PDF_PATH = r"C:\Users\Manjit Singh\PycharmProjects\ResumeIQ\data\Manjit_Singh.pdf"
INDEX_DIR = r"C:\Users\Manjit Singh\PycharmProjects\ResumeIQ\data\resume_faiss"  # creates index.faiss + index.pkl here
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

def load_and_split_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

def build_and_save_faiss(pdf_path: str, index_dir: str):
    # 1) Load + split
    chunks = load_and_split_pdf(pdf_path)

    # 2) Cost-efficient embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # ✅ cheaper

    # 3) Build FAISS
    vs = FAISS.from_documents(chunks, embedding=embeddings)

    # 4) Save as index.faiss + index.pkl
    os.makedirs(index_dir, exist_ok=True)
    vs.save_local(index_dir)
    print(f"✅ Saved:\n  {os.path.join(index_dir, 'index.faiss')}\n  {os.path.join(index_dir, 'index.pkl')}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    build_and_save_faiss(PDF_PATH, INDEX_DIR)
