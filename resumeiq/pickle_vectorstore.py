# pickle_vectorstore.py

from resumeiq.vectorstore import create_vectorstore
import pickle

PDF_PATH = "data/Manjit_Singh.pdf"
VECTORSTORE_PATH = "data/resume_vectorstore.pkl"

vectorstore = create_vectorstore(PDF_PATH)
with open(VECTORSTORE_PATH, "wb") as f:
    pickle.dump(vectorstore, f)

print("✅ Vectorstore pickled and saved!")
