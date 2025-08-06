# resumeiq/vectorstore.py

import pickle

def load_vectorstore(path: str):
    with open(path, "rb") as f:
        vectorstore = pickle.load(f)
    return vectorstore
