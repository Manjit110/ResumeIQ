# resumeiq/qa_chain.py

import pickle
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def get_resume_bot(vectorstore_path):

    with open(vectorstore_path, "rb") as f:
        vectorstore = pickle.load(f)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain
