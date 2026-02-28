import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

# ---------------- ENV ----------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "cyber-chatbot")

# ---------------- APP ----------------
app = FastAPI(title="Cyber Security RAG Chatbot API")

class Query(BaseModel):
    question: str

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # lightweight
)

# ---------------- VECTORSTORE ----------------
vectorstore = Pinecone(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------- QA CHAIN ----------------
qa_chain = RetrievalQA.from_chain_type(
    llm=None,  # No LLM here to avoid huge dependencies
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False
)

# ---------------- ROUTES ----------------
@app.get("/")
def root():
    return {"status": "Cyber Security API is Running"}

@app.post("/chat")
def chat(query: Query):
    # For lightweight, we only retrieve relevant docs from Pinecone
    docs = retriever.get_relevant_documents(query.question)
    return {"results": [doc.page_content for doc in docs]}