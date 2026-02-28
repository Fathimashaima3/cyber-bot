# api/index.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

# ---------------- ENV ----------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-west1-gcp")  # change if needed
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "cyber-chatbot")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is missing!")

# ---------------- APP ----------------
app = FastAPI(title="Cyber Security RAG Chatbot API")

class ChatRequest(BaseModel):
    message: str

# ---------------- PINECONE ----------------
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX_NAME)

vectorstore = Pinecone(
    index=index,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    text_key="text"
)

# ---------------- QA CHAIN ----------------
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=False
)

# ---------------- ROUTES ----------------
@app.get("/")
def root():
    return {"status": "Cyber Security API is Running"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        result = qa_chain({"query": req.message})
        return {"answer": result["result"]}
    except Exception as e:
        return {"error": str(e)}