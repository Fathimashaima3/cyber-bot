import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone

# ---------------- ENV ----------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "cyber-chatbot"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------- APP ----------------

app = FastAPI()

# ---------------- LLM ----------------

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# ---------------- EMBEDDINGS ----------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- PINECONE ----------------

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# ---------------- QA CHAIN ----------------

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=False
)

# ---------------- API ----------------

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    result = qa_chain({"query": query.question})
    return {"answer": result["result"]}