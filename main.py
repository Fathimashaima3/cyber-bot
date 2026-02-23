import os
import json
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API KEY
raw_key = os.getenv("GROQ_API_KEY")
if raw_key:
    os.environ["GROQ_API_KEY"] = raw_key
else:
    print("WARNING: GROQ_API_KEY not found")

# Load Data
with open("cyber_security.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [Document(page_content=entry["text"]) for entry in data if "text" in entry]

# Smaller embedding model (Render safe)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

system_prompt = (
    "You are an expert Cyber Security Educator.\n"
    "Use context first, otherwise answer normally.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat(input: ChatInput):
    response = rag_chain.invoke({"input": input.message})
    return {"reply": response["answer"]}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
