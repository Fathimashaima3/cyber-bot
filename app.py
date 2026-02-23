import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

# 1. LOAD DOTENV PROPERLY (Path-Safe)
# This finds the .env file in the same directory as this app.py
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# 2. CONFIGURATION & DEBUGGING
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "cyber-chatbot")

# --- STARTUP SECURITY CHECK ---
if not GROQ_API_KEY or "gsk_" not in GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY is missing or invalid in .env")
else:
    # Print the first and last few chars to verify no spaces
    print(f"✅ SUCCESS: Groq Key detected: {GROQ_API_KEY[:7]}...{GROQ_API_KEY[-4:]}")

# 3. INITIALIZE FASTAPI
app = FastAPI(title="Cyber Security RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# 4. INITIALIZE COMPONENTS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to Pinecone
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize Groq with explicit key
# Initialize the NEW supported LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,      # <--- Make sure there is a COMMA here
    model_name="llama-3.3-70b-versatile", 
    temperature=0              # <--- No comma needed on the last line
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# 5. ROUTES
@app.get("/")
def root():
    return {"status": "Cyber Security API is Running"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        # We use invoke for the new version of LangChain
        response = qa_chain.invoke({"query": req.message})
        return {"answer": response["result"]}
    except Exception as e:
        # Specifically catch the 401 to make it clear in the UI
        error_str = str(e)
        if "401" in error_str:
            return {"error": "Invalid API Key. Please check your .env file and restart the server."}
        return {"error": error_str}