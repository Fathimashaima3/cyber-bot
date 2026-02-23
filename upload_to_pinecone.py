from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import os

# Load Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("cyber-chatbot")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load PDF
pdf_path = "data/book.pdf"
reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text()

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(text)

# Embed & upload
vectors = []
for i, chunk in enumerate(chunks):
    emb = model.encode(chunk).tolist()
    vectors.append((f"id-{i}", emb, {"text": chunk}))

index.upsert(vectors=vectors)

print(f"Upload completed: {len(vectors)} chunks")