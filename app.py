import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# List of pages
pages = [
    "https://savesomeonecharity.org",
    "https://savesomeonecharity.org/about",
    "https://savesomeonecharity.org/causes",
    "https://savesomeonecharity.org/event",
    "https://savesomeonecharity.org/contact",
    "https://savesomeonecharity.org/donate",
    "https://savesomeonecharity.org/single",
]


def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# Scrape and build FAISS only once
all_chunks, metadata = [], []

for url in pages:
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    title = str(soup.title.string) if soup.title else ""
    desc_tag = soup.find("meta", attrs={"name": "description"})
    description = str(desc_tag["content"]) if desc_tag else ""  # type: ignore
    body_text = soup.get_text(separator="\n")
    chunks = chunk_text(body_text)

    for chunk in chunks:
        all_chunks.append(chunk)
        metadata.append(
            {
                "title": title,
                "description": description,
                "chunk": chunk,
                "url": url,
            }
        )

print(f"Total chunks: {len(all_chunks)}")

# Embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(
    all_chunks, show_progress_bar=True, convert_to_numpy=True
)

# FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)  # type: ignore
faiss.write_index(index, "embeddings.faiss")
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f, protocol=4)

print("FAISS index and metadata saved successfully!")

# Load precomputed FAISS + metadata
index = faiss.read_index("embeddings.faiss")
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# FastAPI app
app = FastAPI()


class Query(BaseModel):
    question: str
    top_k: int = 3


@app.get("/")
def root():
    return {"message": "Hello, this is my FastAPI app running on Hugging Face Spaces!"}

@app.post("/retrieve")
def retrieve(query: Query):
    q_emb = embed_model.encode([query.question], convert_to_numpy=True)
    D, I = index.search(q_emb, query.top_k)  # type: ignore
    results = [metadata[i] for i in I[0]]
    return {"results": results}
