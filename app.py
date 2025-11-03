# app.py
import os
import chromadb
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi.responses import HTMLResponse
from openai import OpenAI
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # fast & good
RERANK_TOP_K = 5
oai = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

client = chromadb.PersistentClient(path="./chroma")  # must match ingest path
SPACE_KEY = os.getenv("CONF_SPACE", "SD")
col = client.get_or_create_collection(f"confluence_{SPACE_KEY}")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="Confluence QA")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # dev only; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Query(BaseModel):
    question: str
    user_email: str = "someone@company.com"  # from SSO in real app
    k: int = 8

SYSTEM_PROMPT = (

"You are a helpful assistant that answers strictly using the provided Confluence"
   "If the answer is not in the context, say you don’t know. Keep answers under 8 sentences. "
    "Cite 2–5 sources by title as [Title](URL)."
)


def build_context_blocks(docs, metas, max_chars=5500):
    """Pack top chunks until we hit a safe context size."""
    blocks, cites = [], []
    used = 0
    for d, m in zip(docs, metas):
        snippet = d.strip()
        title = m.get("title", "Untitled")
        url = m.get("url")
        block = f"Title: {title}\nURL: {url}\nContent:\n{snippet}\n---\n"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        cites.append({"title": title, "url": url})
        used += len(block)
    return "\n".join(blocks), cites



def security_filter(metas, user_email):
    # TODO: implement real ACLs; here we accept all "company"
    return [m for m in metas]  # replace with real checks


with open("index.html", "r", encoding="utf-8") as f:
    INDEX_HTML = f.read()



@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse(INDEX_HTML)

@app.post("/ask")
def ask(q: Query):
    # 1) retrieve
    qvec = embedder.encode([q.question], normalize_embeddings=True)[0].tolist()
    res = col.query(query_embeddings=[qvec], n_results=q.k,
                    include=["metadatas", "documents", "distances"])

    metas = [m for sub in res["metadatas"] for m in sub]
    docs  = [d for sub in res["documents"] for d in sub]

    if not docs:
        raise HTTPException(status_code=404, detail="No relevant content found.")

    context_text, cites = build_context_blocks(docs, metas)

    # 2) craft prompt
    user_prompt = (
        f"QUESTION:\n{q.question}\n\n"
        f"CONTEXT (Confluence excerpts, newest to oldest may vary):\n{context_text}\n"
        "Return a concise answer followed by a bullet list of citations."
    )

    # 3) call OpenAI (Responses API)
    try:
        resp = oai.responses.create(
            model=os.getenv("OAI_MODEL", "gpt-4o-mini"),
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": SYSTEM_PROMPT}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt}
                    ],
                },
            ],
            max_output_tokens=600,
        )
        answer_text = resp.output_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")
    

    return {"answer": answer_text, "sources": cites[:5]}
# new
