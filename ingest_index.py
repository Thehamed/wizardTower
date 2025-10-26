# ingest_index.py
import os, time, re
from atlassian import Confluence
from markdownify import markdownify as md
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb

CONF_URL = os.getenv("CONF_URL")             # e.g., https://your-domain.atlassian.net/wiki
CONF_USER = os.getenv("CONF_USER")           # email for Cloud; or username for Server/DC
CONF_TOKEN = os.getenv("CONF_TOKEN")         # API token or PAT
SPACE_KEY = os.getenv("CONF_SPACE", "SD")   # which space to index

# 1) Confluence client
confluence = Confluence(url=CONF_URL, username=CONF_USER, password=CONF_TOKEN, cloud=True)

# 2) Get pages in space
pages = confluence.get_all_pages_from_space(space=SPACE_KEY, start=0, limit=500, status=None, expand="body.view,version,space,metadata")

def html_to_markdown(html):
    # strip macros/noise if needed
    soup = BeautifulSoup(html, "html.parser")
    # Example: remove code macro headers/expanders if noisy
    for tag in soup.find_all("ac:structured-macro"):
        tag.unwrap()
    return md(str(soup))

def chunk(text, max_tokens=900, overlap=150):
    # rough tokenization by words (ok for MVP)
    words = text.split()
    step = max_tokens - overlap
    for i in range(0, max(1, len(words) - overlap), step):
        yield " ".join(words[i:i+max_tokens])

# 3) Vector DB (Chroma)
client = chromadb.PersistentClient(path="./chroma")  # any folder you like
collection = client.get_or_create_collection(name=f"confluence_{SPACE_KEY}")
# 4) Embeddings
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

ids = []
docs = []
metas = []
embs = []

for p in pages:
    page_id = p["id"]
    title = p["title"]
    html = p["body"]["view"]["value"]
    url = f"{CONF_URL}/spaces/{SPACE_KEY}/pages/{page_id}"
    updated = p.get("version", {}).get("when")

    text = html_to_markdown(html)
    for j, ck in enumerate(chunk(text)):
        meta = {
            "space": SPACE_KEY,
            "page_id": page_id,
            "title": title,
            "url": url,
            "updated": updated,
            # Minimal ACL placeholder: widen later with real viewer groups/emails
            "visibility": "company"  # or "public", "restricted"
        }
        ids.append(f"{page_id}_{j}")
        docs.append(ck)
        metas.append(meta)

# do embeddings in batches
B = 64
for i in range(0, len(docs), B):
    embs.extend(embedder.encode(docs[i:i+B], normalize_embeddings=True))

collection.upsert(documents=docs, ids=ids, metadatas=metas, embeddings=embs)
print(f"Indexed {len(docs)} chunks from space {SPACE_KEY}.")
