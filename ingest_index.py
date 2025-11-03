# ingest_index.py
import os
import re
import time
import hashlib
from typing import Iterable, List

import chromadb
from atlassian import Confluence
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from sentence_transformers import SentenceTransformer

# ========= Env / Config =========
CONF_URL    = os.getenv("CONF_URL")             # e.g. https://<site>.atlassian.net/wiki
CONF_USER   = os.getenv("CONF_USER")            # Atlassian email (Cloud)
CONF_TOKEN  = os.getenv("CONF_TOKEN")           # API token: id.atlassian.com
SPACE_KEY   = os.getenv("CONF_SPACE", "ENG")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")
COLLECTION  = f"confluence_{SPACE_KEY}"

# Chunking knobs
MAX_CHARS = 1800
OVERLAP   = 220
BATCH_EMB = 64
REBUILD   = os.getenv("REBUILD", "false").lower() in ("1", "true", "yes")

# Fail fast on missing vars
required = {"CONF_URL": CONF_URL, "CONF_USER": CONF_USER, "CONF_TOKEN": CONF_TOKEN, "CONF_SPACE": SPACE_KEY}
missing = [k for k, v in required.items() if not v]
if missing:
    raise SystemExit(f"[error] Missing env vars: {', '.join(missing)}. Ensure CONF_URL includes '/wiki'.")

if "/wiki" not in CONF_URL:
    raise SystemExit("[error] CONF_URL must include '/wiki', e.g. https://<site>.atlassian.net/wiki")

# ========= Clients =========
confluence = Confluence(url=CONF_URL, username=CONF_USER, password=CONF_TOKEN, cloud=True)
embedder   = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client     = chromadb.PersistentClient(path=CHROMA_PATH)

if REBUILD:
    try:
        client.delete_collection(COLLECTION)
        print(f"[info] Rebuilt collection: {COLLECTION}")
    except Exception:
        pass

collection = client.get_or_create_collection(COLLECTION)

# ========= Helpers =========
def fetch_pages(space_key: str, expand: str, page_limit: int = 500) -> List[dict]:
    """Paginate through all content in a space; filter pages locally (API may not support 'type=' kw)."""
    pages: List[dict] = []
    start = 0
    while True:
        batch = confluence.get_all_pages_from_space(
            space=space_key,
            start=start,
            limit=page_limit,
            status=None,
            expand=expand,
        ) or []
        # Some server/DC versions may return mixed content; keep only pages
        batch = [b for b in batch if (b.get("type") or "page") == "page"]
        pages.extend(batch)
        print(f"[fetch] got {len(batch)} (total {len(pages)}) start={start}")
        if len(batch) < page_limit:
            break
        start += page_limit
        time.sleep(0.15)
    return pages

def clean_confluence_html(html: str) -> str:
    """Remove common boilerplate/macros, convert to markdown, normalize whitespace."""
    soup = BeautifulSoup(html or "", "html.parser")

    # Remove obvious noise
    for tag in soup.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()

    for sel in ["div.expand-container", "div.comment", "div.ia-secondary-navigation", "div.ia-fixed-sidebar"]:
        for t in soup.select(sel):
            t.decompose()

    # Unwrap macro wrappers (keep inner text/code)
    for t in soup.find_all(["ac:structured-macro", "ac:parameter", "ac:layout", "ac:layout-section",
                            "ac:layout-cell", "ri:attachment", "ri:page", "ri:user"]):
        t.unwrap()

    markdown = md(str(soup), heading_style="ATX", strip=["a"])
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return markdown

def smart_chunks(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> Iterable[str]:
    """Paragraph/heading-aware chunking with overlap carry."""
    if not text:
        return
    parts = re.split(r"(^#{1,6} .*$)|(\n\s*\n)", text, flags=re.MULTILINE)
    paras = [p for p in parts if p and not p.isspace()]
    buf: List[str] = []
    size = 0
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if size + len(p) + 1 > max_chars and buf:
            chunk = "\n".join(buf).strip()
            yield chunk
            carry = chunk[-overlap:] if overlap > 0 else ""
            buf = [carry, p] if carry else [p]
            size = len(carry) + len(p)
        else:
            buf.append(p)
            size += len(p) + 1
    if buf:
        yield "\n".join(buf).strip()

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def page_url_from(p: dict) -> str:
    base = p.get("_links", {}).get("base")
    webui = p.get("_links", {}).get("webui")
    return base + webui if base and webui else f"{CONF_URL}/spaces/{SPACE_KEY}/pages/{p.get('id')}"

# ========= Ingest =========
def main():
    print(f"[info] Indexing space={SPACE_KEY} → collection={COLLECTION} at {CHROMA_PATH}")

    pages = fetch_pages(
        space_key=SPACE_KEY,
        expand="body.view,version,space,metadata",
        page_limit=500,
    )
    if not pages:
        raise SystemExit("[warn] No pages returned from Confluence.")

    ids:   List[str] = []
    docs:  List[str] = []
    metas: List[dict] = []
    seen  = set()  # global dedup across all chunks

    for idx, p in enumerate(pages, start=1):
        page_id = p.get("id")
        title   = p.get("title") or "Untitled"
        html    = p.get("body", {}).get("view", {}).get("value", "")
        updated = p.get("version", {}).get("when")
        url     = page_url_from(p)

        text = clean_confluence_html(html)
        if not text:
            continue

        prefix = f"# {title}\n"
        local_seen = set()

        for j, ck in enumerate(smart_chunks(text)):
            ck_full = (prefix + ck).strip()
            key = sha1(f"{page_id}:{ck_full}")
            if key in local_seen or key in seen:
                continue
            local_seen.add(key); seen.add(key)

            ids.append(f"{page_id}_{j}")
            docs.append(ck_full)
            metas.append({
                "space": SPACE_KEY,
                "page_id": page_id,
                "title": title,
                "url": url,
                "updated": updated,
                "visibility": "company",
            })

        if idx % 10 == 0:
            print(f"[build] processed {idx}/{len(pages)} pages")

    if not docs:
        raise SystemExit("[warn] No chunks built; aborting before embeddings.")

    # Embed in batches
    embeddings: List[List[float]] = []
    for i in range(0, len(docs), BATCH_EMB):
        batch = docs[i:i + BATCH_EMB]
        embs = embedder.encode(batch, normalize_embeddings=True)
        embeddings.extend(embs)
        if (i // BATCH_EMB) % 10 == 0:
            print(f"[embed] {i + len(batch)}/{len(docs)}")

    # Upsert to Chroma
    collection.upsert(documents=docs, ids=ids, metadatas=metas, embeddings=embeddings)
    print(f"[done] Indexed {len(docs)} chunks from {len(pages)} pages into '{COLLECTION}'.")

if __name__ == "__main__":
    main()
