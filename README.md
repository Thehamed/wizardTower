# wizardTower
this is where wise owl begins
# WizardTower – Confluence Chatbot 🧙‍♂️

A Gen AI RAG chatbot that indexes Confluence pages and answers questions grounded in your documentation.

### 🧩 Tech Stack
- FastAPI backend
- OpenAI `gpt-4o-mini` for synthesis
- Chroma / FAISS for vector storage
- SentenceTransformers or OpenAI embeddings
- Confluence REST API for ingestion
- Tailwind HTML UI

### 🚀 Run locally
```bash
# 1. Set env vars, these I have
export CONF_URL="https://your-site.atlassian.net/wiki"
export CONF_USER="you@example.com"
export CONF_TOKEN="your_confluence_api_token"
export CONF_SPACE="ENG"
export OPENAI_API_KEY="sk-..."

# 2. Index Confluence data
python ingest_index.py

# 3. Start API
uvicorn app:app --reload --port 8000

# 4. Open UI
open index.html
