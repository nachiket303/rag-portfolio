# Nachiket Portfolio RAG Chatbot

A local **Retrieval-Augmented Generation (RAG)** system that lets recruiters
chat with an AI that answers questions about Nachiket Shejwal powered by
**Groq**  and your personal PDF documents.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq API — `llama3-8b-8192` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2|
| Vector Store | ChromaDB (persisted locally at `./chroma_db`) |
| PDF Parsing | PyMuPDF (fitz) |
| RAG Pipeline | LangChain LCEL |
| API Server | FastAPI + Uvicorn |
| Deployment | Render (free tier) |

---

## Local Development

### Step 1 — Get your free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with email only — **no credit card needed**
3. Go to **API Keys** → **Create New Key**
4. Copy the key and paste it into `.env`:

```
GROQ_API_KEY=your_key_here
```

---

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` model (~90 MB) — one time only.

---

### Step 3 — Add your PDF documents

Place all your PDFs into the `./docs/` folder:

```
docs/
├── resume.pdf
├── linkedin.pdf
├── project_report.pdf
├── dissertation.pdf
└── manulife_work.pdf
```

Any PDF with selectable text works. Scanned image-only PDFs will be skipped
with a warning (use OCR first if needed).

---

### Step 4 — Build the knowledge base

```bash
python src/ingest.py
```

This will:
- Extract text from every PDF in `./docs/`
- Split text into overlapping 600-token chunks
- Embed each chunk using the local HuggingFace model
- Store everything in ChromaDB at `./chroma_db/`

**Run this every time you add new PDF files.**

---

### Step 5 — Start the server

```bash
uvicorn src.app:app --reload --port 8000
```

---

### Step 6 — Test the health check

Open your browser: [http://localhost:8000/health](http://localhost:8000/health)

Expected response:
```json
{
  "status": "ok",
  "groq": "connected",
  "vectorstore": "ready"
}
```

---

### Step 7 — Ask a question

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "What technologies have you worked with?"}'
```

Or open the FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Adding More Documents Later

```bash
# Drop new PDFs into ./docs/
# Then run:
python src/ingest.py
# Choose:
#   1 → Add new documents to existing store
#   2 → Wipe and rebuild from scratch
```

---

## Testing Retrieval (Debugging)

```bash
python -c "from src.retriever import test_retrieval; test_retrieval('did you work with Databricks')"
```

Sample output:
```
Query: 'did you work with Databricks'
────────────────────────────────────────────────────────────
Result 1 (score: 0.87) from: cloudaiapex_report.pdf
...text snippet showing the relevant passage...
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Ask a question, get an answer + sources |
| `GET` | `/health` | Check Groq key + vector store status |
| `GET` | `/docs-loaded` | List all loaded documents + chunk count |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Deployment to Render (Free)

### Step 1 — Commit `chroma_db/` to GitHub

```bash
# Temporarily remove chroma_db from .gitignore
# Edit .gitignore and comment out or remove: chroma_db/

git add chroma_db/
git commit -m "add vector store"
git push

# Then restore chroma_db/ to .gitignore
```

> **Why?** Render's free tier has no persistent disk. Committing the
> pre-built vector store means the app works immediately without needing
> to re-run ingest on the server.

---

### Step 2 — Push code to GitHub

```bash
# Create a new GitHub repository, then:
git init
git remote add origin https://github.com/YOUR_USERNAME/portfolio-rag.git
git add .
git commit -m "initial commit"
git push -u origin main
```

Note: `docs/` and `.env` are gitignored — your PDFs and API key stay private.

---

### Step 3 — Deploy on Render

1. Go to [render.com](https://render.com) and sign up free
2. Click **New → Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.app:app --host 0.0.0.0 --port $PORT`
5. Add **Environment Variable:**
   - Key: `GROQ_API_KEY`
   - Value: your Groq API key
6. Click **Deploy**

Your API will be live at: `https://your-app-name.onrender.com`

---

### Step 4 — Connect your Portfolio Frontend

In your `Nachiket_Portfolio.html`, find:

```javascript
const API_URL = 'http://localhost:8000';
```

Change it to:

```javascript
const API_URL = 'https://your-app-name.onrender.com';
```

Upload the updated HTML to GitHub Pages — done! ✅

---

## Project Structure

```
rag-portfolio/
├── docs/                  ← Drop all PDF files here (gitignored)
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── ingest.py          ← Build the knowledge base
│   ├── retriever.py       ← ChromaDB search logic
│   ├── chain.py           ← Groq LLM RAG pipeline
│   └── app.py             ← FastAPI server
├── chroma_db/             ← Auto-created vector store (gitignored by default)
├── .env                   ← Your Groq API key (gitignored)
├── .gitignore
├── requirements.txt
├── Procfile               ← Render deployment config
└── README.md
```

---

## Troubleshooting

**`GROQ_API_KEY not found`**
→ Make sure `.env` exists and contains `GROQ_API_KEY=your_key_here`

**`Vector store not found. Please run python src/ingest.py first.`**
→ You haven't ingested any PDFs yet. Add PDFs to `./docs/` and run `python src/ingest.py`

**`No PDF files found in ./docs/ folder.`**
→ Make sure your files have the `.pdf` extension and are in the `docs/` directory

**`page X appears to be scanned, skipping`**
→ That page is an image — use an OCR tool to convert it to selectable text first

**Answers are vague or off-topic**
→ Run the retrieval debug command to see what chunks are being fetched:
```bash
python -c "from src.retriever import test_retrieval; test_retrieval('your question here')"
```
If results look wrong, try option 2 (rebuild from scratch) when running `ingest.py`

---

## License

MIT — feel free to adapt this for your own portfolio.
