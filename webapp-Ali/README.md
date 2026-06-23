# MediaScope webapp

Factuality prediction for a news homepage screenshot. Serves the trained **MLP**
(`mediascope_artifacts/mlp.pt`) over the exact 1198-dim feature vector from the notebook:
`[DINOv2 768 | provenance 32 | LLM verdict one-hot 14 | MiniLM rationale 384]`.
Bias is not wired up — the UI shows it as a "coming soon" placeholder.

Each `/analyze` makes one internal OpenRouter call (`openai/gpt-5.5`, the engineered joint
prompt from `inference_joint.py`) to build verdict + rationale features. `/explain` makes a
second call afterwards for a plain-language paragraph.

## 1. Paste your OpenRouter key

```
cp backend/.env.example backend/.env
```

Open `backend/.env` and paste your key after the `=`:

```
OPENROUTER_API_KEY=sk-or-...
```

(Loaded in `backend/llm.py`, marked with a comment.)

## 2. Backend

```
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
playwright install chromium     # only if not already installed
uvicorn main:app --port 8000
```

On boot it validates the artifacts (PROV_DIM==32, mu/sd len 32, MLP first Linear
in_features==1198) and loads DINOv2 + MiniLM. First run downloads those weights.

## 3. Frontend

```
cd frontend
npm install
npm run dev
```

Open the printed URL (default http://localhost:5173). Vite proxies `/analyze` and
`/explain` to `http://localhost:8000`.

## Use

- Drag-and-drop or browse a homepage screenshot, then **Analyze**, **or**
- Paste a homepage URL and **Capture** — Playwright takes a full-page screenshot and
  feeds the same pipeline (URL mode also fetches live provenance; uploads use a zero
  provenance vector, matching the notebook's degraded rows).

Factuality renders first; the GPT-5.5 explanation streams into the card below.

## New dependencies vs the repo

Backend adds: `fastapi`, `uvicorn`, `timm`, `sentence-transformers`, `torch`,
`python-multipart`, `beautifulsoup4`, `tldextract`, `python-whois` (Playwright,
pandas, scikit-learn, pillow already in the project). Frontend: Vite + React + TS +
Tailwind.
