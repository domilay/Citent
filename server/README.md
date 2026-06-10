# JUROR Backend (`server/`)

Real FastAPI backend that runs the JUROR agent with **real components, not heuristics**:

| Stage | What's real |
|-------|-------------|
| ① Triage | **TF-IDF + LogisticRegression** trained on 271 940 examples (held-out acc 0.6932, macro-F1 0.5952). Pickles live in `models/`. |
| ② Investigator | Regex + keyword tools (no ML needed) |
| ③ Prosecution Panel | **Real LLM call** for each shortlisted intent (Anthropic / OpenAI / Google / DeepSeek) |
| ④ Cross-Examiner | Rule pre-flag → **real LLM critique** |
| ⑤ Fact-Checker | Exact match → BLEU → **real NLI** (`cross-encoder/nli-deberta-v3-small`, ~140 MB, downloaded lazily) |
| ⑥ Judge | Score + **real LLM that outputs JSON parsed into the verdict** |
| ⑦ Clerk | JSON packaging |

## Install

```bash
cd server/
pip install -r requirements.txt
```

PyTorch + transformers pull ~2 GB. The NLI model itself (~140 MB) downloads on first verification call.

## Configure provider API keys

Set at least one to enable Live mode. Unset providers gracefully error.

```bash
# pick one or more
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export DEEPSEEK_API_KEY="sk-..."

# optional model overrides
export ANTHROPIC_MODEL="claude-opus-4-6"
export OPENAI_MODEL="gpt-5.4"
export GOOGLE_MODEL="gemini-3.1-pro"
export DEEPSEEK_MODEL="deepseek-chat"

# optional NLI override
export JUROR_NLI_MODEL="cross-encoder/nli-deberta-v3-small"
```

## Run (dev)

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000> — the front-end auto-detects the backend and switches the **Live** badge to show "server".

## Run (prod)

```bash
gunicorn -k uvicorn.workers.UvicornWorker server:app \
         --bind 0.0.0.0:8000 --workers 2 --timeout 120
```

Or Docker:

```bash
docker build -t juror-backend .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY juror-backend
```

## Endpoints

| Method | Path | Returns |
|--------|------|---------|
| GET  | `/`               | Static front-end (`demo/index.html`) |
| GET  | `/api/status`     | What's wired (triage, NLI, providers) |
| GET  | `/api/cases`      | Built-in sample cases |
| POST | `/api/run-case`   | Run the agent on one case |
| GET  | `/docs`           | OpenAPI / Swagger UI (FastAPI's gift) |

### Example `/api/run-case` request

```json
POST /api/run-case
Content-Type: application/json

{
  "case": {
    "case_id": "demo",
    "paragraph": "Our proposed kernel estimator ... In contrast to their approach ...",
    "ref_clean_citation": "Yao et al. (2005)",
    "ground_truth": "Comparison"
  },
  "provider": "anthropic"
}
```

Returns the same trace JSON shape the front-end animation player consumes — so the result can be replayed in the courtroom UI as-is.

## Retrain the Triage model

If you want to retrain with different hyperparameters or on a different corpus:

```bash
python train_models.py \
    --citations  /path/to/Citations.csv \
    --labels     /path/to/labels.csv \
    --max-features 10000
```

Writes new pickles into `models/`. Restart the server to pick them up.

## Folder map

```
server/
├── server.py            FastAPI app + static mount
├── agent.py             7-stage orchestrator
├── tools.py             Triage, regex tools, NLI verifier
├── llm_clients.py       Anthropic / OpenAI / Google / DeepSeek
├── train_models.py      offline (re)training script
├── requirements.txt
├── Dockerfile
├── README.md            this file
└── models/              trained artefacts
    ├── tfidf_vectorizer.pkl   (≈35 MB)
    ├── lr_classifier.pkl      (≈300 KB)
    ├── label_encoder.pkl      (≈300 B)
    └── metrics.json           held-out report
```
