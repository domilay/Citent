# Deployment Guide — for the engineer

This guide is for the engineer deploying JUROR. Read this **first**.

---

## Two modes of deployment

| Mode | What runs | When to use |
|------|-----------|-------------|
| **Server-real (recommended)** | FastAPI backend + frontend on the same port. Real TF-IDF + LR triage. Real DeBERTa NLI. Real LLM calls (server holds keys). | The default. Whenever you can run a Python process. |
| **Static-only (fallback)** | Just the `juror_demo_standalone.html`. Demo plays scripted traces. Live mode (if user pastes a key) calls LLMs directly from the browser. | Static-hosting only environments (GitHub Pages, S3). Demo / pitch use. |

The frontend **auto-detects** which mode is available. If you deploy the server, the UI shows a "server" badge next to Live. If you only ship the HTML, it falls back to the in-browser path.

---

## A. Server-real deployment (the real product)

### 1. Install

```bash
cd server/
pip install -r requirements.txt           # ~2 GB (PyTorch + transformers)
```

Tested on Python 3.10+. CPU is fine — NLI inference is small.

### 2. Configure provider API keys

Set at least one provider key. Multiple providers can coexist; the frontend's Settings drawer picks which one to call.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."         # required for Anthropic provider
export OPENAI_API_KEY="sk-..."                # required for OpenAI
export GOOGLE_API_KEY="..."                   # required for Google
export DEEPSEEK_API_KEY="sk-..."              # required for DeepSeek

# optional model overrides:
export ANTHROPIC_MODEL="claude-opus-4-6"
export OPENAI_MODEL="gpt-5.4"
export GOOGLE_MODEL="gemini-3.1-pro"
export DEEPSEEK_MODEL="deepseek-chat"
```

If you want to restrict CORS:

```bash
export CORS_ORIGINS="https://your-domain.example,https://another.example"
```

### 3. Run

```bash
# dev
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# prod
gunicorn -k uvicorn.workers.UvicornWorker server:app \
         --bind 0.0.0.0:8000 --workers 2 --timeout 120
```

Open <http://localhost:8000>. Click **Run** on case 03 → real LLM call lights up the LLM-calls counter on the meter bar.

### 4. Docker

```bash
docker build -t juror-backend .
docker run -p 8000:8000 \
           -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
           juror-backend
```

The Dockerfile pre-downloads the NLI weights at build time so first-request latency is sub-second.

### 5. nginx reverse-proxy (typical prod layout)

```nginx
upstream juror_backend { server 127.0.0.1:8000; }

server {
    server_name juror.example.com;
    location / {
        proxy_pass http://juror_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # LLM calls can take 5–20 s; raise the read timeout
        proxy_read_timeout 120s;
    }
}
```

### What the server does at boot

- Loads `server/models/tfidf_vectorizer.pkl` + `lr_classifier.pkl` (~35 MB total)
- **Lazy** loads `cross-encoder/nli-deberta-v3-small` on first `/api/run-case` call (~140 MB download, then cached)
- Validates env vars for each provider; surfaces "no key" cleanly via 400

### Smoke test

```bash
curl -s http://localhost:8000/api/status | jq
# → ok: true, triage.ready: true, fact_checker.nli_available: true,
#   llm_providers: {anthropic: true|false, ...}
```

---

## B. Static-only deployment (fallback / demo only)

If you really can't run Python:

```bash
# Just ship one file:
cp juror_demo_standalone.html /var/www/html/index.html
```

The page works offline. Live mode (Settings → paste key → Activate) calls LLM providers directly from the browser. This is **not** running the trained ML; in this mode Triage uses regex-based marker scoring.

If you go this route, add the security disclaimer below to your landing page.

---

## ⚠️ Security disclosure

* **Server-real mode**: provider API keys live in server environment variables. They never reach the browser. Safe for public deployment **if the server itself is behind your usual auth**.

* **Static-only mode**: user-supplied keys are sent directly from their browser to the provider. Acceptable for self-service demos where the user knowingly enters their own key, **not appropriate as a public service**. Anthropic flags direct-from-browser as "dangerous" in their official docs.

* **In all modes**: paragraphs sent to providers may contain copyrighted scholarly text; check that you have rights to share this content with the chosen provider.

---

## Required Content Security Policy (if you set CSP headers)

For server-real mode:

```
default-src 'self';
script-src  'self' 'unsafe-inline';
style-src   'self' 'unsafe-inline' https://fonts.googleapis.com;
font-src    'self' https://fonts.gstatic.com;
img-src     'self' data:;
connect-src 'self';
```

For static-only mode (browser calls LLMs directly), add to `connect-src`:

```
connect-src 'self'
            https://api.anthropic.com
            https://api.openai.com
            https://generativelanguage.googleapis.com
            https://api.deepseek.com;
```

---

## File inventory

```
agent_design_v2/
├── juror_demo_standalone.html    Single-file demo (static-only mode)
├── DEPLOYMENT.md                  this file
├── README.md                      project overview
├── AWARD_C_SUBMISSION.md          submission narrative (for contest organisers)
├── architecture.svg               system architecture diagram
├── bundle.py                      rebuilds standalone HTML from demo/
│
├── server/                        ★ FastAPI backend (real product)
│   ├── server.py                  app + routes
│   ├── agent.py                   7-stage orchestrator
│   ├── tools.py                   Triage, NLI, regex tools
│   ├── llm_clients.py             provider HTTP clients
│   ├── train_models.py            re-train Triage from raw CSVs
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── README.md
│   └── models/                    trained pickles (ships pre-built)
│       ├── tfidf_vectorizer.pkl
│       ├── lr_classifier.pkl
│       ├── label_encoder.pkl
│       └── metrics.json           held-out evaluation
│
├── demo/                          frontend source
│   ├── index.html
│   ├── style.css
│   ├── animation.js               trace player + UI glue
│   ├── live-agent.js              browser-direct LLM client (fallback)
│   ├── traces/case_*.json         pre-recorded demo runs
│   └── README.md
│
├── prototype/juror.py             original Python prototype
└── video/script.md                AI-video shooting script
```

**Default ship**: the whole `server/` folder (the real product).
**Backup ship**: just `juror_demo_standalone.html` (static-only).

---

## Smoke-test checklist (before going live)

Server-real:

- [ ] `curl http://localhost:8000/api/status` → 200 with `triage.ready: true`
- [ ] Open `/` in browser, click case 03, click Run → animation plays
- [ ] Verdict in bottom-right shows "Comparison" (ground truth for case 03)
- [ ] LLM-calls counter shows 4–6 calls when the run finishes
- [ ] Stage 1 transcript says `tfidf_lr_classifier → max_prob = … · real LR (n=340K)`
- [ ] Stage 5 transcript shows `cross-encoder/nli-deberta-v3-small` (NLI loaded)

Static-only:

- [ ] Double-click `juror_demo_standalone.html` → page opens
- [ ] Default Demo mode plays all 3 cases via Run
- [ ] Settings → paste a real Anthropic key → Activate Live → Run → real LLM call fires

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `tfidf_vectorizer.pkl` not found | Run `python train_models.py --citations … --labels …` to rebuild. |
| `transformers` import fails | `pip install transformers torch sentencepiece` |
| NLI takes 30 s on first call | Normal — model downloads ~140 MB once, then cached under `$HF_HOME` |
| `server has no API key for 'X'` | Set the corresponding env var: `ANTHROPIC_API_KEY`, etc. |
| Browser shows "No backend detected" badge | The frontend can't reach `/api/status`. Either reverse-proxy correctly, or use static-only mode. |
| 502 / 504 from nginx during a run | Increase `proxy_read_timeout` to 120s. LLM calls are slow. |
