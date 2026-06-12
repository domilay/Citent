# Citent - Citation Intent Labelers

> **From a paragraph and a cited reference — to a calibrated citation-intent label, with every quoted claim verified against the source and every rejected alternative explained.**

[🌐 Playground](http://8.141.116.119:7868) · [📄 Architecture diagram](./architecture.svg) · [▶ Watch on YouTube](https://youtu.be/EBAGmQXdnfo)

<p align="center">
  <video src="./media/demo.mp4"
         controls
         muted
         preload="metadata"
         width="760"
         poster="https://img.youtube.com/vi/EBAGmQXdnfo/maxresdefault.jpg">
    Your browser does not support inline video — <a href="https://youtu.be/EBAGmQXdnfo">watch on YouTube</a>.
  </video>
</p>

---

## What it does

Citent is an LLM-powered annotation agent for **citation-intent classification** — the four-way decision of whether a citation is Background, Technical basis, Comparison, or Fundamental idea. Drop in a paragraph and the citation you want labelled; Citent runs a seven-stage workflow (code-named **JUROR**) that triages on a real ML classifier, gathers structured evidence, lets one advocate argue per intent, cross-examines weak claims, verifies every quoted span against the source, and seals the verdict into an auditable annotation artifact.

The headline output is not a label — it's an **artifact**: each entry shows the predicted intent, the calibrated confidence, every evidence span that survived verification, and every rejected alternative with the reason it was thrown out.

Here is a real run from one of the bundled hard cases — the paragraph cites *Yao et al. (2005)* in a sentence that contrasts convergence rates:

```
Verdict: Comparison  ·  confidence 0.70  ·  margin 0.50

Verified evidence:
  ✓  "faster than the non-parametric rate reported by"   [exact match]
  ✓  "in contrast to their approach"                     [exact match]

Rejected alternatives:
  ✗  Technical basis   no adoption verb in paragraph
  ✗  Background        single weak marker; insufficient signal

Trace: 5 LLM calls, 600 tokens, 11 tool invocations, 1 evidence claim dropped as fabricated.
```

---

## Three input modes

```bash
# 1. Single case (one paragraph + one ref) — call the running server
curl -X POST http://localhost:8000/api/run-case \
     -H "content-type: application/json" \
     -d '{"case":{"case_id":"demo",
                  "paragraph":"... In contrast to their approach ...",
                  "ref_clean_citation":"Yao et al. (2005)"},
          "provider":"anthropic"}'

# 2. Built-in sample cases — open the playground UI, pick 01 / 02 / 03, click Run
python -m uvicorn server:app --host 0.0.0.0 --port 8000
# then open http://localhost:8000

# 3. Offline batch — Python prototype over a CSV of (ID, paragraph, ref)
python prototype/juror.py --case 1
python prototype/juror.py --case 2
python prototype/juror.py --case 3
```

`provider` accepts `anthropic`, `openai`, `google`, or `deepseek`. The server picks the matching key from environment variables; absent providers return a clean 400.

---

## Why this is an agent, not a wrapper

Calling an LLM with "what is the intent of this citation?" sounds like the answer. It isn't. The same citation can be labelled four different ways depending on which sentence-level signal the model anchors on; blind chain-of-thought routinely flips the answer between turns. Worse, advocates invent quotes that don't exist in the source. Count those hallucinations as evidence and you ship a citation database that's confidently wrong.

These are judgment problems, not prompting problems. The architecture is deliberately frugal about where it spends LLM calls:

| Layer | What does the work |
|---|---|
| Difficulty + label shortlist | Real TF-IDF + LogisticRegression (trained on 340 K) |
| Rhetorical markers, section detection, co-citation graph | Deterministic regex / keyword rules |
| Argumentation per candidate intent | LLM advocate (one call per shortlisted label) |
| Adversarial critique | Rule pre-flag → LLM critic (only when needed) |
| Evidence verification (no LLM allowed) | Exact match → BLEU → DeBERTa-NLI entailment |
| Final verdict + calibrated confidence | LLM judge → JSON parsed → tempered by rule margin |
| Annotation artifact packaging | Pure code |

Four of seven stages call no LLM. The LLM is the scalpel, not the hammer.

---

## Pipeline

Seven stages turn a (paragraph, reference) tuple into a verified annotation artifact.

```
Input (paragraph + cited reference)
  └─▶ ① Triage         real TF-IDF + LR (held-out acc 0.69) — difficulty + shortlist
       └─▶ ② Investigator   regex markers, keyword section classifier, co-citation lookup
            └─▶ ③ Prosecution Panel    one LLM advocate per shortlisted intent
                 └─▶ ④ Cross-Examiner   rule pre-flag → LLM rebuttal on weak claims
                      └─▶ ⑤ Fact-Checker  exact → BLEU → DeBERTa-NLI entailment
                           └─▶ ⑥ Judge   LLM JSON verdict, calibrated and margin-checked
                                └─▶ ⑦ Clerk     seals { label, confidence, evidence,
                                                       rejected_alternatives, trace }
```

A typical hard case spends ~5 LLM calls (one per advocate + one critic + one judge) and ~600 tokens. The first four stages are free — the LLM counter on the playground stays at zero until the Prosecution Panel speaks.

---

## Installation

```bash
git clone https://github.com/domilay/Citent.git
cd Citent/server
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, plus the packages below.

```
fastapi>=0.110     scikit-learn>=1.3     transformers>=4.40
uvicorn>=0.27      scipy>=1.10           torch>=2.1
pydantic>=2.5      numpy>=1.24           sentencepiece>=0.2
httpx>=0.26        pandas>=2.0
```

LLM calls go through the providers' **official APIs**. Set whichever keys you intend to use:

```bash
ANTHROPIC_API_KEY=sk-ant-...       # required for Anthropic provider
OPENAI_API_KEY=sk-...              # required for OpenAI
GOOGLE_API_KEY=...                 # required for Google
DEEPSEEK_API_KEY=sk-...            # required for DeepSeek
```

Check the wiring before a real run:

```bash
curl http://localhost:8000/api/status | python -m json.tool
```

---

## Quick start

```bash
# Dev (auto-reload, single worker)
cd server/
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Prod
gunicorn -k uvicorn.workers.UvicornWorker server:app \
         --bind 0.0.0.0:8000 --workers 2 --timeout 120

# Docker
docker build -t citent .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY citent
```

**Playground UI** — open `http://localhost:8000` for a local deploy, or use the live deployment at <http://8.141.116.119:7868>. Three sample cases ship built-in; pick one, optionally switch provider in Settings, click Run. The animation walks through every stage with a live LLM-calls / token counter.

**Offline fallback** — if you cannot run Python, `juror_demo_standalone.html` is a single self-contained file (~130 KB) that runs the demo trace in any browser. Live mode in this fallback calls LLM providers directly from the browser using a user-pasted key.

---

## Output

Each run returns a JSON trace that includes the full agent execution plus the final artifact:

| Field | Contents |
|---|---|
| `final_artifact.label` | The predicted intent (one of four) |
| `final_artifact.confidence` | Calibrated probability (Platt-style softmax over rule + LLM scores) |
| `final_artifact.margin` | Gap between top label and runner-up |
| `final_artifact.evidence` | Quoted spans that the Fact-Checker verified |
| `final_artifact.rejected_alternatives` | Other labels considered, with the reason each was dropped |
| `final_artifact.abstain` | `true` when confidence drops below 0.55 — send to human |
| `stages[*]` | Per-stage tools called, tokens spent, LLM calls, actions emitted |
| `summary` | Roll-up: total tokens, total LLM calls, ground-truth match (when known) |

The trace shape is byte-compatible with the playback player in `demo/animation.js`, so any real run can be replayed in the playground UI as an animation.

---

## Python API

```python
import asyncio
from server.agent import run_case

case = {
    "case_id": "ID_77541",
    "paragraph": (
        "Our proposed kernel estimator achieves a parametric rate of convergence "
        "on subsets, which is faster than the non-parametric rate reported by "
        "Yao et al. (2005). In contrast to their approach, our method does not "
        "require functional principal component analysis."
    ),
    "ref_clean_citation": "Yao et al. (2005)",
    "ground_truth": "Comparison",
}

trace = asyncio.run(run_case(case, provider="anthropic"))

print(trace["final_artifact"]["label"])         # Comparison
print(trace["final_artifact"]["confidence"])    # 0.70
print(trace["summary"]["llm_calls"])            # 5
print(trace["summary"]["total_tokens"])         # ~600

for ev in trace["final_artifact"]["evidence"]:
    print(f"  ✓  \"{ev['span']}\"  [{ev['method']}]")
```

---

## Repository layout

```
citent/
├── server/                       FastAPI backend (the real agent)
│   ├── server.py                 app + route handlers
│   ├── agent.py                  seven-stage orchestrator
│   ├── tools.py                  Triage (TF-IDF + LR), regex tools, NLI verifier
│   ├── llm_clients.py            Anthropic, OpenAI, Google, DeepSeek
│   ├── train_models.py           re-trains Triage from raw CSVs
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── README.md
│   └── models/                   trained artefacts (ship pre-built)
│       ├── tfidf_vectorizer.pkl  ≈35 MB
│       ├── lr_classifier.pkl     ≈300 KB
│       ├── label_encoder.pkl
│       └── metrics.json          held-out accuracy / F1
│
├── demo/                         playground frontend
│   ├── index.html
│   ├── style.css
│   ├── animation.js              trace player + UI glue
│   ├── live-agent.js             browser-direct LLM client (static fallback)
│   └── traces/case_*.json        pre-recorded sample runs
│
├── prototype/juror.py            Python prototype (stdlib only, no API keys)
├── architecture.svg              system diagram
├── juror_demo_standalone.html    single-file playground (static-only mode)
├── bundle.py                     regenerates the standalone HTML
├── DEPLOYMENT.md                 engineer-facing deployment guide
└── README.md                     this file
```

---

## Evaluation

The Triage classifier is evaluated on a stratified 20 % hold-out of the 340 K-instance corpus:

| Metric | Value |
|---|---|
| Accuracy | 0.6932 |
| Macro-F1 | 0.5952 |
| Weighted-F1 | 0.7168 |

Per-class precision / recall / F1 and the full confusion matrix live in `server/models/metrics.json`. The classifier feeds the Prosecution Panel with a label shortlist; downstream stages then use LLM advocacy + symbolic verification to pick the winner among shortlisted candidates.

Re-train on a different corpus with:

```bash
python server/train_models.py \
       --citations /path/to/Citations.csv \
       --labels    /path/to/labels.csv \
       --max-features 10000
```

---

## Tests

```bash
# Smoke test — server boots, all routes return 200
python -m uvicorn server:app --port 8888 &
sleep 3
for p in / /api/status /api/cases; do
  curl -s -o /dev/null -w "%{http_code}  $p\n" http://127.0.0.1:8888$p
done

# Triage smoke — trained model loads and predicts correctly on case 3
python -c "
from server.tools import get_triage
r = get_triage().predict('... in contrast to their approach ...')
assert r['predicted_label'] == 'Comparison', r
print('triage ok')"
```

A complete integration test that exercises every stage end-to-end against the bundled cases is in `prototype/juror.py` — it runs in deterministic-only mode (no LLM key required) and writes a trace JSON identical in shape to the server's output.

---

## Further reading

- [🌐 Live playground](http://8.141.116.119:7868) — try it without installing anything
- [🎬 Demo video](https://youtu.be/EBAGmQXdnfo) — the full pipeline in 1.5 minutes
- [Architecture diagram](./architecture.svg) — seven-stage pipeline, role responsibilities, engine partition
- [Deployment guide](./DEPLOYMENT.md) — server-real vs static-only modes, Docker, CSP, troubleshooting
- [Award C submission narrative](./AWARD_C_SUBMISSION.md) — STAI-X 2026 framing
- [Demo README](./demo/README.md) — frontend layer specifics
- [Server README](./server/README.md) — backend internals
