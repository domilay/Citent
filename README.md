# JUROR — A Schema-Pluggable Annotation Agent

Submission for **STAI-X 2026 · Award C (Statistical Agents)**.

JUROR is a courtroom-themed multi-agent system for citation-intent labeling.
Seven specialised roles — Triage, Investigator, Prosecution Panel, Cross-Examiner,
Fact-Checker, Judge, Clerk — collaborate to produce **auditable annotation
artifacts** instead of bare LLM labels.  Four of the seven roles call no LLM at
all; the LLM is reserved for the hard tail.

## Layout

```
agent_design_v2/
├── README.md                          this file
├── architecture.svg                   the system diagram
├── bundle.py                          rebuilds the standalone HTML
├── juror_demo_standalone.html         single-file demo (just double-click)
├── AWARD_C_SUBMISSION.md              what we ship to the contest
├── prototype/                         deterministic Python prototype
│   ├── juror.py
│   └── README.md
├── demo/                              web demo source (Demo + Live modes)
│   ├── index.html
│   ├── style.css                      premium UI (glassmorphism + aurora)
│   ├── animation.js                   trace player + UI glue
│   ├── live-agent.js                  real-LLM runtime (Anthropic / OpenAI / Google / DeepSeek)
│   ├── traces/                        pre-recorded demo runs
│   └── README.md
└── video/script.md                    AI-video shooting script
```

## Two ways to run

### A. Demo mode (zero setup)

Just **double-click `juror_demo_standalone.html`**. The page opens in your
default browser. Pick a case (`01` / `02` / `03`), click **Run** — the agent
plays back a pre-recorded execution as an animated workflow. No API key,
no internet needed.

### B. Live mode (real LLM calls)

For when you want to see JUROR actually call an LLM in front of you.

1. Open `juror_demo_standalone.html`.
2. Click **Settings** (top-right).
3. Pick a provider: Anthropic / OpenAI / Google / DeepSeek.
4. Paste your API key.  *(The key stays in this browser tab and is sent only to
   the provider's own endpoint — no proxy, no server.)*
5. Click **Activate Live**. The mode switch flips to **Live**.
6. Click **Run**. The Prosecution Panel, Cross-Examiner, and Judge now use your
   real LLM; the non-LLM stages (Triage, Investigator, Fact-Checker, Clerk) run
   locally in JS.

Live mode produces a real, unique trace every run. Token counters reflect actual
usage. Total cost: roughly 5 LLM calls × ~150 tokens each per case ≈ a fraction
of a cent for most providers.

| Provider | Default model id | Notes |
|----------|------------------|-------|
| Anthropic | `claude-opus-4-6` | uses `anthropic-dangerous-direct-browser-access: true` |
| OpenAI    | `gpt-5.4`         | direct CORS supported |
| Google    | `gemini-3.5-flash`  | uses Generative Language API (`generateContent`) |
| DeepSeek  | `deepseek-chat`   | OpenAI-compatible endpoint |

You can override the model id in the Settings drawer before activating.

## The 7-stage pipeline

| # | Stage | Engine | LLM? | Job |
|---|-------|--------|------|-----|
| 01 | Triage | ML (TF-IDF + LR) | no | shortlists candidate intents, routes easy cases to fast-track |
| 02 | Investigator | regex + ML + graph | no | gathers structured evidence (rhetorical markers, section type, co-citations) |
| 03 | Prosecution Panel | **LLM × N** | yes | one advocate per shortlisted intent — each argues why it's the right label |
| 04 | Cross-Examiner | rule + **LLM** | yes | attacks weak / unsupported arguments |
| 05 | Fact-Checker | exact + BLEU + NLI | no | verifies every quoted span against the source; drops fabrications |
| 06 | Judge | **LLM** + Platt | yes | weighs survivors, calibrates confidence, produces verdict |
| 07 | Clerk | pure code | no | seals the annotation artifact (JSON) |

**4 of 7 stages call no LLM.** The visible counter in the demo stays at 0 until
the Panel speaks (~25 s into a run), making the "not just an LLM wrapper" point
unmissable.

## The annotation artifact

Every run ends with a JSON artifact:

```json
{
  "case_id": "ID_77541",
  "label": "Comparison",
  "confidence": 0.70,
  "margin": 0.50,
  "abstain": false,
  "evidence": [
    { "span": "in contrast to their approach",          "verified": true, "method": "exact" },
    { "span": "faster than the non-parametric rate",    "verified": true, "method": "exact" }
  ],
  "rejected_alternatives": [
    { "label": "Background",      "reason": "single weak marker; insufficient signal" },
    { "label": "Technical basis", "reason": "no adoption verb in paragraph" }
  ],
  "ground_truth": "Comparison"
}
```

The artifact is the **deliverable**: replayable, auditable, abstention-aware.
A reviewer can replay the trace and confirm each piece of evidence by hand.

## Selling points (for Award C reviewers)

1. **Genuine agent, not an LLM wrapper.** 4 of 7 stages call no LLM. The LLM is
   reserved for adversarial argumentation.
2. **Reusable analytical component.** The Schema Card is the only thing that
   changes between domains. Citation intent today, overdose ED-signal labeling
   tomorrow — directly aligned with Award C's mandate.
3. **Reuses existing ML baselines as first-class tools.** TF-IDF + LR runs
   inside the agent as the Triage gate.
4. **Evidence-grounded.** Every label is backed by quote-level verification.
5. **Adaptive compute.** Triage routes easy cases to a cheap path.
6. **Auditable output.** The artifact is the contribution.
7. **Live runnable.** Paste your own API key and watch the LLM stages run for
   real against your provider of choice.

## Rebuilding the standalone HTML

If you edit anything under `demo/`, rebuild the single-file bundle:

```bash
python bundle.py
```

This regenerates `juror_demo_standalone.html` with the latest CSS, JS, traces,
and live-agent runtime inlined.

## Regenerating demo traces

The traces in `demo/traces/` come from running the prototype on the sample
cases. To re-record them:

```bash
cd prototype/
python juror.py --case 1
python juror.py --case 2
python juror.py --case 3
```

Each invocation overwrites the matching `demo/traces/case_NNN.json`. Then
re-run `python bundle.py` to refresh the standalone file.
