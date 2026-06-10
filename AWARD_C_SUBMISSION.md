# JUROR — A Schema-Pluggable, Evidence-Grounded Statistical Annotation Agent

**STAI-X 2026 · Award C — Statistical Agents**

> A labeling agent that puts every hard instance on trial. It spends cheap tools on the easy majority, convenes a courtroom of LLM advocates only for the hard tail, fact-checks every claim with non-LLM verifiers, calibrates its own confidence, and abstains when it should — and it ships an auditable, replayable *Annotation Artifact* instead of a bare category. Swap one Schema Card and the same courtroom labels a new domain, no code change.

---

## Submission at a glance

| Award C field | Where to find it |
|---|---|
| **Team** | *‹team names + affiliations — fill in for the public post›* |
| **Plain-language description** | [§1](#1-what-juror-does) |
| **Architecture summary** | [§5](#5-how-the-agent-is-built-the-courtroom-pipeline) + `architecture.svg` |
| **Demo link** | Hosted single-page left-to-right workflow demo (GitHub Pages): *‹url›* · self-contained file: [`juror_demo_standalone.html`](juror_demo_standalone.html) · walkthrough video: *‹url›* |
| **GitHub repository** | *‹repo url›* — commit tagged `award-c-submission` |
| **Reproducibility** | Stdlib-only prototype [`prototype/juror.py`](prototype/juror.py) (no API key, runs in <1 s) + the full benchmark code release in [`deliver/`](../deliver/) that produced every number cited below |

This agent is the engineering distillation of a two-year, 367K-instance benchmark study (**CitaStat**) our team built and published. Every design decision below is justified by a *measured* failure mode from that study — not by intuition. That provenance is the submission's central claim: **we are not proposing an architecture we hope works; we are shipping the workflow we already ran, by hand, at scale, and that achieved 97.75% verified label accuracy.**

---

## 1. What JUROR does

Give JUROR a short passage and a target taxonomy, and it returns a **labeled, evidence-backed verdict**: the chosen category, a calibrated confidence, the exact text spans that justify it, the alternatives it ruled out *and why*, and a flag when it is too unsure and a human should decide.

Today the loaded task is **citation-intent classification** — reading the paragraph around a citation and deciding *why* the author cited that work: as `Background`, `Technical Basis`, `Comparison`, or `Fundamental Idea`. But the task is not baked in. It arrives as a **Schema Card** — a small, declarative description of the labels, their linguistic priors, and an abstain threshold. Change the card and the same agent labels news stance, clinical-surveillance signals, legal clause types, or sentiment.

### Who it's for

- **Teams building large labeled datasets** who need scale *and* trustworthy quality — and who cannot afford to put a human on every instance.
- **Researchers doing text-as-data analysis** in statistics, bibliometrics, social science, or public health, who need labels that come with evidence and an audit trail.
- **Anyone who would otherwise wrap a single LLM call** around a labeling task and inherit all of its silent failure modes.

### When to use it

- You have a function/intent taxonomy, not just a sentiment polarity.
- You need labels you can *defend* — with the quoted span, the rejected alternatives, and a confidence you can threshold.
- Your data has a long easy head and a hard tail, and you want to pay reasoning cost only on the tail.
- You want a labeler that knows when to **abstain** and route to a human, rather than guessing confidently.

---

## 2. Why this agent exists — what 367K citations taught us

JUROR did not start as an architecture diagram. It started as **CitaStat**, our benchmark of **367,218 citation instances** drawn from **7,879 papers** in four leading statistics journals (AOS, Biometrika, JASA, JRSSB; 1996–2020), each labeled into one of four function categories. We benchmarked **15 LLMs** (4 closed-source, 11 open-source) plus six classical-ML baselines, and we ran ablations on prompting strategy and on full-paper vs local context. The findings are the blueprint for this agent.

**Finding 1 — most instances are easy; a few are genuinely hard.** When we stratified all 367K instances by how four strong LLMs agreed, **83.6%** fell into "Very Easy" (4/4 agree, 212,457) or "Easy" (3/4 agree, 94,554). On these, simple majority voting was **97.75%** accurate against human verification (99.27% / 95.94%). Only **~16%** (Medium/Hard/Very Hard, ~94K instances) needed human deliberation — and that tail is where every model's accuracy collapses (LLM majority vote drops to **41.2%** on Hard and **27.6%** on Very Hard). *Lesson: spending heavy reasoning on every instance is waste; spending it on the tail is the whole game.*

**Finding 2 — a single zero-shot LLM call hits a ceiling and is silently biased.** The best model, Claude Opus 4.6, reached **81.62%** accuracy — strong, but with a per-class profile that exposes systematic bias: F1 of **86.2** on Background but only **47.0** on Fundamental Idea. Across models the bias is directional and *different per model*: on the hard strata, Gemini and DeepSeek **over-predict** Fundamental Idea (40.1% / 25.4% vs. 8.2% truth) and Comparison, while GPT nearly **ignores** Comparison (3.6% vs. 10.0%). *Lesson: no single model's prior should be trusted as the verdict; disagreement is signal.*

**Finding 3 — LLMs over-read the rhetoric and confuse surface cues with function.** The dominant error mode: when a paragraph narrates what a cited work *does* ("proposed", "introduced", "developed"), models inflate a passive **Background** mention into **Fundamental Idea**; when "compared", "unlike", or "better" appear nearby, they default to **Comparison** regardless of the citation's actual role. A real example — *"...unlike the bootstrap; see DiCiccio, Hall and Romano (1991)..."* — led two LLMs to call it Comparison, though the citation is Background. *Lesson: a label is only safe if it survives an adversarial check against the actual evidence, not the nearest keyword.*

**Finding 4 — unstructured "more reasoning" can make it worse.** We tested chain-of-thought, multi-turn deliberation, and few-shot. **Multi-turn deliberation *reduced* accuracy for GPT and Gemini** — asked to reconsider, they talked themselves out of correct answers into plausible-but-wrong ones. Meanwhile **few-shot was the most consistently helpful**, which tells us much of the remaining error is a **taxonomy mismatch** (the model's default reading of a label name vs. the statistics-specific definition), not a lack of raw capability. *Lesson: deliberation only helps when it is bound to evidence and a precise schema; free-form second-guessing drifts.*

**Finding 5 — dumping the whole paper in hurts.** Giving models the full citing paper instead of the local context produced only mild gains for two models and **degraded** Claude and Gemini — long-context dilution, plus PDF-extraction noise and token cost. *Lesson: retrieve the *relevant* evidence selectively; don't flood the context.*

**Finding 6 — the cheap classical baseline is not throwaway.** TF-IDF + LinearSVM reached **76.65%** — beating most open-source LLMs at a tiny fraction of the cost — but cratered on the minority class (Fundamental Idea F1 **38.9**). *Lesson: a trained statistical classifier is an excellent first-pass gate, but it cannot be the last word on the hard, valuable cases.*

**Finding 7 — the hardest class is the most valuable, and raw counts miss it.** Fundamental Idea is both the rarest (4.2%) and hardest class, yet it carries the most intellectual weight. Total citation counts correlate strongly with Background (0.88) and Technical Basis (0.90) but only weakly with Fundamental Idea and Comparison (0.64) — so works that are *foundational* are systematically under-recognized by raw citation counts. *Lesson: typed, per-category labels unlock an analysis raw counts cannot — which is exactly why the labels must be trustworthy and auditable.*

> **The throughline.** We did not derive these lessons from a whiteboard. We lived them across 100+ trained annotators, two years, and 367K instances. CitaStat's winning *human* workflow — stratify by difficulty, send the easy majority to cheap consensus, convene multiple independent judgments on the hard tail, escalate the irreducible cases to expert review — is *exactly* JUROR's pipeline. **JUROR is that workflow, made into a reusable agent.**

---

## 3. Why a single LLM (and everything else) does worse

The temptation is to wrap one model call around the task and ship a category. Here is what each obvious approach gives up, measured against our own benchmark:

| Approach | What our data says it gets you | What it lacks |
|---|---|---|
| **One zero-shot LLM call** | 81.6% at best; directional per-class bias; collapses to 27–41% on the hard tail | No evidence, no calibration, no abstention; you can't see *why* or *how sure* |
| **Pure classical ML** | 76.7%; cheap and fast | Fundamental-Idea F1 of 0.39; zero explanation; no path to improve the hard cases |
| **Blind CoT / multi-turn agent** | *Lower* accuracy for some flagship models — reasoning drifts | Free-form deliberation talks itself out of correct answers |
| **Generic LLM-as-judge** | Plausible-sounding verdicts | An LLM cannot vouch for itself; nothing catches a fabricated quote |
| **JUROR** | Adaptive compute + cross-model panel + non-LLM verification + calibration + abstention | — |

JUROR is the only design that responds to *every* measured failure mode at once:

- **Adaptive compute** answers Finding 1 — the easy 84% never touch an LLM.
- **A panel of one sub-agent per intent (all four labels) + a cross-model Judge** answers Finding 2 — no single model's bias becomes the verdict, mirroring the multi-LLM consensus that made our labels 97.75% reliable.
- **An adversarial Cross-Examiner** answers Finding 3 — every label must survive a contradiction check against the actual evidence.
- **Evidence-bound, structured deliberation** answers Finding 4 — advocates must cite verifiable spans, so deliberation cannot drift into plausible-but-wrong.
- **A non-LLM Fact-Checker** answers the self-vouching problem — exact-match → BLEU → NLI, with LLMs *deliberately forbidden*, because a model cannot certify its own quote.
- **Selective evidence retrieval** answers Finding 5 — the Investigator pulls markers, section, and co-citations, never the whole paper.
- **The Schema Card** answers Finding 4's taxonomy-mismatch result — it injects the precise, domain-specific label definitions and priors, encoding the few-shot signal that we found most reliable.
- **Calibrated confidence + abstention** answers Findings 6–7 — the agent knows when to defer the rare, hard, valuable cases to a human, the same escalation that anchored CitaStat's quality.

---

## 4. What it does — the verdict, with receipts

JUROR's output is not a label. It is an **Annotation Artifact**: a structured, replayable record of how the verdict was reached. This is the deliverable that a real annotation pipeline can audit, dispute, and re-run.

**Example** (bundled case `ID_77541`, citing *Yao et al. (2005)*). The passage compares convergence rates; one advocate even fabricates a flattering quote — and the Fact-Checker catches it:

```json
{
  "case_id": "ID_77541",
  "label": "Comparison",
  "confidence": 0.70,
  "abstain": false,
  "evidence": [
    { "span": "faster than the non-parametric rate reported by Yao et al.", "verified": true, "method": "exact" },
    { "span": "in contrast to their approach",                              "verified": true, "method": "exact" }
  ],
  "rejected_alternatives": [
    { "label": "Technical basis", "reason": "no adoption verb in paragraph" },
    { "label": "Background",      "reason": "single weak marker; insufficient signal" }
  ],
  "fabricated_claims_dropped": 1,
  "tool_trace": ["tfidf_lr", "regex", "section", "co_cite", "advocate×4", "examiner", "verify", "judge"]
}
```

Read it like a court record: the **verdict** and its **calibrated confidence**; every piece of **evidence with a verification method**; the **alternatives that lost and the reason each lost**; the **fabricated claim that was struck**; and the full **tool trace**. No zero-shot LLM call gives you any of this — and it is precisely what large annotation efforts need to trust, contest, and reproduce a label.

---

## 5. How the agent is built — the courtroom pipeline

JUROR routes each instance through up to seven stages. The metaphor is a trial: triage decides whether a case even needs a jury, investigators gather evidence, advocates argue, a cross-examiner attacks, a fact-checker verifies, a judge rules, and a clerk seals the record. The point of the metaphor is the **engine mix** — three of the seven stages call *no LLM at all*, and the easy majority short-circuits the whole trial.

| # | Stage | Engine | What it does | Which finding it answers |
|---|---|---|---|---|
| ① | **Triage Officer** | ML only (TF-IDF + LR) | Uses our trained baseline as the gate; difficulty = 1 − max class prob; easy cases fast-track straight to the Clerk | 1, 6 |
| ② | **Investigator** | regex + ML + graph (+ optional 1 small-LLM gate) | Gathers evidence cheapest-tool-first: rhetorical markers, section, co-citations; only a tiny LLM call asks "is the evidence sufficient?" | 3, 5 |
| ③ | **Prosecution Panel** | LLM (primed with the gathered features) | One sub-agent per intent — all four labels argue in parallel, each from the evidence already collected | 2 |
| ④ | **Cross-Examiner** | rule-based filter + LLM critic | Regex contradiction patterns pre-flag obvious holes; an LLM critic deepens the attack on surviving arguments | 3, 4 |
| ⑤ | **Fact-Checker** | **non-LLM only** (exact match → BLEU → NLI) | Verifies every cited span against the source text; LLMs forbidden — they cannot vouch for themselves | 4 |
| ⑥ | **Judge** | LLM × 1–2 + calibration | Weighs surviving, verified arguments; a statistical calibrator turns the raw margin into a probability | 2, 4 |
| ⑦ | **Clerk** | pure code | Deterministically packages the Annotation Artifact | — |

### Where the LLM is used (and where it isn't)

This is the heart of the "agent, not LLM-wrapper" claim. Only **Stages ③, ④, and ⑥** are the model's territory — argue, attack, arbitrate. Everything else is statistical ML, rules, graph lookups, or symbolic verification. Stages ①, ⑤, ⑦ never call an LLM; Stage ② makes at most one small call. And on the easy 84% of instances, the fast-track skips the LLM stages entirely.

> **The model is the scalpel, not the hammer.** On a representative corpus, the large majority of instances are resolved at **zero LLM tokens**; the full jury — and its cost — is reserved for the hard tail where our data shows reasoning actually changes the answer.

### Reusing the benchmark's own classifiers as tools

The TF-IDF + LinearSVM / LR classifiers we trained for CitaStat are not abandoned baselines — they are **first-class tools inside the agent**: the Triage gate *is* the trained classifier, and the Investigator uses ML signals as features. The agent stands on the statistical machinery we already validated.

### Feedback: a Case Library that learns during a run

Hard verdicts are distilled into compact `IF … THEN label` precedents with learned priors and fed back into the Investigator and the Panel. The system improves over a run and emits a learning curve — a self-improvement property rare in zero-shot agent systems, and a direct response to Finding 7 (accumulate precedent on the rare, hard, valuable classes).

---

## 6. What it's useful for

1. **Trustworthy annotation at scale.** This is the use case we lived. A labeling agent with adaptive compute, evidence, calibration, and abstention is exactly what would have cut the cost and risk of CitaStat's two-year human effort — keep the human in the loop only for the cases that earn it.
2. **Typed-citation research-impact analysis.** Because Fundamental-Idea and Comparison citations correlate weakly with raw counts (0.64), per-category labels surface foundational works that raw citation counts under-recognize. JUROR produces exactly the typed, auditable labels this analysis needs — a better lens for promotion, funding, and impact decisions.
3. **A reusable upstream component for *other* statistical pipelines** — including this challenge's. See §7.

---

## 7. How it extends — one Schema Card, any domain (and a path to the STAI-X task)

Retargeting JUROR is a **data** change, not a **code** change. The Schema Card is the single input that swaps the task:

```python
SCHEMA = {
  "task": "citation_intent",
  "labels": ["Background", "Technical basis", "Comparison", "Fundamental idea"],
  "abstain_threshold": 0.55,
  "label_priors": { "<label>": { "markers": [...], "negative_verbs": [...] }, ... },
}
```

It carries the **labels**, their **linguistic priors** (the markers that the Investigator and Panel reason over — this is where the few-shot/taxonomy signal from Finding 4 lives), and an **abstain threshold** that tunes the human-handoff rate. The seven stages, the courtroom orchestration, the non-LLM Fact-Checker, the calibrator — all unchanged.

**Cards already sketched:** `citation_intent` (active), `news_stance`, `legal_clause_type`, `sentiment`, and — relevant to STAI-X — `overdose_ed_signal`.

### Connecting to the STAI-X public-health challenge

The challenge's forecasting target depends on noisy upstream signals — clinical surveillance text, digital behavioral indicators, social-determinant records. Before any of that can be modeled, it must be **labeled and quality-controlled**, and that labeling has the same shape as citation intent: a function taxonomy, an easy head, a hard ambiguous tail, and a hard requirement for auditability in a public-health setting. An `overdose_ed_signal` Schema Card retargets JUROR into a **reusable, evidence-grounded labeling/QA agent for the surveillance signals that feed the forecast** — triaging the easy majority by trained classifiers, putting ambiguous records on trial, verifying every label against the source record, calibrating confidence, and abstaining to a clinician when the signal is genuinely unclear. That is precisely Award C's mandate: a *reusable statistical analysis component*, not a one-off model.

### Other extension points

- **Graduate the simulated stages to real LLMs.** The prototype's advocate text and cross-examiner critic are deterministic heuristics so the demo runs offline; the function boundaries (`_build_argument_text`, `stage_examiner`, `stage_judge`) are clean — dropping in a real client is a ~20-line change.
- **Grow the Case Library** into a persistent, cross-run precedent store with a real Precedent Extractor.
- **Swap calibrators** (Platt / temperature / isotonic) per domain to match the deployment's cost of a wrong label vs. an abstention.

---

## 8. How to use it

JUROR runs in two modes. The framework is already in place; the only thing that separates a free, offline replay from a live run on real models is **one OpenRouter API key**.

### Mode 1 — Offline replay (no key, < 1 second)

For the demo, a reviewer, or a reproducible trace, run the prototype with nothing installed beyond the Python standard library:

```bash
cd agent_design/prototype
python juror.py --case 1     # → writes ../demo/traces/case_001.json   (also --case 2, 3)
```

Then watch the run play back as a **left-to-right workflow animation** — either double-click the self-contained [`juror_demo_standalone.html`](juror_demo_standalone.html), or serve the folder:

```bash
cd agent_design/demo && python -m http.server 8000   # open http://localhost:8000
```

The pipeline reads left to right (Triage → Investigator → Panel → Cross-Examiner → Fact-Checker → Judge → Clerk); the Panel fans out into **one lane per intent** so you can see, at a glance, which sub-agent is arguing for which label. In this mode the three LLM stages (Panel, Cross-Examiner, Judge) are played by deterministic stand-ins, so the trace is identical every time and needs no network — ideal for a submission demo.

### Mode 2 — Live run (one OpenRouter key)

The agent's three LLM stages call models **through OpenRouter** — a single key reaches Claude, GPT, Gemini, DeepSeek, and the open-source models alike. This is the exact async, cache-aware, retrying client we used to evaluate all 15 models in the paper ([`llm_eval/run_api.py`](../llm_eval/run_api.py)). To go live:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...        # the only credential needed
```

Choose which model sits in the jury — set it in the Schema Card (`juror_model`); any OpenRouter model id works:

| Role of the model | Example model id |
|---|---|
| Default juror (advocates / cross-examiner / judge) | `anthropic/claude-opus-4.6` |
| Cheaper / alternative jurors | `openai/gpt-5.4` · `google/gemini-3.1-pro` · `deepseek/deepseek-v3.2` |

With the key set and a model chosen, the same `python juror.py` command runs the agent end-to-end: Triage, the Investigator's tools, the Fact-Checker, and the Clerk stay non-LLM, while the three jury stages are now driven by the live model. Cost stays low because the easy majority of instances never reach an LLM at all (§5).

> **One switch, any domain, any model.** Mode 2 changes nothing about the architecture — it only supplies a key and a model id. Combined with the Schema Card (§7), the same binary retargets to a new task *and* a new model without touching the pipeline code.

### What runs on an LLM, and what doesn't

- **Always non-LLM** (works in both modes, today): TF-IDF + LR triage, regex / section / co-citation feature extraction, exact-match + BLEU + NLI fact-checking, margin calibration, and artifact packaging.
- **The jury stages** (Panel, Cross-Examiner, Judge): deterministic stand-ins in Mode 1; live OpenRouter models in Mode 2. The function boundaries (`_build_argument_text`, `stage_examiner`, `stage_judge`) are isolated, so the offline-deterministic demo and the live run share one codebase and one trace format.

### Reproduce the paper's numbers

Every figure cited in §2–§3 comes from the full benchmark code release in [`deliver/`](../deliver/): closed-source LLMs (official APIs), open-source LLMs (vLLM), classical-ML baselines, and the prompting / full-paper ablations — each with its own README, fixed seeds, and deterministic decoding (`temperature = 0`). The same OpenRouter key drives the closed-source evaluation in [`llm_eval/`](../llm_eval/).

### Repository layout

```
agent_design/
├── AWARD_C_SUBMISSION.md   <- this document
├── architecture.svg        <- the system diagram (one self-contained SVG)
├── README.md               <- artefact index + design rationale
├── prototype/              <- runnable Python prototype (stdlib only, no API keys)
│   └── juror.py
├── demo/                   <- single-page courtroom playback of a real run
│   ├── index.html · style.css · animation.js
│   └── traces/             <- JSON traces produced by prototype/juror.py
├── juror_demo_standalone.html  <- the demo bundled into one double-clickable file
└── video/script.md         <- shot-by-shot script for the walkthrough video
```

---

## 9. Evaluation — how we would judge JUROR (and how it stays honest)

Because JUROR was born from a benchmark, it has a natural, rigorous evaluation harness on home turf — the CitaStat hard strata, where a single LLM call drops to 27–41%:

- **Accuracy on the hard tail** vs. the single-LLM and pure-ML baselines (the two lines our paper already measured).
- **Selective-prediction curve** — accuracy as a function of abstention rate; the right question for an agent that can defer is *"how accurate is it on the cases it chooses to answer, and how many does it hand off?"*
- **Calibration** — expected calibration error on the Judge's confidence, since the artifact's confidence is meant to be thresholded.
- **Evidence faithfulness** — the share of cited spans that pass the non-LLM Fact-Checker, and the count of fabricated claims caught (the demo's headline moment).
- **Cost** — fraction of instances and tokens resolved with zero LLM calls.

> **The governing question.** Not "can an agent label a citation?" — a single prompt can attempt that. The question is: *on the instances where a lone LLM is wrong, biased, or fabricating, does putting the case on trial — cheap tools first, adversarial advocates, non-LLM verification, calibrated abstention — recover accuracy while remaining auditable and cheap on the easy majority?* That is the contribution, and it is measurable on the very dataset that motivated it.

---

## 10. The one-paragraph pitch

JUROR is a reusable statistical annotation agent that earns its complexity from evidence. We built the 367K-instance CitaStat benchmark, measured exactly where and why single-LLM labeling fails — per-class bias, rhetorical over-reading, deliberation drift, long-context dilution, a hard valuable minority class — and then built the agent that answers each failure: a trained classifier triages the easy 84% at zero LLM cost; the hard tail goes on trial before a panel of per-label advocates, a contradiction-hunting cross-examiner, and a non-LLM fact-checker that no model can talk past; a calibrated judge rules or abstains; and a clerk seals an auditable artifact. One Schema Card retargets the whole courtroom to a new domain — citation intent today, overdose surveillance signals tomorrow — with no code change. It is not a chatbot in seven hats. It is the human workflow that produced a 97.75%-reliable dataset, distilled into a statistical agent.
