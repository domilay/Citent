# JUROR · Prototype

A working Python prototype of the JUROR labeling agent. Stdlib-only, no API keys, runs in under a second on a sample case.

The prototype's output is a **JSON trace file** that the `../demo/` web app plays back as a courtroom animation.

## Run

```bash
cd prototype/
python juror.py --case 1     # writes ../demo/traces/case_001.json
python juror.py --case 2     # writes ../demo/traces/case_002.json
python juror.py --case 3     # writes ../demo/traces/case_003.json
```

Each invocation prints a one-line verdict and writes the trace JSON used by the demo.

## What's inside

A single file (`juror.py`) organised top-down:

| Section | Contents |
|---------|----------|
| `SCHEMA` | the schema card: labels, rhetorical priors, abstain threshold |
| `SAMPLE_CASES` | three hand-picked citation instances with ground truth |
| Tools | `tfidf_lr_classifier`, `rhetorical_marker_regex`, `section_classifier`, `co_citation_lookup`, `exact_span_match`, `bleu_overlap` |
| Stages | seven `stage_*` functions, one per courtroom role |
| `run_case` | the orchestrator that wires the stages together |
| CLI | `--case`, `--out` |

The "LLM" stages (panel advocates, cross-examiner critique, judge arbitration) are simulated with deterministic heuristics so the trace is reproducible without API calls. The function boundaries are clean — drop a real LLM client into `_build_argument_text`, `stage_examiner`, and `stage_judge` to graduate from prototype to production.

## Trace format

```json
{
  "case_id": "ID_183878",
  "input":   { "paragraph": "...", "cited_ref": "...", "ground_truth": "..." },
  "schema":  { "task": "citation_intent", "labels": [...] },
  "stages": [
    {
      "id": "triage",
      "name": "Triage Officer",
      "engine": "ML",
      "tools_called": ["tfidf_lr_classifier"],
      "duration_ms": 0.4,
      "tokens": 0,
      "llm_calls": 0,
      "outputs": { ... },
      "actions": [
        { "t": 0,   "type": "say",  "actor": "triage", "text": "..." },
        { "t": 250, "type": "tool_call", "tool": "...", "engine": "ML", "summary": "..." },
        ...
      ]
    },
    ... (six more stages)
  ],
  "final_artifact": { "label": "Background", "confidence": 0.99, "evidence": [...] },
  "summary": { "total_tokens": 680, "llm_calls": 6, ... }
}
```

Each stage's `actions` are pre-timed (`t` is ms from stage start); the demo player concatenates stages into a global timeline and fires actions on a `requestAnimationFrame` loop.

## Bundled cases

| ID | Citation | Ground truth | Why interesting |
|----|----------|--------------|-----------------|
| ID_183878 | Hardle (1989) — confidence bands | **Background** | classic "has been used by, among others, X" framing — Triage gates it cleanly |
| ID_99812  | Buehlmann and Hothorn (2007) — boosting | **Technical basis** | "we follow / we extend" — Methods-section signal |
| ID_77541  | Yao et al. (2005) — convergence rates | **Comparison** | "faster than", "in contrast to" — Comparison advocate wins despite Triage uncertainty |

All three resolve correctly with the deterministic heuristics — they're chosen to showcase the courtroom flow, not to claim accuracy.
