# JUROR · Courtroom Demo

Single-page web demo that **plays back a real run** of the JUROR prototype as a courtroom animation: characters light up, speech bubbles pop, fact-check stamps land, the judge delivers a verdict, the clerk seals the file.

## Run

The browser blocks `fetch()` from `file://`, so serve the folder over HTTP:

```bash
cd demo/
python -m http.server 8000
```

Then open <http://localhost:8000>.

## Files

```
demo/
├── index.html          single-page layout
├── style.css           courtroom styling + animations
├── animation.js        trace player (no dependencies)
├── traces/             JSON traces produced by ../prototype/juror.py
│   ├── case_001.json
│   ├── case_002.json
│   └── case_003.json
└── README.md
```

## What you see

| Region | Content |
|--------|---------|
| Top bar | brand, STAI-X 2026 · Award C tag |
| Case bar | case ID, cited reference, ground truth, full paragraph |
| Courtroom (left) | nine actors in courtroom layout — judge on top, advocates middle, cross-examiner and fact-checker behind them, triage/investigator/clerk on the bottom row. Each one lights up when active. |
| Transcript (right) | live log: who spoke, which tool was called, what was verified — with engine chips (ML, RULE, GRAPH, NLI, LLM) |
| Meters | active engine, tools invoked, LLM calls, LLM tokens, current stage, final verdict |
| Artifact drawer | slides in when the Clerk seals the case — shows the JSON annotation artifact |

## Controls

- `▶ Play`, `⏸ Pause`, `⟲ Restart`
- Speed: `0.5×` / `1×` / `2×` / `4×`
- Case selector: ①, ②, ③
- Keyboard: `Space`/`k` toggle play, `r` restart, `1`/`2`/`3` switch case

## Visual cues

- **Active actor** has a bright outline + scale-up + colored glow.
- **Speech bubbles** pop in from the actor and self-dismiss.
- **Stamp animation** (✓ VERIFIED / ✗ UNVERIFIED) lands on each cited evidence span during the Fact-Checker stage.
- **Red diagonal strike** appears on advocates rejected by Cross-Examiner.
- **Golden halo** appears on the winning advocate when Judge delivers verdict.
- **LLM call / token counters** pulse purple when an LLM is invoked — the rest of the time, they stay at 0.

## What story this tells

Most of the runtime, the LLM counters stay at 0 — every action you see is ML / regex / graph / NLI. Only the Prosecution Panel and the Judge actually call an LLM. The point lands visually: this is an agent that **uses LLMs sparingly**, not a chatbot in seven hats.

## Regenerate traces

If you change the schema or add a new sample case to `../prototype/juror.py`:

```bash
cd ../prototype/
python juror.py --case 1
python juror.py --case 2
python juror.py --case 3
```

The traces overwrite the JSON files in `demo/traces/`, and the demo will pick up the new ones on the next reload.
