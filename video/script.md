# JUROR — The Trial of a Citation
**Demo video script for AI-generated rendering**

> A ~95-second courtroom drama that walks the viewer through one citation being labeled by JUROR.
> Built around **Case ID_77541** — the citation of *Yao et al. (2005)* in a paragraph that compares convergence rates.

---

## Production notes (top-of-script briefing for the video model)

| Field | Value |
|-------|-------|
| **Total runtime** | ~95 seconds |
| **Aspect ratio** | 16:9, 1920×1080 |
| **Visual style** | Cinematic minimalism. A vast, dark, holographic courtroom — like the inside of a digital cathedral. Deep navy backgrounds, soft volumetric light, color-coded spotlights. Think "Tron meets a Pixar short." |
| **Character design** | Each agent role is a **stylized, anthropomorphic glowing card** (no realistic human faces — easier for AI video models to keep consistent). Every card has an icon, a name, and a colored aura matching its role: green Triage, orange Investigator, red Advocates, pink Cross-Examiner, purple Fact-Checker, teal Judge, gold Clerk. Cards float and tilt; their "mouth" is the icon area pulsing as they speak. |
| **Camera language** | Smooth dolly-ins, orbital moves around active speakers, low-angle hero shots on the Judge, top-down god-shot for the courtroom layout. |
| **Typography** | All on-screen text is sans-serif (Inter / SF Pro). HUD elements glow soft cyan. JSON shown in monospace gold. |
| **Music** | Tense legal-procedural underscore — sparse piano + low strings + occasional digital ticks. Builds in Scene 6. Resolves to ambient calm in Scene 9. |
| **Sound design** | Soft data-chime when a tool is invoked. Heavy "thud" on the UNVERIFIED stamp in Scene 6. Single gavel tap in Scene 7. Wax-seal hiss in Scene 8. |
| **Narration** | Single neutral voice (V.O.). Confident, slightly cool, documentary tone. All dialogue in English. |
| **Subtitles** | Always on, even for V.O., positioned as bottom-third. |

---

## Recurring on-screen HUD

A small, glowing HUD lives in the top-right corner the whole video. Updates between scenes:

```
┌────────────────────────────┐
│  CASE  ID_77541            │
│  ──────────────────────    │
│  STAGE      <current>      │
│  ENGINE     <ML/RULE/LLM>  │
│  TOOLS      <n>            │
│  LLM CALLS  <n>            │
│  TOKENS     <n>            │
└────────────────────────────┘
```

The two lines **LLM CALLS** and **TOKENS** are the visual centerpiece — they stay at 0 for the first ~25 seconds, then climb only when a real LLM stage fires.

---

# SCENE 1 — Cold Open (0:00 – 0:08)

**Setting.** Pitch-black void slowly fades up to reveal a vast digital courtroom from a god's-eye top-down view. Holographic chairs in a circular arrangement. A single paper-like document, glowing faintly, drifts down through the air toward the center of the room.

**Camera.** Slow descent from ceiling. The paper grows from a speck to a readable document by the end of the shot.

**On-screen text** (appearing as the paper lands, like floating annotations):
> CASE FILE — ID_77541
> "Our proposed kernel estimator achieves a parametric rate of convergence on subsets, which is faster than the non-parametric rate reported by **Yao et al. (2005)**. In contrast to their approach, our method does not require functional principal component analysis."
>
> CITED REFERENCE: Yao et al. (2005)
> LABEL UNKNOWN

**Narrator (V.O.):**
> "Every citation in every paper hides an intent. Was the cited work the *foundation*? The *method*? Just *background*? Or something the authors were *competing against*?"

(beat)

> "Today, our agent JUROR opens case 77541."

The HUD blinks on in the top-right with all counters at 0.

---

# SCENE 2 — Triage (0:08 – 0:16)

**Setting.** A small green-glowing pedestal at the front of the courtroom. Behind it, a card-shaped figure with a clipboard icon — TRIAGE OFFICER.

**Camera.** Whip-pan from the floating case file to the Triage pedestal. Quick close-up.

**Action.** The case file drops onto the pedestal. Triage's icon pulses. A holographic bar chart materializes above the pedestal, showing four candidate labels with bars of different lengths:

```
Comparison       ████████████████  0.45
Technical basis  █████████        0.25
Background       ███████          0.20
Fundamental idea ████             0.10
```

**Triage** (clipped, efficient voice — like a robotic dispatcher):
> "Difficulty zero-point-five-five. Top three candidates: Comparison, Technical basis, Background. Routing to full jury."

**On-screen badge fades in over the HUD:**
```
⚡ FAST TRACK ELIGIBLE? — NO
   Sending to full investigation.
```

**HUD update:**
```
STAGE     Triage Officer
ENGINE    ML (TF-IDF + LR)
TOOLS     1
LLM CALLS 0     ← still zero
TOKENS    0     ← still zero
```

The case file lifts off the pedestal and glides toward a larger desk in the center of the room.

---

# SCENE 3 — Investigator (0:16 – 0:28)

**Setting.** A large rectangular desk in the middle of the courtroom, glowing orange. Above it, a holographic **evidence board** unfolds — multiple translucent rectangles fanning out like cards in mid-air, each ready to display a different tool's output.

**Camera.** Slow orbit around the Investigator card, 270 degrees.

**Action.** Investigator's icon pulses. Three holographic tool panels light up sequentially with a soft chime each. As each tool fires, a green checkmark appears on its panel.

**Tool panels light up, one by one (~2.5 seconds each):**

1. `rhetorical_marker_regex` — text highlights pop up over the case paragraph: **"in contrast"**, **"faster than"**
2. `section_classifier` — output: **"Section: Experiments / Comparison · conf 0.68"**
3. `co_citation_lookup` — output: **"0 co-citations found in §Methods"**

**Investigator** (curious, almost detective-like, slightly warmer voice than Triage):
> "Two comparative markers in the paragraph. The surrounding language reads like an Experiments section. No Methods-section co-citations. Case file assembled — closing investigation."

The three panels collapse together into a single glowing dossier. It slides forward across the floor.

**HUD update:**
```
STAGE     Investigator
ENGINE    REGEX + ML + GRAPH
TOOLS     4
LLM CALLS 0     ← still zero
TOKENS    0     ← still zero
```

**Narrator (V.O.) (whispered, almost as a parenthetical):**
> "Twenty-eight seconds in — zero LLM calls so far."

---

# SCENE 4 — Prosecution Panel (0:28 – 0:50)

**Setting.** Three glowing red podiums rise from the floor in a row. Three red Advocate cards materialize behind them. Above each podium hovers the label they are arguing for.

**Camera.** Wide establishing shot of the three podiums. Then dolly in on each speaker in turn, one at a time.

**Action.** Each advocate speaks while their podium lights brighter and the other two dim. Speech bubbles with their argument materialize in mid-air.

The HUD's **LLM CALLS** counter ticks up by one with each advocate (sound: soft *bing*). Tokens counter climbs in 140-unit jumps.

---

### Advocate A (label: Comparison) — speaks first, confident

The figure tilts forward, almost smug.

**Advocate A:**
> "Your honor — the paragraph says **'faster than the non-parametric rate reported by Yao et al.'** and **'in contrast to their approach.'** Two textbook comparative markers."

(beat — they lean in, voice lowering, slightly conspiratorial)

> "And for emphasis: the authors **directly outperform the cited method on benchmarks**."

A faint shimmer of red runs around the third quote — the camera lingers on it for half a second longer than the others, a subtle *something's off*. The HUD does not flag it. Only the audience notices.

---

### Advocate B (label: Technical basis) — hesitant

**Advocate B:**
> "I argue Technical basis. The citing paper relies on the cited method as a foundation..."

(longer beat, the figure shifts uncomfortably)

> "...although, I concede, no adoption verb appears in the paragraph itself."

The figure shrinks slightly, light dimming.

---

### Advocate C (label: Background) — almost dismissive

**Advocate C:**
> "And I say Background. The cited work is part of the broader landscape of convergence-rate literature. Nothing more."

Brief, no flourish. Sits.

---

**HUD update:**
```
STAGE     Prosecution Panel
ENGINE    LLM
TOOLS     7
LLM CALLS 3     ← first jump!
TOKENS    420
```

The HUD glows purple briefly as the LLM counter increments.

---

# SCENE 5 — Cross-Examiner (0:50 – 1:00)

**Setting.** A pink-violet figure strides into the central space from offscreen. Sharp posture, formal court robe. CROSS-EXAMINER.

**Camera.** Tracking shot following the Cross-Examiner as they pace from advocate to advocate.

**Cross-Examiner** (cold, surgical, slightly faster pace than the others):
> "Advocate B. You admit no adoption verb. Your own argument concedes its weakness."

A red diagonal slash dramatically cuts across Advocate B's podium. The card visibly recoils.

**Cross-Examiner** (turns):
> "Advocate C. Your evidence is a single weak signal. Insufficient."

A second red slash crosses Advocate C. They sit, dimmed.

The Cross-Examiner pauses in front of Advocate A. The audience holds breath.

**Cross-Examiner** (after a beat):
> "Advocate A. Your case looks strong. But I do not have the authority to verify your claims."

(turning to address the entire courtroom)

> "Fact-Checker — your floor."

**HUD update:**
```
STAGE     Cross-Examiner
ENGINE    RULE + LLM
TOOLS     8
LLM CALLS 4
TOKENS    500
```

---

# SCENE 6 — Fact-Checker (1:00 – 1:16)  **← DRAMATIC PEAK**

**Setting.** A new figure glides into view — FACT-CHECKER, deep purple aura, no human voice yet, just a soft electronic hum. In their "hand" is a translucent stamp.

**Music.** Quiet falls. Only a soft pulse — almost a heartbeat.

**Camera.** Push in on the Fact-Checker. Then a tight shot of three floating evidence quotes hanging in mid-air, each from Advocate A.

**Fact-Checker** (slow, scientific, calm — slightly synthetic, deliberately *not* warm):
> "I do not argue. I verify."

(beat)

> "Every claim must appear in the source text."

The camera cuts to the three floating quotes. Each gets stamped in sequence — the stamp swings in with weight and lands with a **thud**.

---

**Quote 1:** *"faster than the non-parametric rate reported by Yao et al."*
→ Green **✓ VERIFIED** stamp lands. A soft chime.

**Quote 2:** *"in contrast to their approach"*
→ Green **✓ VERIFIED** stamp lands. A soft chime.

**Quote 3:** *"the authors directly outperform the cited method on benchmarks"*
→ The Fact-Checker pauses. The hum darkens. Camera tightens. The audience already suspects.
→ Heavy red **✗ UNVERIFIED** stamp lands with a thunderous percussive hit. The quote literally disintegrates into ash and blows away.

---

**Fact-Checker:**
> "This passage does not appear in the source paragraph. The claim is dropped."

The Advocate A figure flinches but does not fall — their other two pieces of evidence still glow green behind them.

A SUB-BANNER appears just under the HUD, in green text:
```
⚖  Even the winning side is fact-checked.
    The Fact-Checker uses no LLM — LLMs cannot vouch for themselves.
```

**HUD update:**
```
STAGE     Fact-Checker
ENGINE    EXACT-MATCH + NLI
TOOLS     10
LLM CALLS 4     ← still 4, no jump
TOKENS    500   ← no jump either
```

---

# SCENE 7 — Judge (1:16 – 1:28)

**Setting.** A grand teal-robed figure on a high dais at the front of the courtroom. Above them, a holographic scale of justice in glowing teal.

**Camera.** Low-angle hero shot. The judge looms large.

**Action.** The remaining evidence floats up to the Judge — Advocate A's two verified spans glow steady; Advocates B and C have only their original (weakened) arguments. A bar chart materializes:

```
Comparison       ████████████████████  70%
Technical basis  ██████                20%
Background       ███                   10%
```

**Judge** (deliberate, low, resonant):
> "Weighing the surviving evidence."

(beat — the bars hold)

> "Verdict: **Comparison**. Confidence: seventy percent. Margin: fifty points."

A golden halo blossoms around Advocate A's podium. The other two podiums fade further into shadow.

A single, clear **gavel tap** echoes through the courtroom.

**HUD update:**
```
STAGE     Judge
ENGINE    LLM + Calibration
TOOLS     12
LLM CALLS 5
TOKENS    600
VERDICT   Comparison · 70.0%
```

---

# SCENE 8 — Clerk & the Sealed Artifact (1:28 – 1:38)

**Setting.** A small gold-robed figure at a side desk — CLERK. A scroll-like parchment unfurls in midair, lit warmly from below.

**Camera.** Slow zoom into the scroll as JSON-formatted text types itself out, character by character:

```json
{
  "case_id": "ID_77541",
  "label": "Comparison",
  "confidence": 0.70,
  "abstain": false,
  "evidence": [
    {
      "span": "faster than the non-parametric rate ...",
      "verified": true,
      "method": "exact"
    },
    {
      "span": "in contrast to their approach ...",
      "verified": true,
      "method": "exact"
    }
  ],
  "rejected_alternatives": [
    {
      "label": "Technical basis",
      "reason": "no adoption verb in paragraph"
    },
    {
      "label": "Background",
      "reason": "single weak marker; insufficient signal"
    }
  ],
  "fabricated_claims_dropped": 1,
  "tool_trace": ["tfidf_lr", "regex", "section", "co_cite",
                 "advocate×3", "examiner", "verify×3", "judge"]
}
```

A golden wax seal ⚖ stamps the bottom of the scroll. Soft hiss of sealing wax.

**Clerk** (gentle, archival, almost reverent):
> "Case sealed. Annotation artifact filed. Every claim verified. Every alternative explained."

---

# SCENE 9 — Outro: Schema Card Reveal (1:38 – 1:45)

**Setting.** Camera pulls back from the scroll. The courtroom is now empty — only the scroll remains on the central table. The lights dim.

**Camera.** Slow zoom-out, ending in a wide shot.

**Action.** A new card descends from above the courtroom: **SCHEMA CARD — citation_intent ✓**. As it locks into place, four smaller cards fade in below it:

```
  □  news_stance
  □  overdose_ed_signal   ← STAI-X 2026
  □  legal_clause_type
  □  sentiment
```

The four smaller cards pulse softly, indicating they're available.

**Narrator (V.O.):**
> "Same courtroom. Different cases. Different domains. Swap the schema — the agent never changes."

Title card fades in over the courtroom:

```
              ⚖
            JUROR
A schema-pluggable, evidence-grounded
       annotation agent

  STAI-X 2026 · Award C · Statistical Agents
```

(Beat — held for 2 seconds.)

Fade to black.

---

## Shot list (production reference)

| # | Scene | Duration | Camera | Key visual |
|---|-------|----------|--------|-----------|
| 1 | Cold open | 0:00–0:08 | Top-down descent | Case file lands in empty courtroom |
| 2 | Triage | 0:08–0:16 | Whip-pan + close-up | Bar chart of label probabilities |
| 3 | Investigator | 0:16–0:28 | 270° orbit | Three tool panels light up |
| 4 | Panel | 0:28–0:50 | Wide → dolly per speaker | Three advocates argue, last is sus |
| 5 | Cross-Examiner | 0:50–1:00 | Tracking pace | Red slashes cut down losers |
| 6 | Fact-Checker | 1:00–1:16 | Push-in + tight on stamps | Heavy ✗ UNVERIFIED stamp |
| 7 | Judge | 1:16–1:28 | Low-angle hero | Bar chart of verdict probabilities |
| 8 | Clerk | 1:28–1:38 | Slow zoom into scroll | JSON types itself out |
| 9 | Outro | 1:38–1:45 | Slow zoom out | Schema cards reveal extensibility |

## Three lines that must land

These are the rhetorical hooks; make sure the AI video model emphasizes them with pacing and music:

1. **"Twenty-eight seconds in — zero LLM calls so far."** (end of Scene 3)
2. **"Even the winning side is fact-checked. The Fact-Checker uses no LLM — LLMs cannot vouch for themselves."** (Scene 6 sub-banner)
3. **"Same courtroom. Different cases. Different domains."** (Scene 9 V.O.)

## Style references for the AI model

> Visual mood-board cues: *Tron: Legacy* (volumetric light, neon edges), *Ex Machina* (cold minimal interiors), *Black Mirror: USS Callister* opening titles (floating UI), *Apple WWDC keynote intros* (smooth motion graphics). Avoid: actual human faces, realistic wood-paneled courtroom, gavel close-ups that look like stock footage.

## Generation strategy

If your AI video tool has per-clip duration limits (e.g., Sora ~10s, Runway ~10s, Veo ~10s):

- **Render each scene as a separate clip.** Most scenes are already ≤10 seconds.
- **Scene 4 (panel)** is 22 seconds — split into three 7-second sub-clips, one per advocate, and stitch in post.
- **Scene 6 (fact-checker)** is 16 seconds — split into two 8-second clips: (a) Fact-Checker entering + first two ✓ stamps, (b) the ✗ stamp + the sub-banner.

Stitch in any standard video editor (DaVinci Resolve, Premiere, CapCut). Add the HUD overlay and subtitles as a separate top layer, since the AI model will struggle to keep the HUD text consistent across clips.

## Voice cast

| Voice | Suggested texture |
|-------|-------------------|
| Narrator | Calm male/female alto, mid-tempo, neutral. Think *Apple keynote voiceover*. |
| Triage | Synthetic, slightly robotic, clipped. |
| Investigator | Curious, mid-male, faintly detective-toned. |
| Advocate A | Confident, smooth, slightly smug. |
| Advocate B | Hesitant, breathy, light-female. |
| Advocate C | Dismissive, flat. |
| Cross-Examiner | Cold, surgical, mid-tempo, slightly sibilant. |
| Fact-Checker | Synthetic, slow, deliberately *uncanny* — no warmth. |
| Judge | Deep, resonant, slow. The most "human" voice. |
| Clerk | Warm, gentle, archival. |

Eleven Labs or OpenAI TTS will handle all of these with stock voices; consider tagging Fact-Checker's lines with a slight pitch-shift to keep the "non-human verification" theme audible.

---

## End of script
