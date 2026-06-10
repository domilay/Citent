#!/usr/bin/env python3
"""
JUROR prototype: a schema-pluggable, evidence-grounded citation-intent agent.

Pure Python.  No API keys, no LLM downloads, no heavy dependencies (stdlib only).
The "LLM" stages are replaced by deterministic heuristics so the prototype
produces a reproducible JSON trace that the courtroom demo plays back.
Real LLM calls can be swapped in by replacing the *_llm_* functions.

Usage:
    python juror.py                          # runs the bundled case, writes trace
    python juror.py --case 2                 # run a different bundled case
    python juror.py --out path/to/trace.json # custom output path
"""

import os
import re
import json
import math
import time
import argparse
from collections import Counter


# ===========================================================================
# SCHEMA CARD  (the only thing that needs to change to retarget the agent)
# ===========================================================================

SCHEMA = {
    "task": "citation_intent",
    "labels": ["Background", "Technical basis", "Comparison", "Fundamental idea"],
    "abstain_threshold": 0.55,
    "label_priors": {
        "Background": {
            "markers": ["see e.g.", "for a review", "has been used", "among others",
                        "as discussed", "in general", "previously", "literature"],
            "negative_verbs": ["we follow", "we adopt", "we use", "compared to",
                               "outperform"],
        },
        "Technical basis": {
            "markers": ["we follow", "we adopt", "we use", "we extend",
                        "based on", "building on", "we apply", "we implement"],
            "negative_verbs": [],
        },
        "Comparison": {
            "markers": ["compared to", "in contrast", "outperform", "worse than",
                        "vs.", "performs better"],
            "negative_verbs": [],
        },
        "Fundamental idea": {
            "markers": ["based on the idea", "inspired by", "the key insight",
                        "motivated by", "underlying principle"],
            "negative_verbs": [],
        },
    },
}


# ===========================================================================
# SAMPLE CASES
# ===========================================================================

SAMPLE_CASES = {
    1: {
        "case_id": "ID_183878",
        "paragraph": (
            "In general, simultaneous confidence bands for a function f are constructed "
            "by studying the asymptotic distribution of the sup. The approach of "
            "Bickel and Rosenblatt (1973) relates this to a study of the distribution "
            "of a Gaussian process. This approach to constructing confidence bands has "
            "been used in the context of nonparametric estimation by, among others, "
            "Hardle (1989) for M-estimators and Claeskens and Van Keilegom (2003) for "
            "local polynomial likelihood estimators."
        ),
        "ref_clean_citation": "Hardle (1989)",
        "ground_truth": "Background",
    },
    2: {
        "case_id": "ID_99812",
        "paragraph": (
            "We follow the boosting framework of Buehlmann and Hothorn (2007), in which "
            "regularization is achieved indirectly via the application of penalized "
            "base learners. We extend their componentwise gradient boosting to handle "
            "conditional transformation models, but the iterative scheme and the "
            "stopping criterion remain as in the original work."
        ),
        "ref_clean_citation": "Buehlmann and Hothorn (2007)",
        "ground_truth": "Technical basis",
    },
    3: {
        "case_id": "ID_77541",
        "paragraph": (
            "Our proposed kernel estimator achieves a parametric rate of convergence "
            "on subsets, which is faster than the non-parametric rate reported by "
            "Yao et al. (2005) under the sparsely observed regime. In contrast to "
            "their approach, our method does not require functional principal "
            "component analysis as a preprocessing step."
        ),
        "ref_clean_citation": "Yao et al. (2005)",
        "ground_truth": "Comparison",
    },
}


# ===========================================================================
# NON-LLM TOOLS
# ===========================================================================

def tfidf_lr_classifier(text):
    """Stand-in for the TF-IDF + LogisticRegression baseline trained on 340 K
    instances (see deliver/ml_based/).  Returns label probabilities."""
    lower = text.lower()
    scores = {}
    for label, prior in SCHEMA["label_priors"].items():
        s = 0.5  # base
        for m in prior["markers"]:
            if m in lower:
                s += 1.0
        for v in prior["negative_verbs"]:
            if v in lower:
                s -= 0.7
        scores[label] = max(s, 0.05)

    total = sum(scores.values())
    probs = {l: round(s / total, 3) for l, s in scores.items()}
    top = max(probs, key=probs.get)
    return {
        "predicted_label": top,
        "probs": probs,
        "max_prob": probs[top],
        "difficulty": round(1 - probs[top], 2),
    }


def rhetorical_marker_regex(text):
    """Find all rhetorical markers from the schema's prior list."""
    lower = text.lower()
    found = []
    for label, prior in SCHEMA["label_priors"].items():
        for m in prior["markers"]:
            for match in re.finditer(re.escape(m), lower):
                found.append({"marker": m, "label": label, "pos": match.start()})
    return found


def section_classifier(text):
    """Heuristic section classifier."""
    t = text.lower()
    if any(k in t for k in ["we propose", "our method", "we follow", "we extend",
                            "we apply", "we implement"]):
        return {"section": "Methods", "confidence": 0.78}
    if any(k in t for k in ["in general", "for a review", "literature",
                            "previously", "has been used"]):
        return {"section": "Related Work", "confidence": 0.74}
    if any(k in t for k in ["outperform", "results show", "achieves a",
                            "compared to", "in contrast"]):
        return {"section": "Experiments / Comparison", "confidence": 0.68}
    return {"section": "Unknown", "confidence": 0.30}


def co_citation_lookup(ref, paragraph):
    """Mock: find other references in the same paragraph (proxy for graph lookup)."""
    refs = re.findall(r"[A-Z][A-Za-z]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][A-Za-z]+)?\s*\(\d{4}\)",
                      paragraph)
    return [r for r in refs if r.strip() != ref.strip()][:5]


def exact_span_match(span, source):
    span_norm = re.sub(r"\s+", " ", span).strip().lower()
    src_norm = re.sub(r"\s+", " ", source).strip().lower()
    return span_norm in src_norm


def bleu_overlap(span, source):
    s = set(span.lower().split())
    t = set(source.lower().split())
    if not s:
        return 0.0
    return len(s & t) / len(s)


# ===========================================================================
# STAGES
# ===========================================================================

def stage_triage(case):
    t0 = time.perf_counter()
    out = tfidf_lr_classifier(case["paragraph"])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # One sub-agent per intent: every label in the schema gets an advocate.
    # We rank them by the triage classifier's probability (for display order)
    # but never drop any -- with only four intents, the full panel always sits.
    sorted_probs = sorted(out["probs"].items(), key=lambda x: -x[1])
    shortlist = [l for l, _ in sorted_probs]

    fast_track = out["difficulty"] < 0.30

    actions = [
        {"t": 0, "type": "say", "actor": "triage",
         "text": f"New case received: {case['case_id']}."},
        {"t": 250, "type": "tool_call", "tool": "tfidf_lr_classifier",
         "engine": "ML",
         "summary": f"max_prob = {out['max_prob']:.3f}"},
        {"t": 700, "type": "say", "actor": "triage",
         "text": f"Difficulty = {out['difficulty']:.2f}. "
                 f"Convening one advocate per intent: {', '.join(shortlist)}."},
        {"t": 1100, "type": "shortlist", "value": shortlist,
         "fast_track": fast_track},
    ]

    return {
        "id": "triage",
        "name": "Triage Officer",
        "engine": "ML",
        "tools_called": ["tfidf_lr_classifier"],
        "duration_ms": round(elapsed_ms, 1),
        "tokens": 0,
        "llm_calls": 0,
        "outputs": {
            "difficulty": out["difficulty"],
            "shortlist": shortlist,
            "probs": out["probs"],
            "fast_track": fast_track,
        },
        "actions": actions,
    }


def stage_investigator(case, triage_out):
    t0 = time.perf_counter()
    markers = rhetorical_marker_regex(case["paragraph"])
    section = section_classifier(case["paragraph"])
    co_cites = co_citation_lookup(case["ref_clean_citation"], case["paragraph"])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    actions = [
        {"t": 0, "type": "say", "actor": "investigator",
         "text": "Gathering evidence; cheapest tools first."},
        {"t": 350, "type": "tool_call", "tool": "rhetorical_marker_regex",
         "engine": "RULE",
         "summary": f"{len(markers)} marker(s) found"},
        {"t": 800, "type": "say", "actor": "investigator",
         "text": f"Found {len(markers)} rhetorical marker(s): "
                 f"{', '.join(sorted({m['marker'] for m in markers}))}."},
        {"t": 1250, "type": "tool_call", "tool": "section_classifier",
         "engine": "ML",
         "summary": section["section"]},
        {"t": 1600, "type": "say", "actor": "investigator",
         "text": f"Section: {section['section']} (conf {section['confidence']})."},
        {"t": 2000, "type": "tool_call", "tool": "co_citation_lookup",
         "engine": "GRAPH",
         "summary": f"{len(co_cites)} co-citation(s)"},
        {"t": 2400, "type": "say", "actor": "investigator",
         "text": f"Case file complete — {len(markers)} markers, "
                 f"section={section['section']}, {len(co_cites)} co-cites."},
    ]

    return {
        "id": "investigator",
        "name": "Investigator",
        "engine": "RULE+ML+GRAPH",
        "tools_called": ["rhetorical_marker_regex", "section_classifier",
                         "co_citation_lookup"],
        "duration_ms": round(elapsed_ms, 1),
        "tokens": 0,
        "llm_calls": 0,
        "outputs": {
            "markers": markers,
            "section": section,
            "co_citations": co_cites,
        },
        "actions": actions,
    }


def stage_panel(case, triage_out, investigator_out):
    """One advocate per shortlisted label.  Each builds an argument from
    the case file (no real LLM call; we synthesise a plausible argument
    from the detected features)."""
    t0 = time.perf_counter()
    shortlist = triage_out["outputs"]["shortlist"]
    markers = investigator_out["outputs"]["markers"]
    section = investigator_out["outputs"]["section"]["section"]

    advocates = []
    for label in shortlist:
        prior = SCHEMA["label_priors"][label]
        label_markers = [m for m in markers if m["label"] == label]

        # build evidence spans = small windows around each detected marker
        ev_spans = []
        for m in label_markers:
            start = max(0, m["pos"] - 20)
            end = min(len(case["paragraph"]),
                      m["pos"] + len(m["marker"]) + 30)
            ev_spans.append({"span": case["paragraph"][start:end],
                             "marker": m["marker"]})

        # For the Comparison advocate, fabricate one extra evidence span
        # that does NOT appear in the paragraph -- exposes the Fact-Checker.
        if label == "Comparison" and not any(
                "outperform" in s["span"].lower() for s in ev_spans):
            ev_spans.append({
                "span": "the authors directly outperform the cited method on benchmarks",
                "marker": "(fabricated)",
            })

        neg_hits = [v for v in prior["negative_verbs"] if v in case["paragraph"].lower()]

        # raw strength
        strength = 1.0 * len(label_markers) - 0.8 * len(neg_hits)
        if section == "Related Work" and label == "Background":
            strength += 0.6
        if section == "Methods" and label == "Technical basis":
            strength += 0.6
        if section.startswith("Experiments") and label == "Comparison":
            strength += 0.4

        argument = _build_argument_text(label, label_markers, neg_hits,
                                        section, case["ref_clean_citation"])

        advocates.append({
            "label": label,
            "argument": argument,
            "evidence_spans": ev_spans,
            "raw_strength": round(max(strength, 0.0), 2),
        })

    elapsed_ms = (time.perf_counter() - t0) * 1000

    actions = []
    t = 0
    for adv in advocates:
        actions.append({"t": t, "type": "advocate_speak",
                        "label": adv["label"], "text": adv["argument"],
                        "strength": adv["raw_strength"]})
        actions.append({"t": t + 100, "type": "tool_call",
                        "tool": f"llm_advocate[{adv['label']}]", "engine": "LLM",
                        "summary": f"strength={adv['raw_strength']}"})
        t += 1800

    tokens_per_advocate = 140
    return {
        "id": "panel",
        "name": "Prosecution Panel",
        "engine": "LLM",
        "tools_called": [f"llm_advocate[{a['label']}]" for a in advocates],
        "duration_ms": round(elapsed_ms, 1),
        "tokens": tokens_per_advocate * len(advocates),
        "llm_calls": len(advocates),
        "outputs": {"advocates": advocates},
        "actions": actions,
    }


def _build_argument_text(label, markers, negs, section, ref):
    """Cheap synthesised argument text for visualisation purposes."""
    if not markers and not negs:
        return f"I argue {label}, though direct surface evidence is thin."

    bits = []
    if markers:
        unique = sorted({m["marker"] for m in markers})
        bits.append(f"{len(markers)} marker(s) found ({', '.join(unique)})")
    if section:
        bits.append(f"paragraph is from {section}")
    if negs:
        bits.append(f"WARNING: counter-markers {negs}")
    return f"I argue {label}: " + "; ".join(bits) + "."


def stage_examiner(panel_out, investigator_out):
    """Attack each advocate using rule-based contradiction patterns."""
    t0 = time.perf_counter()
    advocates = panel_out["outputs"]["advocates"]
    section = investigator_out["outputs"]["section"]["section"]

    attacks = []
    for adv in advocates:
        attack = None
        flag = None

        if adv["label"] == "Technical basis" and section in ("Related Work",):
            attack = ("Counter: paragraph is from Related Work, not Methods — "
                      "the 'technical basis' claim is implausible here.")
            flag = "contradicted"
        elif adv["label"] == "Comparison":
            # check that the markers are real comparative markers
            has_real_comp = any(m["marker"] in ("compared to", "in contrast",
                                                "outperform", "worse than", "vs.")
                                for m in [{"marker": x["marker"]}
                                          for x in adv["evidence_spans"]])
            if not has_real_comp and adv["raw_strength"] < 1.0:
                attack = ("Counter: no real comparative verb appears in the "
                          "paragraph; this argument cannot stand on the textual evidence.")
                flag = "unsupported"
        elif adv["label"] == "Background" and adv["raw_strength"] < 0.3:
            attack = "Counter: only a single weak marker; not enough signal."
            flag = "weak"

        if attack:
            attacks.append({"label": adv["label"], "attack": attack, "flag": flag})

    elapsed_ms = (time.perf_counter() - t0) * 1000

    actions = []
    if attacks:
        t = 0
        for atk in attacks:
            actions.append({"t": t, "type": "attack",
                            "label": atk["label"], "text": atk["attack"],
                            "flag": atk["flag"]})
            t += 1400
    else:
        actions.append({"t": 0, "type": "say", "actor": "examiner",
                        "text": "No advocate argument fails the contradiction filter."})

    return {
        "id": "examiner",
        "name": "Cross-Examiner",
        "engine": "RULE+LLM",
        "tools_called": ["contradiction_pattern"] + (
            ["llm_critic"] if attacks else []),
        "duration_ms": round(elapsed_ms, 1),
        "tokens": 80 * len(attacks),
        "llm_calls": len(attacks),
        "outputs": {"attacks": attacks},
        "actions": actions,
    }


def stage_fact_checker(panel_out, case):
    t0 = time.perf_counter()
    advocates = panel_out["outputs"]["advocates"]

    verifications = []
    for adv in advocates:
        for ev in adv["evidence_spans"]:
            exact = exact_span_match(ev["span"], case["paragraph"])
            bleu = bleu_overlap(ev["span"], case["paragraph"])
            verified = exact or bleu >= 0.85
            verifications.append({
                "label": adv["label"],
                "span": ev["span"],
                "exact_match": exact,
                "bleu": round(bleu, 2),
                "verified": verified,
            })

    elapsed_ms = (time.perf_counter() - t0) * 1000

    actions = []
    t = 0
    for v in verifications:
        short = v["span"][:80] + ("..." if len(v["span"]) > 80 else "")
        actions.append({"t": t, "type": "verify", "label": v["label"],
                        "span": short, "verified": v["verified"],
                        "exact": v["exact_match"], "bleu": v["bleu"]})
        t += 420

    return {
        "id": "fact_checker",
        "name": "Fact-Checker",
        "engine": "RULE+NLI",
        "tools_called": ["exact_span_match", "bleu_overlap"],
        "duration_ms": round(elapsed_ms, 1),
        "tokens": 0,
        "llm_calls": 0,
        "outputs": {"verifications": verifications},
        "actions": actions,
    }


def stage_judge(panel_out, examiner_out, factchecker_out, triage_out):
    t0 = time.perf_counter()
    advocates = panel_out["outputs"]["advocates"]
    attacks_by_label = {a["label"]: a for a in examiner_out["outputs"]["attacks"]}

    verified_count = {}
    for v in factchecker_out["outputs"]["verifications"]:
        if v["verified"]:
            verified_count[v["label"]] = verified_count.get(v["label"], 0) + 1

    scores = {}
    for adv in advocates:
        s = adv["raw_strength"] + 0.5 * verified_count.get(adv["label"], 0)
        if adv["label"] in attacks_by_label:
            flag = attacks_by_label[adv["label"]]["flag"]
            s -= {"contradicted": 1.5, "unsupported": 1.2, "weak": 0.7}.get(flag, 0.5)
        s += triage_out["outputs"]["probs"].get(adv["label"], 0)
        scores[adv["label"]] = round(max(s, 0.02), 3)

    # softmax for calibrated confidence
    exp_scores = {l: math.exp(s) for l, s in scores.items()}
    Z = sum(exp_scores.values())
    probs = {l: round(s / Z, 3) for l, s in exp_scores.items()}
    winner = max(probs, key=probs.get)
    sorted_p = sorted(probs.values(), reverse=True)
    margin = round(sorted_p[0] - sorted_p[1], 3) if len(sorted_p) > 1 else 1.0
    confidence = probs[winner]
    abstain = confidence < SCHEMA["abstain_threshold"]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    actions = [
        {"t": 0, "type": "say", "actor": "judge",
         "text": "Weighing arguments and verified evidence."},
        {"t": 700, "type": "scores", "scores": scores, "probs": probs},
        {"t": 1400, "type": "verdict", "label": winner,
         "confidence": confidence, "margin": margin, "abstain": abstain},
        {"t": 1700, "type": "say", "actor": "judge",
         "text": (f"Verdict: {winner}  ·  conf {confidence:.2f}  ·  margin {margin:.2f}"
                  + ("  ·  ABSTAINED (below threshold)" if abstain else ""))},
    ]

    return {
        "id": "judge",
        "name": "Judge",
        "engine": "LLM+CAL",
        "tools_called": ["llm_arbitrator", "platt_calibrator"],
        "duration_ms": round(elapsed_ms, 1),
        "tokens": 100,
        "llm_calls": 1,
        "outputs": {
            "label": winner, "confidence": confidence, "margin": margin,
            "scores": scores, "probs": probs, "abstain": abstain,
        },
        "actions": actions,
    }


def stage_clerk(case, triage, inv, panel, exam, fact, judge):
    t0 = time.perf_counter()
    winner = judge["outputs"]["label"]
    verified = [v for v in fact["outputs"]["verifications"]
                if v["label"] == winner and v["verified"]]

    attacks_by_label = {a["label"]: a for a in exam["outputs"]["attacks"]}
    rejected_alts = []
    for adv in panel["outputs"]["advocates"]:
        if adv["label"] == winner:
            continue
        atk = attacks_by_label.get(adv["label"])
        rejected_alts.append({
            "label": adv["label"],
            "reason": atk["attack"] if atk
            else f"lower posterior than winner ({judge['outputs']['probs'].get(adv['label'], 0):.2f})",
        })

    artifact = {
        "case_id": case["case_id"],
        "label": winner,
        "confidence": judge["outputs"]["confidence"],
        "margin": judge["outputs"]["margin"],
        "abstain": judge["outputs"]["abstain"],
        "evidence": [
            {"span": v["span"], "verified": True,
             "method": "exact" if v["exact_match"] else "bleu"}
            for v in verified
        ],
        "rejected_alternatives": rejected_alts,
        "ground_truth": case.get("ground_truth"),
    }

    elapsed_ms = (time.perf_counter() - t0) * 1000

    actions = [
        {"t": 0, "type": "say", "actor": "clerk", "text": "Sealing the case file."},
        {"t": 500, "type": "show_artifact", "artifact": artifact},
        {"t": 1300, "type": "say", "actor": "clerk", "text": "Annotation artifact filed."},
    ]

    return {
        "id": "clerk", "name": "Clerk", "engine": "CODE",
        "tools_called": [],
        "duration_ms": round(elapsed_ms, 1),
        "tokens": 0,
        "llm_calls": 0,
        "outputs": {"artifact": artifact},
        "actions": actions,
    }


# ===========================================================================
# ORCHESTRATOR
# ===========================================================================

def run_case(case):
    trace = {
        "case_id": case["case_id"],
        "input": {
            "paragraph": case["paragraph"],
            "cited_ref": case["ref_clean_citation"],
            "ground_truth": case.get("ground_truth"),
        },
        "schema": {"task": SCHEMA["task"], "labels": SCHEMA["labels"]},
        "stages": [],
    }

    triage = stage_triage(case);                              trace["stages"].append(triage)
    inv    = stage_investigator(case, triage);                trace["stages"].append(inv)
    panel  = stage_panel(case, triage, inv);                  trace["stages"].append(panel)
    exam   = stage_examiner(panel, inv);                      trace["stages"].append(exam)
    fact   = stage_fact_checker(panel, case);                 trace["stages"].append(fact)
    judge  = stage_judge(panel, exam, fact, triage);          trace["stages"].append(judge)
    clerk  = stage_clerk(case, triage, inv, panel, exam, fact, judge); trace["stages"].append(clerk)

    trace["final_artifact"] = clerk["outputs"]["artifact"]
    trace["summary"] = {
        "total_duration_ms": round(sum(s["duration_ms"] for s in trace["stages"]), 1),
        "total_tokens": sum(s["tokens"] for s in trace["stages"]),
        "tools_invoked": sum(len(s["tools_called"]) for s in trace["stages"]),
        "llm_calls": sum(s["llm_calls"] for s in trace["stages"]),
        "ground_truth_match": (
            clerk["outputs"]["artifact"]["label"] == case.get("ground_truth")
        ),
    }
    return trace


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=1, choices=[1, 2, 3],
                        help="Which bundled sample case to run (1/2/3)")
    parser.add_argument("--out", default=None,
                        help="Output trace JSON path (default = ../demo/traces/case_<N>.json)")
    args = parser.parse_args()

    case = SAMPLE_CASES[args.case]
    trace = run_case(case)

    out_path = args.out or os.path.join(
        os.path.dirname(__file__), "..", "demo", "traces",
        f"case_{args.case:03d}.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    art = trace["final_artifact"]
    print(f"\nCase {case['case_id']}")
    print(f"  predicted:    {art['label']}  (conf {art['confidence']:.2f})")
    print(f"  ground truth: {case.get('ground_truth', 'unknown')}")
    print(f"  correct:      {trace['summary']['ground_truth_match']}")
    print(f"  total tokens: {trace['summary']['total_tokens']}")
    print(f"  llm calls:    {trace['summary']['llm_calls']}")
    print(f"  trace -> {out_path}")


if __name__ == "__main__":
    main()
