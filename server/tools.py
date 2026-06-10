"""
JUROR · real (non-LLM) tools for the backend.

These are the SAME tools the architecture calls out, but implemented for real:

  - TfidfLrTriage         : loads our trained TF-IDF + LR (69.3 % held-out acc.)
  - rhetorical_markers    : regex over linguistic priors per intent
  - section_classifier    : keyword heuristic for which section a paragraph
                            likely came from
  - co_citation_lookup    : finds other refs cited near the target ref
  - exact_span_match      : literal substring check
  - bleu_overlap          : token n-gram set overlap (cheap)
  - NliVerifier           : DeBERTa-v3 MNLI entailment via HuggingFace
"""

from __future__ import annotations

import os
import re
import pickle
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("juror.tools")

HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")


# ───────────────────────────────────────────────────────────────────
#  Triage  ·  TF-IDF + LogisticRegression  (real artefacts)
# ───────────────────────────────────────────────────────────────────
class TfidfLrTriage:
    """Loads the trained TF-IDF + LR + LabelEncoder from server/models/.

    All three artefacts come out of ``train_models.py`` so they reflect the
    real label distribution (53 % Background, 33 % TB, 10 % CMP, 4 % FI) of the
    340 K-instance corpus.
    """

    def __init__(self, models_dir: str = MODELS_DIR) -> None:
        with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(os.path.join(models_dir, "lr_classifier.pkl"), "rb") as f:
            self.clf = pickle.load(f)
        with open(os.path.join(models_dir, "label_encoder.pkl"), "rb") as f:
            self.le = pickle.load(f)
        self.label_names: List[str] = list(self.le.classes_)
        log.info("triage loaded: %d-class LR, vocab=%d",
                 len(self.label_names), len(self.vectorizer.vocabulary_))

    def predict(self, paragraph: str) -> Dict:
        """Return calibrated probabilities + shortlist + difficulty score."""
        X = self.vectorizer.transform([paragraph])
        proba = self.clf.predict_proba(X)[0]              # (4,)
        # Map from encoder order back to label names
        probs = {self.label_names[i]: float(round(proba[i], 4))
                 for i in range(len(self.label_names))}
        ordered = sorted(probs.items(), key=lambda kv: -kv[1])
        top_label, top_p = ordered[0]
        # Difficulty = 1 − margin of top over second
        difficulty = round(1.0 - (top_p - ordered[1][1]), 3)
        shortlist = [lbl for lbl, p in ordered if p > 0.05][:3]
        if len(shortlist) < 2:
            shortlist = [lbl for lbl, _ in ordered[:2]]
        return {
            "predicted_label": top_label,
            "probs":           probs,
            "max_prob":        round(top_p, 4),
            "difficulty":      difficulty,
            "shortlist":       shortlist,
            "fast_track":      bool(difficulty < 0.30),
        }


# ───────────────────────────────────────────────────────────────────
#  Linguistic priors  (used by Investigator + Cross-Examiner)
# ───────────────────────────────────────────────────────────────────
LABEL_PRIORS: Dict[str, Dict[str, List[str]]] = {
    "Background": {
        "markers": [
            "see e.g.", "for a review", "has been used", "among others",
            "as discussed", "in general", "previously", "literature",
        ],
        "negative_verbs": [
            "we follow", "we adopt", "we use", "compared to", "outperform",
        ],
    },
    "Technical basis": {
        "markers": [
            "we follow", "we adopt", "we use", "we extend",
            "based on", "building on", "we apply", "we implement",
        ],
        "negative_verbs": [],
    },
    "Comparison": {
        "markers": [
            "compared to", "in contrast", "outperform", "worse than",
            "vs.", "performs better",
        ],
        "negative_verbs": [],
    },
    "Fundamental idea": {
        "markers": [
            "based on the idea", "inspired by", "the key insight",
            "motivated by", "underlying principle",
        ],
        "negative_verbs": [],
    },
}

ALL_LABELS = list(LABEL_PRIORS.keys())


def rhetorical_markers(text: str) -> List[Dict]:
    """All marker hits across all label priors.  Real."""
    lower = text.lower()
    found = []
    for label, prior in LABEL_PRIORS.items():
        for m in prior["markers"]:
            for hit in re.finditer(re.escape(m), lower):
                found.append({"marker": m, "label": label, "pos": hit.start()})
    return found


def section_classifier(text: str) -> Dict:
    """Lightweight heuristic — predicts which section a paragraph is from.

    Real (not gated on anything heavier — section gold isn't available in our
    dataset, so a fast keyword rule is the best honest choice).
    """
    t = text.lower()
    if any(k in t for k in ["we propose", "our method", "we follow",
                            "we extend", "we apply", "we implement"]):
        return {"section": "Methods", "confidence": 0.78}
    if any(k in t for k in ["in general", "for a review", "literature",
                            "previously", "has been used"]):
        return {"section": "Related Work", "confidence": 0.74}
    if any(k in t for k in ["outperform", "results show", "achieves a",
                            "compared to", "in contrast"]):
        return {"section": "Experiments / Comparison", "confidence": 0.68}
    return {"section": "Unknown", "confidence": 0.30}


_REF_PATTERN = re.compile(
    r"[A-Z][A-Za-z]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][A-Za-z]+)?\s*\(\d{4}\)"
)


def co_citation_lookup(target_ref: str, paragraph: str) -> List[str]:
    """Other references in the same paragraph, excluding the target."""
    refs = _REF_PATTERN.findall(paragraph)
    seen, out = set(), []
    for r in refs:
        r = r.strip()
        if r == target_ref.strip() or r in seen:
            continue
        seen.add(r)
        out.append(r)
        if len(out) >= 5:
            break
    return out


# ───────────────────────────────────────────────────────────────────
#  Fact-Checker primitives
# ───────────────────────────────────────────────────────────────────
def exact_span_match(span: str, source: str) -> bool:
    """Substring presence, whitespace-normalised."""
    a = re.sub(r"\s+", " ", span).strip().lower()
    b = re.sub(r"\s+", " ", source).strip().lower()
    return a in b


def bleu_overlap(span: str, source: str) -> float:
    s = set(span.lower().split())
    t = set(source.lower().split())
    if not s:
        return 0.0
    return round(len(s & t) / len(s), 3)


class NliVerifier:
    """Real entailment check using a small HF NLI model.

    Default model: cross-encoder/nli-deberta-v3-small (~140 MB).
    Downloaded lazily on first use; cached under HF_HOME.

    If transformers / torch are not installed (e.g. user only wants the
    backend without ML deps), `available` is False and the verifier
    transparently falls back to BLEU overlap, surfaced as a separate signal.
    """

    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or os.environ.get(
            "JUROR_NLI_MODEL", self.DEFAULT_MODEL)
        self.available = False
        self._pipe = None
        try:
            from transformers import pipeline       # noqa: F401
            self.available = True
        except Exception as e:                       # pragma: no cover
            log.warning("transformers unavailable (%s); NLI falls back to BLEU.", e)

    def _ensure_loaded(self) -> None:
        if self._pipe is None and self.available:
            from transformers import pipeline
            log.info("loading NLI model %s ...", self.model_name)
            self._pipe = pipeline(
                "text-classification", model=self.model_name,
                top_k=None,
            )

    def verify(self, claim: str, source: str) -> Dict:
        """Return {verified, exact_match, bleu, entailment_score?}."""
        result = {
            "exact_match": exact_span_match(claim, source),
            "bleu":        bleu_overlap(claim, source),
        }
        if result["exact_match"]:
            result["verified"] = True
            result["method"] = "exact"
            return result
        if not self.available:
            verified = result["bleu"] >= 0.85
            result["verified"] = bool(verified)
            result["method"]   = "bleu"
            return result

        # Real NLI path
        try:
            self._ensure_loaded()
            # Cross-encoder NLI: (premise, hypothesis) → ENTAILMENT/NEUTRAL/CONTRADICTION
            scores = self._pipe({"text": source, "text_pair": claim})
            ent = next((s for s in scores
                        if s["label"].upper().startswith("ENT")), None)
            entail_score = float(ent["score"]) if ent else 0.0
            result["entailment_score"] = round(entail_score, 4)
            result["verified"] = bool(entail_score >= 0.6 or result["bleu"] >= 0.85)
            result["method"]   = "nli" if entail_score >= 0.6 else "bleu"
        except Exception as e:                       # pragma: no cover
            log.warning("NLI verify failed (%s); falling back to BLEU.", e)
            result["verified"] = bool(result["bleu"] >= 0.85)
            result["method"]   = "bleu"
        return result


# ───────────────────────────────────────────────────────────────────
#  Module-level singletons (loaded once per worker)
# ───────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_triage() -> TfidfLrTriage:
    return TfidfLrTriage()


@lru_cache(maxsize=1)
def get_nli() -> NliVerifier:
    return NliVerifier()
