#!/usr/bin/env python3
"""
Train a compact TF-IDF + LogisticRegression classifier for JUROR's Triage stage.

This produces two small pickles (~5 MB total) that ship inside `server/models/`
so the deployed agent can run real ML — not heuristics — on every case.

Inputs:
    Citations.csv                          (paragraph + ref)
    第一轮major标注结果-stage1.csv         (gold labels)

Outputs (under server/models/):
    tfidf_vectorizer.pkl     compact (max_features=10000) fitted TF-IDF
    lr_classifier.pkl        balanced LR trained on the 80% split
    label_encoder.pkl        sklearn LabelEncoder so we can map back to label names
    metrics.json             held-out accuracy / per-class F1 (for the badge in the UI)

Usage:
    python train_models.py \
        --citations  ../../Citations.csv \
        --labels     ../../第一轮major标注结果-stage1.csv
"""

import os
import json
import time
import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")


def log(msg):
    print(f"[train] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--citations", required=True,
                        help="Path to Citations.csv")
    parser.add_argument("--labels", required=True,
                        help="Path to the gold-label CSV (Major_Result column)")
    parser.add_argument("--max-features", type=int, default=10000,
                        help="TF-IDF vocab cap (default 10000, keeps pickle ~5MB)")
    parser.add_argument("--out-dir", default=MODELS_DIR,
                        help="Where to write the pickles")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    log(f"loading {args.citations} ...")
    citations = pd.read_csv(args.citations, usecols=["ID", "Para_text"])
    log(f"  {len(citations):,} citation rows")

    log(f"loading {args.labels} ...")
    labels = pd.read_csv(args.labels, usecols=["ID", "Major_Result"])
    log(f"  {len(labels):,} label rows")

    log("merging ...")
    df = pd.merge(citations, labels, on="ID", how="inner")
    log(f"  {len(df):,} rows after inner join")
    log(f"  label distribution:")
    for lab, cnt in df["Major_Result"].value_counts().items():
        log(f"    {lab:18s}  {cnt:>7,}  ({100*cnt/len(df):.1f}%)")

    X_text = df["Para_text"].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(df["Major_Result"].values)
    label_names = list(le.classes_)
    log(f"  label encoding: {dict(zip(label_names, range(len(label_names))))}")

    log("splitting 80/20 ...")
    X_tr_text, X_te_text, y_tr, y_te = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    log(f"  train={len(X_tr_text):,}  test={len(X_te_text):,}")

    log(f"fitting TF-IDF (max_features={args.max_features}, ngram=1-2) ...")
    t0 = time.time()
    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        dtype=np.float32,
    )
    X_tr = tfidf.fit_transform(X_tr_text)
    X_te = tfidf.transform(X_te_text)
    log(f"  TF-IDF train shape {X_tr.shape}  ({time.time()-t0:.1f}s)")

    log("training LogisticRegression (class_weight=balanced) ...")
    t0 = time.time()
    n_workers = os.cpu_count() or 4
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs",
        max_iter=300, n_jobs=n_workers, random_state=42,
    )
    clf.fit(X_tr, y_tr)
    log(f"  trained in {time.time()-t0:.1f}s")

    log("evaluating on held-out split ...")
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro")
    f1w = f1_score(y_te, y_pred, average="weighted")
    log(f"  acc={acc:.4f}  macro-F1={f1m:.4f}  weighted-F1={f1w:.4f}")
    log("  per-class report:")
    print(classification_report(y_te, y_pred, target_names=label_names, digits=4))

    pc, rc, fc, sc = precision_recall_fscore_support(
        y_te, y_pred, average=None, zero_division=0)
    per_class = [
        {"label": label_names[i], "precision": round(pc[i], 4),
         "recall": round(rc[i], 4), "f1": round(fc[i], 4), "support": int(sc[i])}
        for i in range(len(label_names))
    ]

    log("saving artifacts ...")
    with open(os.path.join(args.out_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f, protocol=4)
    with open(os.path.join(args.out_dir, "lr_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f, protocol=4)
    with open(os.path.join(args.out_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f, protocol=4)

    metrics = {
        "trained_on":  os.path.basename(args.citations),
        "n_train":     int(len(X_tr_text)),
        "n_test":      int(len(X_te_text)),
        "max_features": int(args.max_features),
        "accuracy":    round(acc, 4),
        "macro_f1":    round(f1m, 4),
        "weighted_f1": round(f1w, 4),
        "per_class":   per_class,
        "label_order": label_names,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # report sizes
    log("artifact sizes:")
    for name in ("tfidf_vectorizer.pkl", "lr_classifier.pkl", "label_encoder.pkl",
                 "metrics.json"):
        path = os.path.join(args.out_dir, name)
        if os.path.exists(path):
            log(f"  {name:24s}  {os.path.getsize(path)/1024:.1f} KB")
    log("done")


if __name__ == "__main__":
    main()
