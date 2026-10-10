"""Shared evaluation helpers: metrics, experiment log, prediction dumps."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from config import LABELS, PROC_DIR, RESULTS_DIR

EXP_LOG = RESULTS_DIR / "experiments.jsonl"


def load_splits():
    return {s: pd.read_csv(PROC_DIR / f"{s}.csv", keep_default_na=False,
                           dtype={"tweet": str, "text_light": str, "text_full": str})
            for s in ("train", "dev", "test")}


def compute_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4), "macro_precision": round(p, 4),
            "macro_recall": round(r, 4), "macro_f1": round(f, 4), "weighted_f1": round(wf, 4)}


def log_experiment(name, family, config, dev_true, dev_pred, test_true, test_pred, extra=None):
    """Append one experiment (dev + test metrics) to results/experiments.jsonl."""
    rec = {"name": name, "family": family, "config": config,
           "dev": compute_metrics(dev_true, dev_pred), "test": compute_metrics(test_true, test_pred),
           "test_per_class": classification_report(test_true, test_pred, labels=LABELS, output_dict=True,
                                                   zero_division=0),
           "test_confusion": confusion_matrix(test_true, test_pred, labels=LABELS).tolist(),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        rec.update(extra)
    # replace an earlier run with the same name
    rows = []
    if EXP_LOG.exists():
        rows = [json.loads(l) for l in EXP_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [r for r in rows if r["name"] != name]
    rows.append(rec)
    EXP_LOG.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"[{name}] dev macro-F1={rec['dev']['macro_f1']:.4f}  test macro-F1={rec['test']['macro_f1']:.4f} "
          f"acc={rec['test']['accuracy']:.4f}")
    return rec


def save_predictions(name, df, pred, proba=None):
    out = df[["tweet", "label", "agreement"]].copy()
    out["pred"] = pred
    if proba is not None:
        proba = np.asarray(proba)
        for i, l in enumerate(LABELS):
            out[f"p_{l}"] = proba[:, i].round(4)
    pdir = RESULTS_DIR / "predictions"
    pdir.mkdir(exist_ok=True)
    out.to_csv(pdir / f"{name}_test.csv", index=False)
