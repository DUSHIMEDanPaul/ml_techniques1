"""Step 6 - experiment summary, evaluation figures and error analysis.

Reads results/experiments.jsonl and results/predictions/*_test.csv and writes
  results/experiments_table.md / .csv   the experiment table for the report
  results/figures/model_comparison.png
  results/figures/confusion_<model>.png
  results/error_analysis.json           sliced accuracy, significance tests, CIs
  results/errors_<model>.csv            all misclassified test tweets (for manual inspection)
  results/examples_<model>.md           correct / incorrect examples for the report

Usage:  python src/analyze.py --best <final_model_name> --baseline tfidf_char_lr
"""
import argparse
import json
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score

from config import FIG_DIR, LABELS, RESULTS_DIR
from metrics import load_splits
from preprocess import full_clean

EXP_LOG = RESULTS_DIR / "experiments.jsonl"
PRED_DIR = RESULTS_DIR / "predictions"
LAUGH = ("😂", "🤣", "😅", "😆")
# frequent English / French function words - a rough code-mixing detector
FOREIGN = set("the and you is are of to in for with this that my your not it be on what so but have just "
              "le la les de des et est pas pour une un que qui dans je tu il vous nous".split())


def experiment_table():
    rows = [json.loads(l) for l in EXP_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    order = {"baseline": 0, "classical": 1, "rnn": 2, "transformer": 3}
    rows.sort(key=lambda r: (order[r["family"]], r["test"]["macro_f1"]))
    recs = []
    for r in rows:
        c = r["config"]
        if r["family"] == "transformer":
            desc = f"{c['model'].split('/')[-1]}, {c['strategy']}, lr={c['lr']}, trainable={c['trainable_params'] / 1e6:.2f}M"
        elif r["family"] == "rnn":
            desc = (f"{'Bi' + c['cell'].upper()}, emb={c['embeddings']}{' (frozen)' if c['frozen'] else ''}, "
                    f"pool={c['pooling']}")
        else:
            desc = ", ".join(f"{k}={v}" for k, v in c.items() if k != "dev_grid")
        seed = (f"{r['test_macro_f1_mean']:.4f} ± {r['test_macro_f1_std']:.4f}"
                if "test_macro_f1_mean" in r else "-")
        recs.append({"Experiment": r["name"], "Family": r["family"], "Configuration": desc,
                     "Dev macro-F1": r["dev"]["macro_f1"], "Test accuracy": r["test"]["accuracy"],
                     "Test macro-P": r["test"]["macro_precision"], "Test macro-R": r["test"]["macro_recall"],
                     "Test macro-F1": r["test"]["macro_f1"], "Test weighted-F1": r["test"]["weighted_f1"],
                     "Test macro-F1 (3 seeds mean ± std)": seed,
                     "Train time (s)": r.get("train_time_s", "-")})
    df = pd.DataFrame(recs)
    df.to_csv(RESULTS_DIR / "experiments_table.csv", index=False)
    (RESULTS_DIR / "experiments_table.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    print(df[["Experiment", "Dev macro-F1", "Test accuracy", "Test macro-F1"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(df) + 1))
    colors = {"baseline": "#999999", "classical": "#4C72B0", "rnn": "#DD8452", "transformer": "#55A868"}
    ax.barh(df["Experiment"], df["Test macro-F1"], color=[colors[f] for f in df["Family"]])
    for i, v in enumerate(df["Test macro-F1"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Test macro-F1"); ax.set_xlim(0, 1)
    ax.set_title("Model comparison (AfriSenti-kin test set)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "model_comparison.png", dpi=150); plt.close(fig)
    return rows


def plot_learning_curves(rows):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for r in rows:
        h = r.get("history") or r.get("history_best_seed")
        if h and r["family"] in ("transformer", "rnn") and (r["family"] == "transformer" or r["name"] == "bilstm_fasttext_attn"):
            ax.plot([e["epoch"] for e in h], [e["dev_macro_f1"] for e in h], marker="o", label=r["name"])
    ax.set_xlabel("epoch"); ax.set_ylabel("dev macro-F1"); ax.set_title("Dev macro-F1 per epoch")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "learning_curves.png", dpi=150); plt.close(fig)


def plot_confusion(name, df):
    cm = confusion_matrix(df["label"], df["pred"], labels=LABELS)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"confusion_{name}.png", dpi=150); plt.close(fig)


def bootstrap_f1(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    scores = [f1_score(y[i], p[i], average="macro") for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)]


def mcnemar(y, a, b):
    ca, cb = (np.asarray(a) == np.asarray(y)), (np.asarray(b) == np.asarray(y))
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": round(float(p), 5)}


def slice_acc(df, mask):
    m = np.asarray(mask)
    return {"n": int(m.sum()), "accuracy": round(float((df["pred"][m] == df["label"][m]).mean()), 4) if m.sum() else None}


def slices(df, train_vocab):
    toks = df["tweet"].map(full_clean).str.split()
    oov = toks.map(lambda t: np.mean([w not in train_vocab for w in t]) if t else 0)
    n = toks.map(len)
    conf = df[[f"p_{l}" for l in LABELS]].max(axis=1) if "p_negative" in df else None
    out = {
        "annotator_agreement_3of3": slice_acc(df, df["agreement"] == 3),
        "annotator_agreement_2of3": slice_acc(df, df["agreement"] == 2),
        "annotator_agreement_unknown": slice_acc(df, df["agreement"].isna()),
        "has_laughing_emoji": slice_acc(df, df["tweet"].map(lambda t: any(e in t for e in LAUGH))),
        "no_laughing_emoji": slice_acc(df, ~df["tweet"].map(lambda t: any(e in t for e in LAUGH))),
        "code_mixed_en_fr": slice_acc(df, toks.map(lambda t: sum(w in FOREIGN for w in t) >= 2)),
        "not_code_mixed": slice_acc(df, toks.map(lambda t: sum(w in FOREIGN for w in t) < 2)),
        "short_<=5_tokens": slice_acc(df, n <= 5),
        "medium_6-15_tokens": slice_acc(df, (n > 5) & (n <= 15)),
        "long_>15_tokens": slice_acc(df, n > 15),
        "oov_rate_<20%": slice_acc(df, oov < 0.2),
        "oov_rate_20-50%": slice_acc(df, (oov >= 0.2) & (oov < 0.5)),
        "oov_rate_>=50%": slice_acc(df, oov >= 0.5),
        "contains_negation_si/nta/ntabwo": slice_acc(df, toks.map(
            lambda t: any(w in ("ntabwo", "nta", "si", "ntago", "ntibyo") or w.startswith(("nti", "ntu")) for w in t))),
    }
    if conf is not None:
        for lo, hi in [(0, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
            out[f"confidence_{lo}-{min(hi, 1)}"] = slice_acc(df, (conf >= lo) & (conf < hi))
    return out


def examples_md(name, df, k=12):
    df = df.copy()
    df["conf"] = df[[f"p_{l}" for l in LABELS]].max(axis=1)
    ok = df[df.pred == df.label].sort_values("conf", ascending=False)
    bad = df[df.pred != df.label].sort_values("conf", ascending=False)
    rng = np.random.default_rng(1)
    lines = [f"# Examples for {name}\n", "## Correct, highest confidence\n", "| tweet | gold | pred | conf |", "|---|---|---|---|"]
    for r in ok.head(k).itertuples():
        lines.append(f"| {r.tweet.replace('|', '/')} | {r.label} | {r.pred} | {r.conf:.2f} |")
    lines += ["\n## Wrong, highest confidence (most 'confident mistakes')\n", "| tweet | gold | pred | conf | agreement |",
              "|---|---|---|---|---|"]
    for r in bad.head(k * 2).itertuples():
        lines.append(f"| {r.tweet.replace('|', '/')} | {r.label} | {r.pred} | {r.conf:.2f} | {r.agreement} |")
    lines += ["\n## Wrong, random sample\n", "| tweet | gold | pred | conf | agreement |", "|---|---|---|---|---|"]
    for r in bad.iloc[rng.permutation(len(bad))[:k * 2]].itertuples():
        lines.append(f"| {r.tweet.replace('|', '/')} | {r.label} | {r.pred} | {r.conf:.2f} | {r.agreement} |")
    (RESULTS_DIR / f"examples_{name}.md").write_text("\n".join(lines), encoding="utf-8")
    bad.to_csv(RESULTS_DIR / f"errors_{name}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True)
    ap.add_argument("--baseline", default="tfidf_char_lr")
    ap.add_argument("--compare", nargs="*", default=[])
    a = ap.parse_args()
    rows = experiment_table()
    plot_learning_curves(rows)
    data = load_splits()
    train_vocab = Counter(t for s in data["train"]["text_full"] for t in s.split())

    report = {}
    names = [a.best, a.baseline] + a.compare
    preds = {}
    for n in names:
        df = pd.read_csv(PRED_DIR / f"{n}_test.csv", keep_default_na=False, na_values=[""])
        df["agreement"] = pd.to_numeric(df["agreement"], errors="coerce")
        preds[n] = df
        plot_confusion(n, df)
        report[n] = {"macro_f1": round(f1_score(df.label, df.pred, average="macro"), 4),
                     "macro_f1_95ci": bootstrap_f1(df.label, df.pred), "slices": slices(df, train_vocab)}
    examples_md(a.best, preds[a.best])
    y = preds[a.best]["label"]
    report["mcnemar_best_vs_baseline"] = mcnemar(y, preds[a.best]["pred"], preds[a.baseline]["pred"])
    for n in a.compare:
        report[f"mcnemar_best_vs_{n}"] = mcnemar(y, preds[a.best]["pred"], preds[n]["pred"])
    both_wrong = (preds[a.best]["pred"] != y) & (preds[a.baseline]["pred"] != y)
    report["errors_shared_with_baseline"] = {
        "best_errors": int((preds[a.best]["pred"] != y).sum()), "both_wrong": int(both_wrong.sum())}
    b = preds[a.best]
    pairs = Counter(zip(b.label[b.pred != b.label], b.pred[b.pred != b.label]))
    report["confusion_pairs_best"] = {f"{g}->{p}": c for (g, p), c in pairs.most_common()}
    (RESULTS_DIR / "error_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
