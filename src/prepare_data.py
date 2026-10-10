"""Step 1 - download, clean, de-duplicate and analyse the AfriSenti Kinyarwanda data.

Outputs
  data/processed/{train,dev,test}.csv   columns: tweet, label, text_light, text_full, agreement
  results/data_stats.json               dataset statistics used in the report
  results/figures/*.png                 class distribution / length histograms

Usage:  python src/prepare_data.py
"""
import json
import urllib.request
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import ANNOTATOR_URL, FIG_DIR, LABELS, PROC_DIR, RAW_DIR, RESULTS_DIR, SPLIT_URLS
from preprocess import full_clean, light_clean


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    targets = {f"kin_{s}.tsv": u for s, u in SPLIT_URLS.items()}
    targets["kinyarwanda_individual_labels.csv"] = ANNOTATOR_URL
    for name, url in targets.items():
        path = RAW_DIR / name
        if not path.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)


def load_split(split):
    df = pd.read_csv(RAW_DIR / f"kin_{split}.tsv", sep="\t", quoting=3, dtype=str)
    df = df.dropna().rename(columns={"tweet": "tweet", "label": "label"})
    df["label"] = df["label"].str.strip().str.lower()
    assert set(df.label) <= set(LABELS), set(df.label)
    return df


def annotator_agreement():
    """Map normalised tweet text -> number of annotators (out of 3) agreeing with the gold label.

    The raw annotation file keeps the original @handles, the released splits
    replace them with @user; light_clean() maps both to the same string.
    """
    ann = pd.read_csv(RAW_DIR / "kinyarwanda_individual_labels.csv", dtype=str)
    ann["key"] = ann["text"].map(light_clean).str.lower()
    labs = ann[["label_1", "label_2", "label_3"]].apply(lambda c: c.str.lower())
    return ann["key"], labs


def main():
    download()
    splits = {s: load_split(s) for s in ("train", "dev", "test")}
    stats = {"raw_sizes": {s: len(d) for s, d in splits.items()}}

    for s, d in splits.items():
        d["text_light"] = d["tweet"].map(light_clean)
        d["text_full"] = d["tweet"].map(full_clean)

    # ---- de-duplication / leakage removal (training split only, dev/test untouched)
    tr = splits["train"]
    key = tr["text_light"].str.lower()
    conflict_keys = tr.groupby(key)["label"].nunique()
    conflict_keys = set(conflict_keys[conflict_keys > 1].index)
    n_conflict = int(key.isin(conflict_keys).sum())
    tr = tr[~key.isin(conflict_keys)]
    n_before = len(tr)
    tr = tr.loc[~tr["text_light"].str.lower().duplicated()]
    n_dup = n_before - len(tr)
    eval_keys = set(pd.concat([splits["dev"], splits["test"]])["text_light"].str.lower())
    leak = tr["text_light"].str.lower().isin(eval_keys)
    n_leak = int(leak.sum())
    tr = tr[~leak]
    empty = tr["text_full"].str.len() == 0
    tr = tr[~empty]
    splits["train"] = tr.reset_index(drop=True)
    stats["cleaning"] = {
        "train_removed_conflicting_label_duplicates": n_conflict,
        "train_removed_exact_duplicates": n_dup,
        "train_removed_overlap_with_dev_test": n_leak,
        "train_removed_empty_after_cleaning": int(empty.sum()),
    }

    # ---- annotator agreement for every tweet (used in error analysis)
    keys, labs = annotator_agreement()
    lookup = {}
    for k, row in zip(keys, labs.itertuples(index=False)):
        lookup[k] = list(row)
    for s, d in splits.items():
        def agree(r):
            l = lookup.get(r.text_light.lower())
            return None if l is None else sum(x == r.label for x in l)
        d["agreement"] = d.apply(agree, axis=1)
        stats.setdefault("agreement_matched", {})[s] = int(d["agreement"].notna().sum())
        stats.setdefault("agreement_dist", {})[s] = {
            str(int(k)): int(v) for k, v in d["agreement"].value_counts().sort_index().items()}

    # ---- statistics
    for s, d in splits.items():
        n_tok = d["text_full"].str.split().map(len)
        stats.setdefault("sizes", {})[s] = len(d)
        stats.setdefault("label_dist", {})[s] = d["label"].value_counts().reindex(LABELS).to_dict()
        stats.setdefault("tokens_mean", {})[s] = round(float(n_tok.mean()), 2)
        stats.setdefault("tokens_max", {})[s] = int(n_tok.max())
        stats.setdefault("pct_with_emoji", {})[s] = round(
            float(d["tweet"].map(lambda t: any(ch in t for ch in "😂🤣😭🙏❤😍🥰😁😊👏🔥💔😢😡🙄😏")).mean() * 100), 1)

    train_vocab = Counter(t for x in splits["train"]["text_full"] for t in x.split())
    stats["train_vocab_size"] = len(train_vocab)
    stats["train_hapax_pct"] = round(100 * sum(1 for c in train_vocab.values() if c == 1) / len(train_vocab), 1)
    for s in ("dev", "test"):
        toks = [t for x in splits[s]["text_full"] for t in x.split()]
        stats.setdefault("oov_token_rate", {})[s] = round(100 * sum(t not in train_vocab for t in toks) / len(toks), 1)
    stats["top_tokens_per_class"] = {
        l: [w for w, _ in Counter(t for x in splits["train"].query("label == @l")["text_full"]
                                  for t in x.split() if len(t) > 3 or not t.isalpha()).most_common(15)]
        for l in LABELS}

    for s, d in splits.items():
        d[["tweet", "label", "text_light", "text_full", "agreement"]].to_csv(PROC_DIR / f"{s}.csv", index=False)
    with open(RESULTS_DIR / "data_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # ---- figures
    fig, ax = plt.subplots(figsize=(6, 3.5))
    dist = pd.DataFrame({s: splits[s]["label"].value_counts().reindex(LABELS) for s in splits})
    dist.plot.bar(ax=ax, rot=0, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("tweets"); ax.set_title("Label distribution per split")
    fig.tight_layout(); fig.savefig(FIG_DIR / "label_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for l in LABELS:
        ax.hist(splits["train"].query("label == @l")["text_full"].str.split().map(len),
                bins=30, alpha=0.5, label=l)
    ax.set_xlabel("tokens per tweet (after cleaning)"); ax.set_ylabel("tweets"); ax.legend()
    ax.set_title("Tweet length by class (train)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "length_histogram.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
