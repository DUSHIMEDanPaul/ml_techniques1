"""Step 3 - train static word embeddings for Kinyarwanda.

Corpus = KINNEWS news articles (Niyongabo et al., 2020; ~21k articles, used here
only as *unlabelled* text - its topic labels are ignored) + the AfriSenti
training tweets. Dev/test tweets are never used.

Two models are trained with gensim:
  * Word2Vec skip-gram (Mikolov et al., 2013)
  * FastText skip-gram with character n-grams (Bojanowski et al., 2017) - can
    build vectors for unseen, morphologically complex words.

Usage:  python src/train_embeddings.py
"""
import re
import urllib.request
import zipfile

import pandas as pd
from gensim.models import FastText, Word2Vec

from config import MODELS_DIR, ROOT
from metrics import load_splits
from preprocess import full_clean

KINNEWS_URL = "https://github.com/saradhix/kinnews_kirnews/raw/master/KINNEWS.zip"
EXT_DIR = ROOT / "data" / "external" / "kinnews"
DIM = 100


def kinnews_sentences():
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    zpath = EXT_DIR / "KINNEWS.zip"
    if not zpath.exists():
        print("downloading KINNEWS ...")
        urllib.request.urlretrieve(KINNEWS_URL, zpath)
    csv = EXT_DIR / "KINNEWS" / "raw" / "train.csv"
    if not csv.exists():
        zipfile.ZipFile(zpath).extractall(EXT_DIR)
    sents = []
    for split in ("train", "test"):
        df = pd.read_csv(EXT_DIR / "KINNEWS" / "raw" / f"{split}.csv", dtype=str).fillna("")
        for text in (df["title"] + ". " + df["content"]):
            for s in re.split(r"(?<=[.!?])\s+", text):
                toks = full_clean(s).split()
                if len(toks) >= 3:
                    sents.append(toks)
    return sents


def main():
    news = kinnews_sentences()
    tweets = [t.split() for t in load_splits()["train"]["text_full"] if t]
    corpus = news + tweets * 3          # up-weight the in-domain tweet text
    n_tok = sum(map(len, news))
    print(f"KINNEWS sentences: {len(news):,}  tokens: {n_tok:,}; tweets: {len(tweets):,}")
    common = dict(vector_size=DIM, window=5, min_count=2, sg=1, epochs=5, workers=8, seed=42)
    w2v = Word2Vec(corpus, **common)
    w2v.wv.save(str(MODELS_DIR / "w2v_kin.kv"))
    ft = FastText(corpus, min_n=3, max_n=6, **common)
    ft.wv.save(str(MODELS_DIR / "fasttext_kin.kv"))
    print(f"vocab: w2v {len(w2v.wv):,}  fasttext {len(ft.wv):,}")
    for w in ["mwiza", "nabi", "urukundo", "ikibazo", "neza"]:
        if w in w2v.wv:
            print(w, "->", [x for x, _ in w2v.wv.most_similar(w, topn=6)])
    # FastText can embed words never seen in training (e.g. elongated / misspelled forms)
    print("fasttext OOV 'nibyizaaa' ->", [x for x, _ in ft.wv.most_similar("nibyizaaa", topn=5)])


if __name__ == "__main__":
    main()
