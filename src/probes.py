"""Step 7 - behavioural probes and out-of-domain checks for a saved Transformer model.

1. Hand-written probe sentences grouped by phenomenon (plain sentiment, negation,
   sarcasm, neutral announcements). These are written by the author purely as a
   *test* of specific behaviours - they are never used for training - and the
   intended label is the author's judgement.
2. Domain shift: KINNEWS news headlines (formal news register instead of Twitter).
   No sentiment gold labels exist, so we report the predicted label distribution
   and the mean confidence.
3. Non-Kinyarwanda input (English, French, Swahili, nonsense) - the model should
   ideally be *uncertain*; we report its confidence.

Usage:  python src/probes.py --model models/final
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import RESULTS_DIR, ROOT
from preprocess import light_clean

PROBES = [
    # (text, intended label, phenomenon, English gloss)
    ("Ndishimye cyane uyu munsi", "positive", "plain", "I am very happy today"),
    ("Uyu muziki ni mwiza cyane", "positive", "plain", "This music is very good"),
    ("Murakoze cyane ku bufasha bwanyu", "positive", "plain", "Thank you very much for your help"),
    ("Amavubi yatsinze! Twishimye cyane 🇷🇼", "positive", "plain", "Amavubi won! We are very happy"),
    ("Imana ishimwe, umwana wanjye yarakize", "positive", "plain", "Praise God, my child has recovered"),
    ("Ndababaye cyane", "negative", "plain", "I am very sad"),
    ("Iyi serivisi ni mbi cyane", "negative", "plain", "This service is very bad"),
    ("Ibiciro biri hejuru cyane, abantu barashonje", "negative", "plain", "Prices are very high, people are hungry"),
    ("Amavubi yongeye gutsindwa, birababaje", "negative", "plain", "Amavubi lost again, it is sad"),
    ("Inama izaba ejo saa tatu za mu gitondo", "neutral", "plain", "The meeting will be tomorrow at 9am"),
    ("Minisiteri y'Uburezi yatangaje ingengabihe nshya y'amashuri", "neutral", "plain",
     "The Ministry of Education announced a new school calendar"),
    ("Ese ejo hazaba imvura?", "neutral", "plain", "Will it rain tomorrow?"),
    ("Ntabwo ari byiza", "negative", "negation", "It is not good"),
    ("Ntabwo nishimiye ibyo wakoze", "negative", "negation", "I am not pleased with what you did"),
    ("Si bibi na gato", "positive", "negation", "It is not bad at all"),
    ("Wowe uri umuhanga pe 😂😂", "negative", "sarcasm", "You are really clever 😂😂 (mocking)"),
    ("Ni byiza cyane kubura amazi iminsi itatu 🙄", "negative", "sarcasm",
     "It is great to have no water for three days 🙄"),
    ("Byiza", "positive", "very_short", "Good"),
    ("Oya", "negative", "very_short", "No"),
]

NON_KIN = [
    ("I love this so much, best day ever!", "english"),
    ("This government is terrible", "english"),
    ("Je suis très content aujourd'hui", "french"),
    ("Nimefurahi sana leo", "swahili"),
    ("asdf qwerty zxcv", "nonsense"),
]


@torch.no_grad()
def run(model, tok, texts):
    enc = tok([light_clean(t) for t in texts], truncation=True, max_length=128, padding=True, return_tensors="pt")
    p = torch.softmax(model(**enc).logits, -1).numpy()
    labs = [model.config.id2label[i].lower() for i in p.argmax(1)]
    return labs, p.max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "final"))
    ap.add_argument("--n_news", type=int, default=300)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).eval()
    out = {}

    labs, conf = run(model, tok, [p[0] for p in PROBES])
    rows = [{"text": t, "gloss": g, "phenomenon": ph, "intended": y, "pred": l, "conf": round(float(c), 3),
             "correct": y == l} for (t, y, ph, g), l, c in zip(PROBES, labs, conf)]
    df = pd.DataFrame(rows)
    out["probe_accuracy_by_phenomenon"] = df.groupby("phenomenon")["correct"].agg(["mean", "count"]).round(3).to_dict("index")
    out["probes"] = rows

    news_csv = ROOT / "data" / "external" / "kinnews" / "KINNEWS" / "raw" / "test.csv"
    if not news_csv.exists():           # download KINNEWS (same source as train_embeddings.py)
        import urllib.request, zipfile
        from train_embeddings import EXT_DIR, KINNEWS_URL
        EXT_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(KINNEWS_URL, EXT_DIR / "KINNEWS.zip")
        zipfile.ZipFile(EXT_DIR / "KINNEWS.zip").extractall(EXT_DIR)
    news = pd.read_csv(news_csv, dtype=str)
    titles = news["title"].dropna().sample(a.n_news, random_state=0).tolist()
    labs, conf = run(model, tok, titles)
    out["kinnews_headlines"] = {"n": len(titles), "pred_distribution": pd.Series(labs).value_counts(normalize=True).round(3).to_dict(),
                                "mean_confidence": round(float(np.mean(conf)), 3),
                                "examples": [{"text": t, "pred": l, "conf": round(float(c), 3)}
                                             for t, l, c in list(zip(titles, labs, conf))[:10]]}

    labs, conf = run(model, tok, [t for t, _ in NON_KIN])
    out["non_kinyarwanda"] = [{"text": t, "lang": lang, "pred": l, "conf": round(float(c), 3)}
                              for (t, lang), l, c in zip(NON_KIN, labs, conf)]

    # in-domain reference: mean confidence on the AfriSenti test set
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv", keep_default_na=False)
    _, conf = run(model, tok, test["tweet"].tolist()[:300])
    out["afrisenti_test_mean_confidence_first300"] = round(float(np.mean(conf)), 3)

    (RESULTS_DIR / "probes.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(df[["text", "intended", "pred", "conf", "phenomenon"]].to_string(index=False))
    print(json.dumps({k: v for k, v in out.items() if k not in ("probes",)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
