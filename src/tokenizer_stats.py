"""Tokeniser analysis: subword fertility (subwords per whitespace word) and [UNK] rate
of each pretrained tokenizer on the AfriSenti-kin training tweets.

Usage:  python src/tokenizer_stats.py   -> results/tokenizer_stats.json
"""
import json

from transformers import AutoTokenizer

from config import RESULTS_DIR
from metrics import load_splits

MODELS = ["castorini/afriberta_small", "Davlan/afro-xlmr-small", "Davlan/afro-xlmr-base", "jean-paul/KinyaBERT-small"]
EXAMPLE = "Ese umuntu aguhaye miliyoni 7 zidorali ngo aryamane numugore wawe 🙄"


def main():
    texts = list(load_splits()["train"]["text_light"])
    n_words = sum(len(t.split()) for t in texts)
    out = {}
    for m in MODELS:
        tok = AutoTokenizer.from_pretrained(m)
        pieces = [tok.tokenize(t) for t in texts]
        n = sum(map(len, pieces))
        unk = sum(p == tok.unk_token for s in pieces for p in s)
        out[m] = {"vocab_size": len(tok), "subwords_per_word": round(n / n_words, 2),
                  "unk_rate_pct": round(100 * unk / n, 2), "example": tok.tokenize(EXAMPLE)}
        print(m, out[m])
    (RESULTS_DIR / "tokenizer_stats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
