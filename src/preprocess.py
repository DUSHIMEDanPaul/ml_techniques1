"""Text normalisation for Kinyarwanda tweets.

Two levels of cleaning are used in the project:

* ``light_clean``  - for subword Transformer models. Keeps case, punctuation and
  emojis (the pretrained tokenizers can handle them) but removes noise that
  carries no sentiment: URLs, runs of anonymised user mentions, extra spaces.
* ``full_clean``   - for the classical / word-level models (TF-IDF, Word2Vec,
  BiLSTM). Additionally lower-cases, reduces character elongations
  ("byizaaaaa" -> "byizaa"), splits emojis into separate tokens and removes
  punctuation that is not part of a word.

The same functions are imported by the web app so that training-time and
inference-time preprocessing are identical.
"""
import re

import emoji

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
MULTI_MENTION_RE = re.compile(r"(@user\s*){2,}")
ELONG_RE = re.compile(r"(\w)\1{2,}")           # 3+ repeated word chars
PUNCT_RE = re.compile(r"[^\w\s'’<>]")           # keep apostrophes (n'umugore)
SPACE_RE = re.compile(r"\s+")
HASHTAG_RE = re.compile(r"#(\w+)")


def light_clean(text: str) -> str:
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub("@user", text)
    text = MULTI_MENTION_RE.sub("@user ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return SPACE_RE.sub(" ", text).strip()


def _space_emojis(text: str) -> str:
    # surround every emoji with spaces so that "nibyiza😂😂" -> "nibyiza 😂 😂"
    return emoji.replace_emoji(text, replace=lambda e, _: f" {e} ")


def full_clean(text: str) -> str:
    text = light_clean(text).lower()
    text = text.replace("@user", " ")
    text = text.replace("’", "'")
    text = ELONG_RE.sub(r"\1\1", text)
    text = _space_emojis(text)
    # protect emojis from punctuation removal
    tokens = []
    for tok in text.split():
        if emoji.is_emoji(tok):
            tokens.append(tok)
        else:
            tok = PUNCT_RE.sub(" ", tok)
            tok = re.sub(r"\d+", " <num> ", tok)
            tokens.append(tok)
    text = " ".join(tokens)
    return SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list:
    """Whitespace tokenisation of a ``full_clean``-ed string."""
    return full_clean(text).split()


if __name__ == "__main__":
    s = "@user @user @user Hhhhhh ntabyihogoza, ubu x abo yishe 😂😂 https://t.co/x #Rwanda"
    print(light_clean(s))
    print(full_clean(s))
