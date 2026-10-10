"""Step 2 - classical baselines.

* Majority class                          (lower bound)
* TF-IDF word 1-2 grams + Multinomial NB
* TF-IDF word 1-2 grams + Logistic Regression
* TF-IDF char 2-5 grams + Logistic Regression   (robust to morphology / spelling variation)
* TF-IDF word + char    + Linear SVM

The regularisation strength C (or NB alpha) is selected on the dev set; the test
set is used only once, for the final number. The best model is saved for the
web app as a lightweight fallback.

Usage:  python src/baselines.py
"""
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from config import MODELS_DIR, SEED
from metrics import compute_metrics, load_splits, log_experiment, save_predictions


def word_vec():
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                           token_pattern=r"[^\s]+")


def char_vec():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)


def candidates():
    return {
        "tfidf_word_nb": (lambda a: Pipeline([("v", word_vec()), ("c", MultinomialNB(alpha=a))]),
                          [0.1, 0.3, 0.5, 1.0], "alpha"),
        "tfidf_word_lr": (lambda c: Pipeline([("v", word_vec()), ("c", LogisticRegression(
            C=c, max_iter=2000, class_weight="balanced", random_state=SEED))]), [0.5, 1, 2, 5, 10], "C"),
        "tfidf_char_lr": (lambda c: Pipeline([("v", char_vec()), ("c", LogisticRegression(
            C=c, max_iter=2000, class_weight="balanced", random_state=SEED))]), [0.5, 1, 2, 5, 10], "C"),
        "tfidf_word+char_svm": (lambda c: Pipeline([("v", FeatureUnion([("w", word_vec()), ("ch", char_vec())])),
                                                    ("c", LinearSVC(C=c, class_weight="balanced",
                                                                    random_state=SEED))]),
                                [0.05, 0.1, 0.3, 1.0], "C"),
    }


def main():
    d = load_splits()
    Xtr, ytr = d["train"]["text_full"], d["train"]["label"]
    Xdv, ydv = d["dev"]["text_full"], d["dev"]["label"]
    Xte, yte = d["test"]["text_full"], d["test"]["label"]

    maj = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    log_experiment("majority_class", "baseline", {"strategy": "most_frequent"},
                   ydv, maj.predict(Xdv), yte, maj.predict(Xte))

    best = (None, -1, None)
    for name, (make, grid, hp) in candidates().items():
        scores = []
        for v in grid:
            m = make(v).fit(Xtr, ytr)
            scores.append(compute_metrics(ydv, m.predict(Xdv))["macro_f1"])
        v = grid[int(np.argmax(scores))]
        m = make(v).fit(Xtr, ytr)
        print(f"{name}: dev F1 per {hp} {dict(zip(grid, scores))} -> {hp}={v}")
        rec = log_experiment(name, "classical", {hp: v, "dev_grid": dict(zip(map(str, grid), scores))},
                             ydv, m.predict(Xdv), yte, m.predict(Xte))
        proba = m.predict_proba(Xte) if hasattr(m, "predict_proba") else None
        save_predictions(name, d["test"], m.predict(Xte), proba)
        if rec["dev"]["macro_f1"] > best[1] and hasattr(m, "predict_proba"):
            best = (name, rec["dev"]["macro_f1"], m)

    print(f"best probabilistic classical model on dev: {best[0]}")
    joblib.dump(best[2], MODELS_DIR / "tfidf_baseline.joblib")


if __name__ == "__main__":
    main()
