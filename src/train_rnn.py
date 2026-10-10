"""Step 4 - recurrent neural classifiers (BiLSTM / BiGRU) on word embeddings.

Architecture
  tokens -> Embedding (random | Word2Vec | FastText, frozen or fine-tuned)
         -> dropout -> bidirectional LSTM/GRU (1 layer, 128 units / direction)
         -> pooling over time: [max ; mean]  or  additive attention (Bahdanau-style)
         -> dropout -> Linear(3)

Training: Adam (lr 1e-3), cross-entropy, batch 32, up to 15 epochs, early
stopping on dev macro-F1 (patience 3). Each configuration is run with 3 random
seeds; mean (and std) over seeds is reported and the predictions of the best
dev seed are saved.

Usage:  python src/train_rnn.py            # all configurations
        python src/train_rnn.py --only bilstm_w2v_ft
"""
import argparse
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors

from config import LABEL2ID, LABELS, MODELS_DIR
from metrics import compute_metrics, load_splits, log_experiment, save_predictions

torch.set_num_threads(8)
PAD, UNK = 0, 1
MAX_LEN = 64

CONFIGS = {
    # name:               (cell,  embeddings, freeze, pooling)
    "bilstm_random":       ("lstm", None,       False, "maxmean"),
    "bilstm_w2v_frozen":   ("lstm", "w2v",      True,  "maxmean"),
    "bilstm_w2v_ft":       ("lstm", "w2v",      False, "maxmean"),
    "bilstm_fasttext_ft":  ("lstm", "fasttext", False, "maxmean"),
    "bilstm_fasttext_attn": ("lstm", "fasttext", False, "attention"),
    "bigru_fasttext_attn": ("gru",  "fasttext", False, "attention"),
}


class Attention(nn.Module):
    """Additive attention pooling: score_t = v^T tanh(W h_t)."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, h, mask):
        scores = self.v(torch.tanh(self.proj(h))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=-1)
        return (alpha.unsqueeze(-1) * h).sum(1), alpha


class RNNClassifier(nn.Module):
    def __init__(self, emb_matrix, freeze, cell="lstm", hidden=128, pooling="maxmean", n_cls=3, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix, dtype=torch.float),
                                                freeze=freeze, padding_idx=PAD)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(emb_matrix.shape[1], hidden, batch_first=True, bidirectional=True)
        self.pooling = pooling
        self.drop = nn.Dropout(dropout)
        if pooling == "attention":
            self.attn = Attention(2 * hidden)
            out_dim = 2 * hidden
        else:
            out_dim = 4 * hidden
        self.fc = nn.Linear(out_dim, n_cls)

    def forward(self, x):
        mask = x != PAD
        lengths = mask.sum(1).clamp(min=1).cpu()
        e = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        if self.pooling == "attention":
            z, _ = self.attn(h, mask)
        else:
            m = mask.unsqueeze(-1)
            mx = h.masked_fill(~m, -1e9).max(1).values
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
            z = torch.cat([mx, mean], -1)
        return self.fc(self.drop(z))


def build_vocab(train_texts, kv=None, extra_texts=()):
    """Vocabulary = training tokens (+ the pretrained vocabulary, + for FastText
    the dev/test tokens whose vectors are composed from character n-grams)."""
    cnt = Counter(t for s in train_texts for t in s.split())
    words = list(cnt)
    if kv is not None:
        words += [w for w in kv.index_to_key if w not in cnt]
        if extra_texts:
            seen = set(words)
            words += [t for s in extra_texts for t in s.split() if t not in seen and not seen.add(t)]
    return {w: i + 2 for i, w in enumerate(words)}


def embedding_matrix(vocab, kv, dim=100, rng=None):
    rng = rng or np.random.default_rng(0)
    m = rng.normal(0, 0.1, (len(vocab) + 2, dim)).astype(np.float32)
    m[PAD] = 0
    hits = 0
    if kv is not None:
        for w, i in vocab.items():
            try:
                m[i] = kv[w]          # FastText also returns vectors for OOV words
                hits += 1
            except KeyError:
                pass
    return m, hits


def encode(texts, vocab):
    out = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, s in enumerate(texts):
        ids = [vocab.get(t, UNK) for t in s.split()][:MAX_LEN]
        out[i, :len(ids)] = ids
    return torch.tensor(out)


def predict(model, X, bs=256):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            probs.append(torch.softmax(model(X[i:i + bs]), -1))
    return torch.cat(probs).numpy()


def train_one(cfg, data, seed, epochs=15, patience=3):
    cell, emb_name, freeze, pooling = cfg
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    kv = None
    if emb_name:
        kv = KeyedVectors.load(str(MODELS_DIR / ("w2v_kin.kv" if emb_name == "w2v" else "fasttext_kin.kv")))
    extra = list(data["dev"]["text_full"]) + list(data["test"]["text_full"]) if emb_name == "fasttext" else ()
    vocab = build_vocab(data["train"]["text_full"], kv, extra)
    emb, hits = embedding_matrix(vocab, kv, rng=np.random.default_rng(seed))
    X = {s: encode(list(data[s]["text_full"]), vocab) for s in data}
    y = {s: torch.tensor([LABEL2ID[l] for l in data[s]["label"]]) for s in data}

    model = RNNClassifier(emb, freeze, cell=cell, pooling=pooling)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state, bad = -1, None, 0
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X["train"]))
        tot = 0
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = loss_fn(model(X["train"][idx]), y["train"][idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        dev_pred = predict(model, X["dev"]).argmax(1)
        f1 = compute_metrics([LABELS[i] for i in y["dev"]], [LABELS[i] for i in dev_pred])["macro_f1"]
        history.append({"epoch": ep + 1, "train_loss": round(tot / len(perm), 4), "dev_macro_f1": f1})
        if f1 > best_f1:
            best_f1, best_state, bad = f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, X, best_f1, history, hits, len(vocab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seeds", type=int, nargs="*", default=[13, 42, 2024])
    args = ap.parse_args()
    data = load_splits()
    for name, cfg in CONFIGS.items():
        if args.only and name not in args.only:
            continue
        runs = []
        for seed in args.seeds:
            model, X, dev_f1, hist, hits, vsize = train_one(cfg, data, seed)
            pdev, pte = predict(model, X["dev"]), predict(model, X["test"])
            runs.append({"seed": seed, "dev_f1": dev_f1, "pdev": pdev, "pte": pte, "hist": hist})
            print(f"  {name} seed={seed} dev F1={dev_f1:.4f} epochs={len(hist)}")
        test_f1s = [compute_metrics(data["test"]["label"], [LABELS[i] for i in r["pte"].argmax(1)])["macro_f1"]
                    for r in runs]
        best = max(runs, key=lambda r: r["dev_f1"])
        dev_lab = [LABELS[i] for i in best["pdev"].argmax(1)]
        te_lab = [LABELS[i] for i in best["pte"].argmax(1)]
        log_experiment(name, "rnn",
                       {"cell": cfg[0], "embeddings": cfg[1] or "random", "frozen": cfg[2], "pooling": cfg[3],
                        "hidden": 128, "vocab_size": vsize, "pretrained_hits": hits, "seeds": args.seeds},
                       data["dev"]["label"], dev_lab, data["test"]["label"], te_lab,
                       extra={"test_macro_f1_seeds": test_f1s,
                              "test_macro_f1_mean": round(float(np.mean(test_f1s)), 4),
                              "test_macro_f1_std": round(float(np.std(test_f1s)), 4),
                              "history_best_seed": best["hist"]})
        save_predictions(name, data["test"], te_lab, best["pte"])


if __name__ == "__main__":
    main()
