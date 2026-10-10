"""Step 5 - fine-tune pretrained Transformer encoders for 3-way sentiment.

Models compared (all from the Hugging Face Hub):
  * castorini/afriberta_small   - 4-layer XLM-R-style model pretrained from scratch on
                                  11 African languages incl. Gahuza (Kinyarwanda+Kirundi)
                                  (Ogueji et al., 2021)
  * Davlan/afro-xlmr-small      - XLM-R adapted to 17 African languages incl. Kinyarwanda
                                  with MAFT, vocabulary reduced to 70k (Alabi et al., 2022)
  * jean-paul/KinyaBERT-small   - BERT pretrained on Kinyarwanda only

Strategies:
  * full fine-tuning (all weights updated)
  * --freeze  : encoder frozen, only the classification head is trained (linear probe)
  * --lora    : LoRA adapters on attention query/value (Hu et al., 2022) + trainable head

Classification head: the standard *ForSequenceClassification head (first-token
representation -> dense -> tanh -> linear). Optimiser AdamW, linear warm-up/decay
schedule, gradient clipping 1.0, early stopping on dev macro-F1.

Usage examples
  python src/finetune_transformer.py --model castorini/afriberta_small --name afriberta_small_ft
  python src/finetune_transformer.py --model Davlan/afro-xlmr-small --name afroxlmr_small_lora --lora --lr 5e-4
"""
import argparse
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import ID2LABEL, LABEL2ID, LABELS, MODELS_DIR
from metrics import compute_metrics, load_splits, log_experiment, save_predictions

torch.set_num_threads(12)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--freeze_embeddings", action="store_true",
                    help="full fine-tuning but with the (70k x 768) embedding matrix frozen - saves ~40%% optimiser memory")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--save", action="store_true", help="save the best model to models/<name>")
    return ap.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def batches(tok, texts, labels, bs, max_len, shuffle):
    idx = list(range(len(texts)))

    def collate(b):
        enc = tok([texts[i] for i in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([labels[i] for i in b])
        return enc
    return DataLoader(idx, batch_size=bs, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    out = []
    for b in loader:
        b.pop("labels")
        b = {k: v.to(DEVICE) for k, v in b.items()}
        out.append(torch.softmax(model(**b).logits.float(), -1).cpu())
    return torch.cat(out).numpy()


def main():
    a = get_args()
    set_seed(a.seed)
    data = load_splits()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.model_max_length = 512
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    if a.freeze:
        for n, p in model.named_parameters():
            if not n.startswith("classifier"):
                p.requires_grad = False
    if a.freeze_embeddings:
        for p in model.base_model.embeddings.parameters():
            p.requires_grad = False
    if a.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=a.lora_r, lora_alpha=2 * a.lora_r,
                                                 lora_dropout=0.1, target_modules=["query", "value"]))
    model.to(DEVICE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"{a.model}: trainable {n_train:,} / {n_total:,} params on {DEVICE}")

    texts = {s: list(data[s]["text_light"]) for s in data}
    labels = {s: [LABEL2ID[l] for l in data[s]["label"]] for s in data}
    tr = batches(tok, texts["train"], labels["train"], a.batch, a.max_len, True)
    dv = batches(tok, texts["dev"], labels["dev"], 64, a.max_len, False)
    te = batches(tok, texts["test"], labels["test"], 64, a.max_len, False)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=a.weight_decay)
    steps = len(tr) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * steps), steps)

    best_f1, best_state, bad, hist = -1, None, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot = 0
        for i, b in enumerate(tr):
            b = {k: v.to(DEVICE) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
            if i % 50 == 0:
                print(f"  ep{ep + 1} step {i}/{len(tr)} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
        pdev = predict(model, dv)
        f1 = compute_metrics(data["dev"]["label"], [LABELS[i] for i in pdev.argmax(1)])["macro_f1"]
        hist.append({"epoch": ep + 1, "train_loss": round(tot / len(tr), 4), "dev_macro_f1": f1,
                     "elapsed_s": round(time.time() - t0)})
        print(f"epoch {ep + 1}: train loss {tot / len(tr):.4f}  dev macro-F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)

    pdev, pte = predict(model, dv), predict(model, te)
    dev_lab = [LABELS[i] for i in pdev.argmax(1)]
    te_lab = [LABELS[i] for i in pte.argmax(1)]
    strategy = "lora" if a.lora else ("frozen_encoder" if a.freeze else
                                      "full_finetune_frozen_emb" if a.freeze_embeddings else "full_finetune")
    log_experiment(a.name, "transformer",
                   {"model": a.model, "strategy": strategy, "lr": a.lr, "epochs_max": a.epochs,
                    "batch": a.batch, "max_len": a.max_len, "device": DEVICE.type, "trainable_params": n_train, "total_params": n_total,
                    **({"lora_r": a.lora_r} if a.lora else {})},
                   data["dev"]["label"], dev_lab, data["test"]["label"], te_lab,
                   extra={"history": hist, "train_time_s": round(time.time() - t0)})
    save_predictions(a.name, data["test"], te_lab, pte)

    if a.save:
        out = MODELS_DIR / a.name
        if a.lora:
            model = model.merge_and_unload()
        model.save_pretrained(out)
        tok.save_pretrained(out)
        (out / "training_info.json").write_text(json.dumps({"base_model": a.model, "strategy": strategy,
                                                            "dev_macro_f1": best_f1, "args": vars(a)}, indent=2))
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
