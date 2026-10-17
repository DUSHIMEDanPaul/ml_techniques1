"""Export the final model to ONNX (+ int8 dynamic quantisation) for in-browser inference
with transformers.js, used by the static Hugging Face Space (deploy/static_space/).

Writes models/final_onnx/ laid out the way transformers.js expects:
  config.json, tokenizer.json, tokenizer_config.json
  onnx/model.onnx             (fp32, ~540 MB)
  onnx/model_quantized.onnx   (int8, ~140 MB - evaluated but NOT used: it loses ~9 macro-F1 points)
and checks the ONNX models against PyTorch on the test set (results/onnx_export.json).
The web page loads the fp32 model, which reproduces the PyTorch predictions.

Usage:  python deploy/export_onnx.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocess import light_clean  # noqa: E402

SRC = ROOT / "models" / "final"
OUT = ROOT / "models" / "final_onnx"


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


def run_onnx(path, tok, texts, bs=64):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=96, padding=True, return_tensors="np")
        out.append(sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                         "attention_mask": enc["attention_mask"].astype(np.int64)})[0])
    return np.concatenate(out).argmax(1)


def main():
    (OUT / "onnx").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForSequenceClassification.from_pretrained(SRC).eval()
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(SRC / f, OUT / f)

    # PyTorch reference predictions - computed BEFORE export (export modifies the module in place)
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv", keep_default_na=False)
    texts = [light_clean(t) for t in test["tweet"]]
    gold = test["label"].map(model.config.label2id).values
    with torch.no_grad():
        ref = np.concatenate([model(**tok(texts[i:i + 64], truncation=True, max_length=96, padding=True,
                                          return_tensors="pt")).logits.argmax(1).numpy()
                              for i in range(0, len(texts), 64)])
    report = {"pytorch_test_macro_f1": round(float(f1_score(gold, ref, average="macro")), 4)}

    enc = tok(["Ndishimye cyane uyu munsi"], return_tensors="pt")
    fp32 = OUT / "onnx" / "model.onnx"
    torch.onnx.export(Wrapper(model), (enc["input_ids"], enc["attention_mask"]), str(fp32),
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                    "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    q8 = OUT / "onnx" / "model_quantized.onnx"
    quantize_dynamic(str(fp32), str(q8), weight_type=QuantType.QInt8)

    for name, path in (("onnx_fp32", fp32), ("onnx_int8_dynamic", q8)):
        p = run_onnx(path, tok, texts)
        report[name] = {"size_mb": round(path.stat().st_size / 1e6), "agreement_with_pytorch": round(float(np.mean(p == ref)), 4),
                        "test_macro_f1": round(float(f1_score(gold, p, average="macro")), 4)}
    print(json.dumps(report, indent=2))
    (ROOT / "results" / "onnx_export.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
