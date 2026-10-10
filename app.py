"""Gradio web app - Kinyarwanda tweet sentiment classifier.

The fine-tuned Transformer is loaded from (in order of priority)
  1. the env var MODEL_ID  (a Hugging Face Hub repo id, e.g. "<user>/kinyarwanda-sentiment-afroxlmr")
  2. the local folder models/final  (created by src/finetune_transformer.py --save)

Run locally:  python app.py   ->  http://127.0.0.1:7860
"""
import os
import sys
from pathlib import Path

import gradio as gr
import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from preprocess import full_clean, light_clean  # noqa: E402

LABELS = ["negative", "neutral", "positive"]
MODEL_ID = os.environ.get("MODEL_ID") or str(ROOT / "models" / "final")
BASELINE_PATH = ROOT / "models" / "tfidf_baseline.joblib"
LOW_CONFIDENCE = 0.55

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
baseline = joblib.load(BASELINE_PATH) if BASELINE_PATH.exists() else None


@torch.no_grad()
def classify(text):
    if not text or not text.strip():
        return {}, "Please enter a sentence in Kinyarwanda.", {}
    enc = tokenizer(light_clean(text), truncation=True, max_length=96, return_tensors="pt")
    probs = torch.softmax(model(**enc).logits, -1)[0].tolist()
    id2label = model.config.id2label
    result = {id2label[i].lower(): float(p) for i, p in enumerate(probs)}
    top, conf = max(result.items(), key=lambda kv: kv[1])
    note = f"Prediction: **{top}** (confidence {conf:.0%})."
    if conf < LOW_CONFIDENCE:
        note += ("\n\n⚠️ Low confidence - the model is unsure. This often happens with sarcasm, "
                 "mixed sentiment, very short inputs or text that is not Kinyarwanda.")
    base = {}
    if baseline is not None:
        bp = baseline.predict_proba([full_clean(text)])[0]
        base = {c: float(p) for c, p in zip(baseline.classes_, bp)}
    return result, note, base


EXAMPLES = [
    "Ndishimye cyane! Imana ishimwe ku bw'ibyiza yankoreye 🙏",
    "Uyu mukino wari mwiza cyane, abakinnyi bacu bitwaye neza",
    "Ibi bintu birambabaje cyane, ntabwo byari bikwiye na gato",
    "Inama izabera i Kigali ku wa gatatu saa tatu za mu gitondo",
    "Ese ubu koko ibi ni ibiki? Ni agahomamunwa 😡",
    "Wowe uri umuhanga pe 😂😂",
]

with gr.Blocks(title="Kinyarwanda Sentiment") as demo:
    gr.Markdown(
        "# Kinyarwanda Tweet Sentiment Classifier\n"
        "Type a sentence or tweet in **Kinyarwanda** and the model predicts whether it is "
        "*negative*, *neutral* or *positive*. The model is a pretrained African-language Transformer "
        "fine-tuned on the AfriSenti Kinyarwanda Twitter dataset (SemEval-2023 Task 12).")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Kinyarwanda text", lines=4, placeholder="Andika hano...")
            btn = gr.Button("Classify", variant="primary")
            gr.Examples(EXAMPLES, inp)
        with gr.Column():
            out = gr.Label(label="Fine-tuned Transformer", num_top_classes=3)
            note = gr.Markdown()
            base = gr.Label(label="Baseline (TF-IDF char n-grams + Logistic Regression)", num_top_classes=3)
    gr.Markdown("Limitations: trained on ~3.2k tweets only; sarcasm, code-mixing with English/French, "
                "and text outside the Twitter domain are frequent sources of error. "
                "Not suitable for high-stakes decisions.")
    btn.click(classify, inp, [out, note, base])
    inp.submit(classify, inp, [out, note, base])

if __name__ == "__main__":
    demo.launch()
