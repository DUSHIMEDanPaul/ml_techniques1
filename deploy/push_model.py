"""Push the final fine-tuned model + tokenizer to the Hugging Face Hub.

Log in first (once):   hf auth login
Then run:              python deploy/push_model.py
"""
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "final"
repo_name = "DanpaulDUSHIME/kinyarwanda-sentiment-afroxlmr"

# Load the fine-tuned model and tokenizer (AfroXLMR-small, selected on dev macro-F1)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Push to the Hugging Face repository
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"Model pushed: https://huggingface.co/{repo_name}")
